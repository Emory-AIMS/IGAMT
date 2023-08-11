import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from utils.train_utils import to_var, sample_start_feature_time_mask_gender_race, sample_mask_from_prob
from utils.train_utils import ehr_model_inference as model_inference, extract_incr_time_from_tempo_step

from pyvacy import optim, analysis
from pyvacy.optim.dp_optimizer import DPAdam, DPSGD
import pyvacy.analysis.moments_accountant as moments_accountant

from nn.transformers.mixed_embedding_transformer import Transformer, GeneralTransformerDecoder
from nn.generator import MLP_Generator
from nn.discriminator import MLP_Discriminator, CNN_Discriminator, CNN_Auxiliary_Discriminator, CNN_Net


lin_params_size =  4

def train_model(args, datasets, prob_mask, **kwargs):
    if not args.test:
        # model define
        if args.load_model:
            model_path = os.path.join(args.model_path, args.pretrained_model_filename)
            models = load_model(model_path)
            Trans = models["Trans"]
            Dx = models["Dx"]
            Dm = models["Dm"]
            G = models["G"]
            Dz = models["Dz"]
            Imi = models["Imi"]
            
        else:
            dim_time=int((kwargs["time_shift"]+kwargs["time_scale"])*1.5)
            Trans = Transformer(
                num_encoder_layers=args.num_encoder_layers, #6 #1
                num_decoder_layers=args.num_decoder_layers, #6 #1
                max_length=args.max_length,
                dim_feature=args.feature_size,
                dim_model=args.latent_size, #512 #128
                dim_time=dim_time, # 
                num_heads=args.num_heads, #6 #3
                dim_feedforward=args.hidden_size, #2048 #128
                encoder_dropout=args.encoder_dropout,
                decoder_dropout=args.decoder_dropout,
                use_prob_mask=args.use_prob_mask,
                dim_gender = 2,
                dim_race = 3
                )

            Dx = CNN_Auxiliary_Discriminator(
                feature_size=args.feature_size,
                feature_dropout=args.feature_dropout,
                filter_size=args.filter_size,
                window_sizes=args.window_sizes,
                use_spectral_norm = args.use_spectral_norm
                )
            
            # for mask
            Dm = CNN_Discriminator(
                feature_size=args.feature_size,
                feature_dropout=args.feature_dropout,
                filter_size=args.filter_size,
                window_sizes=args.window_sizes,
                use_spectral_norm = args.use_spectral_norm
                )

            G = MLP_Generator(
                input_size=args.noise_size,
                output_size=args.latent_size,
                archs=args.gmlp_archs
                )

            Dz = CNN_Discriminator(
                feature_size=args.latent_size,#*2,
                feature_dropout=args.feature_dropout,
                filter_size=args.filter_size,
                window_sizes=args.dz_window_sizes,
                use_spectral_norm = args.use_spectral_norm
                )
            # imi
            Imi = GeneralTransformerDecoder(
                time_embedding=None,
                num_layers=args.num_decoder_layers,
                max_length=args.max_length,
                dim_feature=args.feature_size,
                dim_model=args.latent_size,
                dim_time=dim_time,
                num_heads=args.num_heads,
                dim_feedforward=args.hidden_size,
                dropout=args.decoder_dropout,
                use_prob_mask=args.use_prob_mask,
                linear=Trans.decoder.linear,
                linear_m=Trans.decoder.linear_m,
                dim_gender = 2,
                dim_race = 3
            )
            

        if torch.cuda.is_available():
            Trans = Trans.cuda()
            Dx = Dx.cuda()
            Dm = Dm.cuda()
            G = G.cuda()
            Dz = Dz.cuda()
            Imi = Imi.cuda()

        
        
        opt_enc = torch.optim.Adam(Trans.encoder.parameters(), lr=args.enc_learning_rate)
        opt_dec = torch.optim.Adam(Trans.decoder.parameters(), lr=args.dec_learning_rate)
        opt_dx = torch.optim.Adam(Dx.parameters(), lr=args.dx_learning_rate)
        opt_dm = torch.optim.Adam(Dm.parameters(), lr=args.dm_learning_rate)
        opt_dz = torch.optim.Adam(Dz.parameters(), lr=args.dz_learning_rate)
        opt_gen = torch.optim.Adam(G.parameters(), lr=args.g_learning_rate)
        opt_imi = torch.optim.Adam(Imi.parameters(), lr=args.imi_learning_rate)
        #opt_imi = torch.optim.Adam(filter(lambda p: p.requires_grad, Imi.parameters()), lr=args.imi_learning_rate)
        #
        if args.dp_sgd == True: # opt_dx and opt_dz access origin data too?
            opt_dec = torch.optim.Adam(list(Trans.decoder.parameters())[:-lin_params_size], lr=args.dec_learning_rate)
            opt_lin = DPSGD(params=list(Trans.decoder.parameters())[-lin_params_size:], lr=args.dec_learning_rate, minibatch_size=args.batch_size, microbatch_size=args.batch_size,
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            opt_dx = DPSGD(params=Dx.parameters(), lr=args.dx_learning_rate, minibatch_size=args.batch_size, microbatch_size=args.batch_size,
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            opt_dm = DPSGD(params=Dm.parameters(), lr=args.dm_learning_rate, minibatch_size=args.batch_size, microbatch_size=args.batch_size,
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            opt_dz = DPSGD(params=Dz.parameters(), lr=args.dz_learning_rate, minibatch_size=args.batch_size, microbatch_size=args.batch_size, 
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            opt_imi = torch.optim.Adam(list(Imi.parameters())[:-lin_params_size], lr=args.imi_learning_rate)
            
            epsilon = moments_accountant.epsilon(len(datasets['train'].data), args.batch_size, args.noise_multiplier, args.epochs, args.delta)

            print('Training procedure satisfies (%f, %f)-DP' % (epsilon, args.delta)) # ?? question, why 2 epsilon?


            lr_lin = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_lin, gamma=args.dec_lr_decay_rate)


        lr_enc = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_enc, gamma=args.enc_lr_decay_rate)
        lr_dec = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dec, gamma=args.dec_lr_decay_rate)
        lr_dx = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dx, gamma=args.dx_lr_decay_rate)
        lr_dm = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dm, gamma=args.dm_lr_decay_rate)
        lr_dz = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dz, gamma=args.dz_lr_decay_rate)
        lr_gen = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_gen, gamma=args.g_lr_decay_rate)
        lr_imi = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_imi, gamma=args.imi_lr_decay_rate)

        
        tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        models = {
                "Trans": Trans,
                "Dx": Dx,
                "Dm": Dm,
                "G": G,
                "Dz": Dz,
                "Imi": Imi
            } 
        
        opts = {
            "enc": opt_enc,
            "dec": opt_dec,
            "dx": opt_dx,
            "dm": opt_dm,
            "dz": opt_dz,
            "gen": opt_gen,
            "imi": opt_imi
        }
        lrs = {
            "enc": lr_enc,
            "dec": lr_dec,
            "dx": lr_dx,
            "dm": lr_dm,
            "dz": lr_dz,
            "gen": lr_gen,
            "imi": lr_imi
        }

        if args.dp_sgd == True: 
            opts["lin"] = opt_lin
            lrs["lin"] = lr_lin
        

        min_valid_loss = float("inf")
        min_valid_path = ""
        for epoch in range(args.epochs):

            print("Epoch\t%02d/%i"%(epoch, args.epochs))
            
            data_loader = DataLoader(
                dataset=datasets["train"],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )
        
            log_file = os.path.join(args.result_path, args.train_log)
            _, models = model_evaluation(args, models, opts, lrs, data_loader, prob_mask, "train", log_file, **kwargs)

            torch.cuda.empty_cache()
            if epoch % args.valid_eval_freq == 0:
                del data_loader
                torch.cuda.empty_cache()
                data_loader = DataLoader(
                    dataset=datasets["valid"],
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=cpu_count(),
                    pin_memory=torch.cuda.is_available()
                )
            
                print("Validation:")
                log_file = os.path.join(args.result_path, args.valid_log)
                valid_loss, models = model_evaluation(args, models, opts, lrs, data_loader, prob_mask, "valid", log_file, **kwargs)
                print("****************************************************")
                print()
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    path = "{}/model_vloss_{}".format(args.model_path, valid_loss)
                    min_valid_path = path

                    save_model(models, path)

            
        # Generate the synthetic sequences as many as you want 
        
        model_path = min_valid_path
    else:
        model_path = os.path.join(args.model_path, args.test_model_filename)
    
    models = load_model(model_path)
    Imi = models["Imi"]
    G = models["G"]
    Imi.eval()
    G.eval()
    model_generation(args, G, Imi, prob_mask, **kwargs)
    #
    Dec = models["Trans"].decoder
    Dec.eval()
    model_generation(args, G, Dec, prob_mask, prefix=args.model_type+"_pub", **kwargs)


def model_generation(args, G_0, G_1, prob_mask, path=None, prefix=None, **kwargs,):
    if path is None:
        path = args.result_path
    if prefix is None:
        prefix = args.model_type
    gen_zs, gen_xs, gen_ms = [], [], []
    for i in range(args.gendata_size//args.batch_size):
        zgen = G_0(batch_size=args.batch_size*args.max_length)
        zgen = torch.reshape(zgen, (args.batch_size, args.max_length, -1))
        Pimi, Mimi = model_inference(args, G_1, zgen, prob_mask, **kwargs)
        
        gen_zs.append(zgen)
        gen_xs.append(Pimi)
        gen_ms.append(Mimi)

    gen_zlist = torch.cat(gen_zs).cpu().detach().numpy()
    gen_xlist = torch.cat(gen_xs).cpu().detach().numpy()
    
    np.save(os.path.join(path, '{}_generated_codes.npy'.format(prefix)), gen_zlist)
    np.save(os.path.join(path, '{}_generated_patients.npy'.format(prefix)), gen_xlist) 
    
    if not args.no_mask and not args.use_prob_mask:
        gen_mlist = torch.cat(gen_ms).cpu().detach().numpy()
        np.save(os.path.join(path, '{}_generated_masks.npy'.format(prefix)), gen_mlist)




def save_model(models, path):
    Trans = models["Trans"]
    Dx = models["Dx"]
    Dm = models["Dm"]
    G = models["G"]
    Dz = models["Dz"]
    Imi = models["Imi"]

    torch.save(Trans, "{}_Trans".format(path))
    torch.save(Dx, "{}_Dx".format(path))
    torch.save(Dm, "{}_Dm".format(path))
    torch.save(G, "{}_G".format(path))
    torch.save(Dz, "{}_Dz".format(path))
    torch.save(Imi, "{}_Imi".format(path))


def load_model(path):
    Trans = torch.load("{}_Trans".format(path))
    Dx = torch.load("{}_Dx".format(path))
    Dm = torch.load("{}_Dm".format(path))
    G = torch.load("{}_G".format(path))
    Dz = torch.load("{}_Dz".format(path))
    Imi = torch.load("{}_Imi".format(path))

    models = {
        "Trans": Trans,
        "Dx": Dx,
        "Dm": Dm,
        "G": G,
        "Dz": Dz,
        "Imi": Imi
    }
    return models


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def defreeze_params(params):
    for param in params:
        param.requires_grad = True


def model_evaluation(args, models, opts, lrs, data_loader, prob_mask, split, log_file, **kwargs):
    Trans = models["Trans"]
    Dx = models["Dx"]
    Dm = models["Dm"]
    G = models["G"]
    Dz = models["Dz"]
    Imi = models["Imi"]

    if split == 'train':
        # opts
        opt_enc = opts["enc"]
        opt_dec = opts["dec"]
        opt_dx = opts["dx"]
        opt_dm = opts["dm"]
        opt_dz = opts["dz"]
        opt_gen = opts["gen"]
        opt_imi = opts["imi"]

        # lr scheduler
        lr_enc = lrs["enc"]
        lr_dec = lrs["dec"]
        lr_dx = lrs["dx"]
        lr_dm = lrs["dm"]
        lr_dz = lrs["dz"]
        lr_gen = lrs["gen"]
        lr_imi = lrs["imi"]

        if args.dp_sgd == True: 
            opt_lin = opts["lin"]
            lr_lin = lrs["lin"]

    
    recon_total_loss, mask_total_loss = 0.0, 0.0
    xCritic_total_loss, zCritic_total_loss, mCritic_total_loss = 0.0, 0.0, 0.0
    input_match_total_loss, output_match_total_loss, gen_match_total_loss = 0.0, 0.0, 0.0
    input_m_match_total_loss, output_m_match_total_loss, gen_m_match_total_loss = 0.0, 0.0, 0.0

    if split == 'train':
        Trans.encoder_dropout=args.encoder_dropout
        Trans.decoder_dropout=args.decoder_dropout
        Trans.train()
        Dx.train()
        Dm.train()
        G.train()
        Dz.train()
        Imi.train()
    else:
        Trans.encoder_dropout=0.0
        Trans.decoder_dropout=0.0
        Trans.eval()
        Dx.eval()
        Dm.eval()
        G.eval()
        Dz.eval()
        Imi.eval()

    for iteration, batch in enumerate(data_loader):
        #
        batch_size = batch['src_tempo'].shape[0]
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = to_var(v)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1

        if torch.cuda.is_available():
            one = one.cuda()
            mone = mone.cuda()
        
        #import pdb; pdb.set_trace()
        # Step 0: Evaluate current loss
        
        #print("max src_time", torch.amax(batch['src_time'], [0,1]), " -- min src_time", torch.amin(batch['src_time'], [0,1]))
        #print("max tgt_time", torch.amax(batch['tgt_time'], [0,1]), " -- min tgt_time", torch.amin(batch['tgt_time'], [0,1]))
        src_tempo = batch['src_tempo']; tgt_tempo = batch['tgt_tempo']
        src_time = batch['src_time']; tgt_time = batch['tgt_time']
        gender = batch['gender']
        race = batch['race']
        src_mask = batch['src_mask']; tgt_mask = batch['tgt_mask']
        src_ava = batch['src_ava']; tgt_ava = batch['tgt_ava']

        #import pdb; pdb.set_trace()
        if args.no_mask:
            z, Pinput, Poutput, Toutput, Moutput = Trans(src_tempo, tgt_tempo, src_time, tgt_time, None, None, src_ava, tgt_ava,
                                                         gender=gender, race=race)
            # loss
            recon_loss = args.beta_recon * Trans.compute_recon_loss(Poutput, tgt_tempo, None, None)
        elif args.use_prob_mask:
            z, Pinput, Poutput, Toutput, Moutput = Trans(src_tempo, tgt_tempo, src_time, tgt_time, src_mask, tgt_mask, src_ava, tgt_ava,
                                                         gender=gender, race=race)
            output_mask = sample_mask_from_prob(prob_mask, tgt_mask.shape[0], tgt_mask.shape[1])
            # loss
            recon_loss = args.beta_recon * Trans.compute_recon_loss(Poutput, tgt_tempo, output_mask, tgt_mask)
        else:
            z, Pinput, Poutput, Toutput, Moutput = Trans(src_tempo, tgt_tempo, src_time, tgt_time, src_mask, tgt_mask, src_ava, tgt_ava,
                                                         gender=gender, race=race)
            # loss
            recon_loss = args.beta_recon * Trans.compute_recon_loss(Poutput, tgt_tempo, Moutput, tgt_mask)
            mask_loss = args.beta_mask * Trans.compute_mask_loss(Moutput, tgt_mask)

        zgen = G(batch_size=z.size(0)*args.max_length)
        zgen = torch.reshape(zgen, (z.size(0), args.max_length, -1))
        # make up start feature
        '''
        start_feature, start_time, start_mask = sample_start_feature_time_mask(z.size(0))
        sampled_gender, sampled_race = sample_gender_race(z.size(0))
        '''
        start_feature, start_time, start_mask, sampled_gender, sampled_race = sample_start_feature_time_mask_gender_race(z.size(0))
        kwargs["start_time"] = start_time
        kwargs["gender"] = sampled_gender
        kwargs["race"] = sampled_race
        kwargs["extract_incr_time_from_tempo_step"] = extract_incr_time_from_tempo_step
        if args.no_mask:
            Pgen, _, Mgen = Trans.decoder.inference(start_feature=start_feature, start_mask=None, memory=zgen, **kwargs)
            Pimi, _, Mimi = Imi.inference(start_feature=start_feature, start_mask=None, memory=zgen, **kwargs)
        elif args.use_prob_mask:
            Pgen, _, Mgen = Trans.decoder.inference(start_feature=start_feature, start_mask=start_mask, prob_mask=prob_mask, memory=zgen, **kwargs)
            Pimi, _, Mimi = Imi.inference(start_feature=start_feature, start_mask=start_mask, prob_mask=prob_mask, memory=zgen, **kwargs)
        else:
            Pgen, _, Mgen = Trans.decoder.inference(start_feature=start_feature, start_mask=start_mask, memory=zgen, **kwargs)
            Pimi, _, Mimi = Imi.inference(start_feature=start_feature, start_mask=start_mask, memory=zgen, **kwargs)

        
        if split == 'train':
            if iteration % args.gen_freq_base < args.gen_freq_hit:
                if iteration % args.critic_freq_base < args.critic_freq_hit:
                    # Step 1: Update the Critic_x
                    params = list(Dx.parameters())
                    # generated data
                    opt_dx.zero_grad()
                    Dinput = Dx(tgt_tempo)
                    Doutput = Dx(Poutput)
                    Dinput = Dinput.mean()
                    Doutput = Doutput.mean()
                    Dinput.backward(mone, inputs=params, retain_graph=True) # maximize
                    Doutput.backward(one, inputs=params, retain_graph=True) # minimize
                    Dx.cal_gradient_penalty(tgt_tempo[:, :Poutput.size(1), :], Poutput, tgt_mask, Moutput).backward(inputs=params, retain_graph=True)
                    opt_dx.step()

                    opt_dx.zero_grad()
                    Dinput = Dx(tgt_tempo)
                    Dgen = Dx(Pgen)
                    Dinput = Dinput.mean()
                    Dgen = Dgen.mean()
                    Dinput.backward(mone, inputs=params, retain_graph=True)
                    Dgen.backward(one, inputs=params, retain_graph=True)
                    Dx.cal_gradient_penalty(tgt_tempo[:, :Pgen.size(1), :], Pgen, tgt_mask, Mgen).backward(inputs=params, retain_graph=True)
                    opt_dx.step()

                    
                    opt_dx.zero_grad()
                    Dinput = Dx(tgt_tempo)
                    Dimi = Dx(Pimi)
                    Dinput = Dinput.mean()
                    Dimi = Dimi.mean()
                    Dinput.backward(mone, inputs=params, retain_graph=True)
                    Dimi.backward(one, inputs=params, retain_graph=True)
                    Dx.cal_gradient_penalty(tgt_tempo[:, :Pimi.size(1), :], Pimi, tgt_mask, Mimi).backward(inputs=params, retain_graph=True)
                    opt_dx.step()
            

                    # Step 2: Update Critic_m
                    params = list(Dm.parameters())
                    opt_dm.zero_grad()
                    Dminput, Dmoutput = Dm(tgt_mask).mean(), Dm(Moutput).mean()
                    Dminput.backward(mone, inputs=params, retain_graph=True)
                    Dmoutput.backward(one, inputs=params, retain_graph=True)
                    Dm.cal_gradient_penalty(tgt_mask[:, :Moutput.size(1), :], Moutput).backward(inputs=params, retain_graph=True)
                    opt_dm.step()

                    opt_dm.zero_grad()
                    Dminput, Dmgen = Dm(tgt_mask).mean(), Dm(Mgen).mean()
                    Dminput.backward(mone, inputs=params, retain_graph=True)
                    Dmgen.backward(one, inputs=params, retain_graph=True)
                    Dm.cal_gradient_penalty(tgt_mask[:, :Mgen.size(1), :], Mgen).backward(inputs=params, retain_graph=True)
                    opt_dm.step()

                    
                    opt_dm.zero_grad()
                    Dminput, Dmimi = Dm(tgt_mask).mean(), Dm(Mimi).mean()
                    Dminput.backward(mone, inputs=params, retain_graph=True)
                    Dmimi.backward(one, inputs=params, retain_graph=True)
                    Dm.cal_gradient_penalty(tgt_mask[:, :Mimi.size(1), :], Mimi).backward(inputs=params, retain_graph=True)
                    opt_dm.step()

                        
                    # Step 3: Update the Critic_z
                    params = list(Dz.parameters())
                    opt_dz.zero_grad()
                    Dreal, Dfake = Dz(z).mean(), Dz(zgen).mean()
                    Dreal.backward(mone, inputs=params, retain_graph=True)
                    Dfake.backward(one, inputs=params, retain_graph=True)
                    Dz.cal_gradient_penalty(z, zgen).backward(inputs=params)
                    opt_dz.step()

            

                # Step 4, 5: Update the Decoder and the Encoder
                if args.dp_sgd:
                    params = list(Trans.parameters())[:-lin_params_size]
                else:
                    params = list(Trans.parameters())
                
                opt_dec.zero_grad()
                Doutput = Dx(Poutput, Moutput)
                Dgen =  Dx(Pgen, Mgen)
                Doutput = Doutput.mean()
                Dgen = Dgen.mean()
                Doutput.backward(mone, inputs=params, retain_graph=True)
                Dgen.backward(mone, inputs=params, retain_graph=True)
                # mask
                Dmoutput, Dmgen = Dm(Moutput).mean(), Dm(Mgen).mean()
                Dmoutput.backward(mone, inputs=params, retain_graph=True)
                Dmgen.backward(mone, inputs=params, retain_graph=True)

                opt_enc.zero_grad()

                Dreal = Dz(z).mean()
                Dreal.backward(one, inputs=params, retain_graph=True)
                
                if args.no_recon == False:
                    recon_loss.backward(inputs=params, retain_graph=True)
                    if not args.no_mask and not args.use_prob_mask:
                        mask_loss.backward(inputs=params, retain_graph=True)

                opt_dec.step()
                opt_enc.step()
            
                if args.dp_sgd:
                    params = list(Trans.parameters())[-lin_params_size:]

                    opt_lin.zero_grad()
                    Doutput = Dx(Poutput, Moutput)
                    Dgen =  Dx(Pgen, Mgen)
                    Doutput = Doutput.mean()
                    Dgen = Dgen.mean()
                    Doutput.backward(mone, inputs=params, retain_graph=True)
                    Dgen.backward(mone, inputs=params, retain_graph=True)
                    # mask
                    Dmoutput, Dmgen = Dm(Moutput).mean(), Dm(Mgen).mean()
                    Dmoutput.backward(mone, inputs=params, retain_graph=True)
                    Dmgen.backward(mone, inputs=params, retain_graph=True)

                    if args.no_recon == False:
                        recon_loss.backward(inputs=params, retain_graph=True)
                        if not args.no_mask and not args.use_prob_mask:
                            mask_loss.backward(inputs=params, retain_graph=True)
                    
                    opt_lin.step()


                # step 6: Update Imi
                opt_imi.zero_grad()
                #import pdb; pdb.set_trace()
                params = list(Imi.parameters())[:-lin_params_size]
                
                Pimi, _, Mimi = Imi.inference(start_feature=start_feature, start_mask=start_mask, memory=zgen, **kwargs)
                #
                #freeze_params(params)
                Dimi =  Dx(Pimi, Mimi)
                Dimi = Dimi.mean()
                Dimi.backward(mone, inputs=params, retain_graph=True)
                # mask
                Dmimi =  Dm(Mimi).mean()
                Dmimi.backward(mone, inputs=params, retain_graph=True)
                opt_imi.step()  
                
                # match
                Pimi, _, Mimi = Imi.inference(start_feature=start_feature, start_mask=start_mask, memory=zgen, **kwargs)
                z, Pinput, Poutput, Toutput, Moutput = Trans(src_tempo, tgt_tempo, src_time, tgt_time, src_mask, tgt_mask, src_ava, tgt_ava,
                                                             gender=gender, race=race)
                gen_match_loss = args.beta_match_g*Imi.compute_recon_loss(Pimi, Pgen, Mimi, Mgen, type="mse")
                input_match_loss = args.beta_match_i*Imi.compute_recon_loss(Pimi, tgt_tempo, Mimi, tgt_mask, type="mse")
                
                # mask match
                gen_m_match_loss = args.beta_match_g*Imi.compute_mask_loss(Mimi, Mgen, type="mse")
                input_m_match_loss = args.beta_match_i*Imi.compute_mask_loss(Mimi, tgt_mask, type="mse")

                opt_imi.zero_grad()
                # match
                gen_match_loss.backward(inputs=params, retain_graph=True)
                # mask match
                gen_m_match_loss.backward(inputs=params, retain_graph=True)            
                #opt_imi.step()

                #opt_imi.zero_grad()
                # match
                input_match_loss.backward(inputs=params, retain_graph=True)
                # mask match
                input_m_match_loss.backward(inputs=params, retain_graph=True)            
                #opt_imi.step()

                output_match_loss = args.beta_match_o*Imi.compute_recon_loss(Pimi, Poutput, Mimi, Moutput, type="mse")
                output_m_match_loss = args.beta_match_o*Imi.compute_mask_loss(Mimi, Moutput, type="mse")
                #opt_imi.zero_grad()
                output_match_loss.backward(inputs=params, retain_graph=True)
                output_m_match_loss.backward(inputs=params, retain_graph=True)
                
                opt_imi.step()   
                
                #defreeze_params(params)
                #


            # Step 6: Update the Generator
            params = list(G.parameters())
            opt_gen.zero_grad()
            Dfake = Dz(zgen).mean()
            Dfake.backward(mone, inputs=params, retain_graph=True)
            opt_gen.step()


        
        else:
            #import pdb; pdb.set_trace()
            # only for tempo data without mask
            Dinput = Dx(tgt_tempo)
            Doutput = Dx(Poutput)
            Dgen = Dx(Pgen)
            Dimi= Dx(Pimi)
            #
            Dinput = Dinput.mean()
            Doutput = Doutput.mean()
            Dgen = Dgen.mean()
            Dimi = Dimi.mean()

            # reshape z, zgen
            #z = torch.reshape(z, (-1, z.size(-1)))
            #zgen = torch.reshape(zgen, (-1, zgen.size(-1)))
            Dreal, Dfake = Dz(z).mean(), Dz(zgen).mean()

            #
            Dminput, Dmoutput, Dmgen, Dmimi = Dm(tgt_mask).mean(), Dm(Moutput).mean(), Dm(Mgen).mean(), Dm(Mimi).mean()

            # match
            gen_match_loss = args.beta_match_g*Imi.compute_recon_loss(Pimi, Pgen, Mimi, Mgen, type="mse")
            input_match_loss = args.beta_match_i*Imi.compute_recon_loss(Pimi, tgt_tempo, Mimi, tgt_mask, type="mse")
            
            # mask match
            gen_m_match_loss = args.beta_match_g*Imi.compute_mask_loss(Mimi, Mgen, type="mse")
            input_m_match_loss = args.beta_match_i*Imi.compute_mask_loss(Mimi, tgt_mask, type="mse")
            #
            output_match_loss = args.beta_match_o*Imi.compute_recon_loss(Pimi, Poutput, Mimi, Moutput, type="mse")
            output_m_match_loss = args.beta_match_o*Imi.compute_mask_loss(Mimi, Moutput, type="mse")
            #
            
            
        xCritic_loss = (- Dinput + (Doutput + Dgen + Dimi) / 3.0).data
        zCritic_loss = (- Dreal + Dfake).data
        mCritic_loss = (- Dminput + (Dmoutput + Dmgen + Dmimi) /3.0).data
        
        #
        recon_loss = recon_loss.data
        if not args.no_mask and not args.use_prob_mask:
            mask_loss = mask_loss.data
        else:
            mask_loss = 0.0
        
        if split != "train":
            recon_total_loss += recon_loss
            if not args.no_mask and not args.use_prob_mask:
                mask_total_loss += mask_loss
            else:
                mask_total_loss = 0.0
            xCritic_total_loss += xCritic_loss
            zCritic_total_loss += zCritic_loss
            mCritic_total_loss += mCritic_loss
            input_match_total_loss += input_match_loss
            input_m_match_total_loss += input_m_match_loss
            output_match_total_loss += output_match_loss
            output_m_match_total_loss += output_m_match_loss
            gen_match_total_loss += gen_match_loss
            gen_m_match_total_loss += gen_m_match_loss

        if split == 'train' and iteration % args.train_eval_freq == 0:
            # print the losses for each epoch
            print("Learning rate:\t%2.8f"%(lr_gen.get_last_lr()[0]))
            print("Batch loss:")
            print("%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tmCritic_loss\t%9.4f\n\tinput_match_loss\t%9.4f\toutput_match_loss\t%9.4f\tgen_match_loss\t%9.4f\n\tinput_mask_match_loss\t%9.4f\toutput_mask_match_loss\t%9.4f\timi_mask_match_loss\t%9.4f"%(
                    split.upper(), recon_loss, mask_loss, xCritic_loss, zCritic_loss, mCritic_loss,
                    input_match_loss, output_match_loss, gen_match_loss, 
                    input_m_match_loss, output_m_match_loss, gen_m_match_loss))
            print()
            with open(log_file, "a+") as file:
                file.write("Learning rate:\t%2.8f\n"%(lr_gen.get_last_lr()[0]))
                file.write("Batch loss:\n")
                file.write("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tmCritic_loss\t%9.4f\n\tinput_match_loss\t%9.4f\toutput_match_loss\t%9.4f\tgen_match_loss\t%9.4f\n\tinput_mask_match_loss\t%9.4f\toutput_mask_match_loss\t%9.4f\timi_mask_match_loss\t%9.4f\n"%(
                    split.upper(), recon_loss, mask_loss, xCritic_loss, zCritic_loss, mCritic_loss,
                    input_match_loss, output_match_loss, gen_match_loss, 
                    input_m_match_loss, output_m_match_loss, gen_m_match_loss))
                file.write("===================================================\n")
    #
    # print the losses for each epoch
    if split == 'train':
        print("Learning rate:\t%2.8f"%(lr_gen.get_last_lr()[0]))
    print("Batch loss:")
    print("%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tmCritic_loss\t%9.4f\n\tinput_match_loss\t%9.4f\toutput_match_loss\t%9.4f\tgen_match_loss\t%9.4f\n\tinput_mask_match_loss\t%9.4f\toutput_mask_match_loss\t%9.4f\timi_mask_match_loss\t%9.4f"%(
            split.upper(), recon_loss, mask_loss, xCritic_loss, zCritic_loss, mCritic_loss,
            input_match_loss, output_match_loss, gen_match_loss, 
            input_m_match_loss, output_m_match_loss, gen_m_match_loss))
    if split != "train":
        print("Accumulated loss:")
        print("%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tmCritic_loss\t%9.4f\n\tinput_match_loss\t%9.4f\toutput_match_loss\t%9.4f\tgen_match_loss\t%9.4f\n\tinput_mask_match_loss\t%9.4f\toutput_mask_match_loss\t%9.4f\timi_mask_match_loss\t%9.4f"%(
                split.upper(), recon_total_loss/iteration, mask_total_loss/iteration, xCritic_total_loss/iteration, zCritic_total_loss/iteration, mCritic_total_loss/iteration,
                input_match_total_loss/iteration, output_match_total_loss/iteration, gen_match_total_loss/iteration, 
                input_m_match_total_loss/iteration, output_m_match_total_loss/iteration, gen_m_match_total_loss/iteration))
    print()
    with open(log_file, "a+") as file:
        if split == 'train':
            file.write("Learning rate:\t%2.8f\n"%(lr_gen.get_last_lr()[0]))
        file.write("Batch loss:\n")
        file.write("%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tmCritic_loss\t%9.4f\n\tinput_match_loss\t%9.4f\toutput_match_loss\t%9.4f\tgen_match_loss\t%9.4f\n\tinput_mask_match_loss\t%9.4f\toutput_mask_match_loss\t%9.4f\timi_mask_match_loss\t%9.4f\n"%(
            split.upper(), recon_loss, mask_loss, xCritic_loss, zCritic_loss, mCritic_loss,
            input_match_loss, output_match_loss, gen_match_loss, 
            input_m_match_loss, output_m_match_loss, gen_m_match_loss))
        if split != "train":
            file.write("Accumulated loss:\n")
            file.write("%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f\tmCritic_loss\t%9.4f\n\tinput_match_loss\t%9.4f\toutput_match_loss\t%9.4f\tgen_match_loss\t%9.4f\n\tinput_mask_match_loss\t%9.4f\toutput_mask_match_loss\t%9.4f\timi_mask_match_loss\t%9.4f\n"%(
                split.upper(), recon_total_loss/iteration, mask_total_loss/iteration, xCritic_total_loss/iteration, zCritic_total_loss/iteration, mCritic_total_loss/iteration,
                input_match_total_loss/iteration, output_match_total_loss/iteration, gen_match_total_loss/iteration, 
                input_m_match_total_loss/iteration, output_m_match_total_loss/iteration, gen_m_match_total_loss/iteration))
        file.write("===================================================\n")
    
    if split == 'train':
        lr_enc.step()
        lr_dec.step()
        lr_dx.step()
        lr_dm.step() # lr_dx.step()
        lr_dz.step()
        lr_gen.step()
        lr_imi.step()
    
    models = {
        "Trans": Trans,
        "Dx": Dx,
        "Dm": Dm,
        "G": G,
        "Dz": Dz,
        "Imi": Imi
    }

    return recon_total_loss/iteration, models