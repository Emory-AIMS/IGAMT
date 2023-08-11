import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from utils.train_utils import to_var, ed_sample_start_feature_mask as sample_start_feature_mask, sample_mask_from_prob
from utils.train_utils import ed_model_generation as model_generation, ed_model_inference as model_inference, ed_extract_incr_time_from_tempo_step as extract_incr_time_from_tempo_step


from pyvacy import optim, analysis
from pyvacy.optim.dp_optimizer import DPAdam, DPSGD
import pyvacy.analysis.moments_accountant as moments_accountant

from nn.seq2seq_ae import Seq2seq_Autoencoder
from nn.generator import MLP_Generator
from nn.discriminator import MLP_Discriminator, CNN_Discriminator


def train_model(args, datasets, prob_mask, **kwargs):
    if not args.test:
        if args.load_model:
            model_path = os.path.join(args.model_path, args.pretrained_model_filename)
            models = load_dgat(model_path)
            AE = models["AE"]
            G = models["G"]
            Dz = models["Dz"]
            
        else:
            # model define
            AE = Seq2seq_Autoencoder(
                max_length=args.max_length,
                rnn_type=args.rnn_type,
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                latent_size=args.latent_size,
                encoder_dropout=args.encoder_dropout,
                decoder_dropout=args.decoder_dropout,
                num_layers=args.num_layers,
                bidirectional=args.bidirectional,
                use_prob_mask=args.use_prob_mask
                )

            G = MLP_Generator(
                input_size=args.noise_size,
                output_size=args.latent_size,
                archs=args.gmlp_archs
                )

            latent_size = args.latent_size*2 if args.bidirectional else args.latent_size
            Dz = MLP_Discriminator(
                input_size=latent_size,
                output_size=1,
                archs=args.dmlp_archs,
                use_spectral_norm = args.use_spectral_norm
                ) 

        if torch.cuda.is_available():
            AE = AE.cuda()
            G = G.cuda()
            Dz = Dz.cuda()
        
        

        opt_enc = torch.optim.Adam(AE.encoder.parameters(), lr=args.enc_learning_rate)
        opt_dec = torch.optim.Adam(AE.decoder.parameters(), lr=args.dec_learning_rate)
        opt_diz = torch.optim.Adam(Dz.parameters(), lr=args.dz_learning_rate)
        opt_gen = torch.optim.Adam(G.parameters(), lr=args.g_learning_rate)
        #
        if args.dp_sgd == True: # ??? why dec, gen?
            opt_dec = DPSGD(params=AE.decoder.parameters(), lr=args.dec_learning_rate, minibatch_size=args.batch_size, microbatch_size=args.batch_size,
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            opt_gen = DPSGD(params=G.parameters(), lr=args.g_learning_rate, minibatch_size=args.batch_size, microbatch_size=args.batch_size,
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            epsilon = moments_accountant.epsilon(len(datasets['train'].data), args.batch_size, args.noise_multiplier, args.epochs, args.delta)

            print('Training procedure satisfies (%f, %f)-DP' % (epsilon, args.delta)) # ?? question, why 2 epsilon?


        lr_enc = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_enc, gamma=args.enc_lr_decay_rate)
        lr_dec = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dec, gamma=args.dec_lr_decay_rate)
        lr_diz = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_diz, gamma=args.dz_lr_decay_rate)
        lr_gen = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_gen, gamma=args.g_lr_decay_rate)

        

        tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        models = {
            "AE": AE,
            "G": G,
            "Dz": Dz
        }
        opts = {
            "enc": opt_enc,
            "dec": opt_dec,
            "diz": opt_diz,
            "gen": opt_gen
        }
        lrs = {
            "enc": lr_enc,
            "dec": lr_dec,
            "diz": lr_diz,
            "gen": lr_gen
        }
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
        
            if epoch % args.valid_eval_freq == 0:
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
    AE = models["AE"]
    G = models["G"]
    AE.eval()
    G.eval()
    model_generation(args, G, AE.decoder, prob_mask, **kwargs)



def save_model(models, path):
    AE = models["AE"]
    G = models["G"]
    Dz = models["Dz"]

    torch.save(AE, "{}_AE".format(path))
    torch.save(G, "{}_G".format(path))
    torch.save(Dz, "{}_Dz".format(path))


def load_model(path):
    AE = torch.load("{}_AE".format(path))
    G = torch.load("{}_G".format(path))
    Dz = torch.load("{}_Dz".format(path))

    models = {
        "AE": AE,
        "G": G,
        "Dz": Dz
    }
    return models


def model_evaluation(args, models, opts, lrs, data_loader, prob_mask, split, log_file, **kwargs):
    if args.use_prob_mask:
        assert prob_mask is not None
    AE = models["AE"]
    G = models["G"]
    Dz = models["Dz"]
    if split == 'train':
        # opts
        opt_enc = opts["enc"]
        opt_dec = opts["dec"]
        opt_diz = opts["diz"]
        opt_gen = opts["gen"]
        # lr scheduler
        lr_enc = lrs["enc"]
        lr_dec = lrs["dec"]
        lr_diz = lrs["diz"]
        lr_gen = lrs["gen"]

    # init
    recon_total_loss, mask_total_loss = 0.0, 0.0
    zCritic_total_loss = 0.0
    
    n_data = 0

    if split == 'train':
        AE.encoder_dropout=args.encoder_dropout
        AE.decoder_dropout=args.decoder_dropout
        AE.train()
        G.train()
        Dz.train()
    else:
        AE.encoder_dropout=0.0
        AE.decoder_dropout=0.0
        AE.eval()
        G.eval()
        Dz.eval()

    for iteration, batch in enumerate(data_loader):
        batch_size = batch['src_tempo'].shape[0]
        n_data += batch_size
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = to_var(v)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1

        if torch.cuda.is_available():
            one = one.cuda()
            mone = mone.cuda()
        
        src_tempo = batch['src_tempo']; tgt_tempo = batch['tgt_tempo']
        src_mask = batch['src_mask']; tgt_mask = batch['tgt_mask']
        #import pdb; pdb.set_trace()
        # Step 0: Evaluate current loss
        if args.no_mask:
            z, Pinput, Poutput, Moutput = AE(src_tempo, tgt_tempo, None, None)
            # loss
            recon_loss = args.beta_recon * AE.compute_recon_loss(Poutput, tgt_tempo, None, None)
        elif args.use_prob_mask:
            z, Pinput, Poutput, Moutput = AE(src_tempo, tgt_tempo, src_mask, tgt_mask)
            output_mask = sample_mask_from_prob(prob_mask, tgt_mask.shape[0], tgt_mask.shape[1])
            # loss
            recon_loss = args.beta_recon * AE.compute_recon_loss(Poutput, tgt_tempo, output_mask, tgt_mask)
        else:
            z, Pinput, Poutput, Moutput = AE(src_tempo, tgt_tempo, src_mask, tgt_mask)
            # loss
            recon_loss = args.beta_recon * AE.compute_recon_loss(Poutput, tgt_tempo, Moutput, tgt_mask)
            mask_loss = args.beta_mask * AE.compute_mask_loss(Moutput, tgt_mask)


        zgen = G(batch_size=z.size(0))
        time_shift = kwargs["time_shift"]
        time_scale = kwargs["time_scale"]
        # make up start feature
        start_feature, start_mask = sample_start_feature_mask(z.size(0), time_shift, time_scale)
        if args.no_mask:
            Pgen, Mgen = AE.decoder.inference(start_feature=start_feature, start_mask=None, memory=zgen)
        elif args.use_prob_mask:
            Pgen, Mgen = AE.decoder.inference(start_feature=start_feature, start_mask=start_mask, prob_mask=prob_mask, memory=zgen)
        else:
            Pgen, Mgen = AE.decoder.inference(start_feature=start_feature, start_mask=start_mask, memory=zgen)

        Dreal, Dfake = Dz(z).mean(), Dz(zgen).mean()

        zCritic_loss = - Dreal + Dfake
        
        if split == 'train':
            if iteration % args.critic_freq_base < args.critic_freq_hit:
                opt_diz.zero_grad()
                Dreal, Dfake = Dz(z).mean(), Dz(zgen).mean()
                Dreal.backward(mone, retain_graph=True)
                Dfake.backward(one, retain_graph=True)
                Dz.cal_gradient_penalty(z, zgen).backward()
                opt_diz.step()

            # Step 3, 4: Update the Decoder and the Encoder
            opt_dec.zero_grad()            
            opt_enc.zero_grad()
            Dreal = Dz(z).mean()
            Dreal.backward(one, retain_graph=True)

            recon_loss.backward(retain_graph=True)
            if not args.no_mask and not args.use_prob_mask:
                mask_loss.backward(retain_graph=True)
            opt_dec.step()
            opt_enc.step()

            # Step 5: Update the Generator
            opt_gen.zero_grad()
            Dfake = Dz(zgen).mean()
            Dfake.backward(mone, retain_graph=True)
            opt_gen.step()
        
        #import pdb; pdb.set_trace()
        #
        recon_total_loss += recon_loss.data
        if not args.no_mask and not args.use_prob_mask:
            mask_total_loss += mask_loss.data
        else:
            mask_total_loss = 0.0
            mask_loss = 0.0
        zCritic_total_loss += zCritic_loss.data

        if split == 'train' and iteration % args.train_eval_freq == 0:
            # print the losses for each epoch
            print("Learning rate:\t%2.8f"%(lr_gen.get_last_lr()[0]))
            print("Batch loss:")
            print("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\tzCritic_loss\t%9.4f"%(split.upper(), recon_loss, mask_loss, zCritic_loss))
            print()
            with open(log_file, "a+") as file:
                file.write("Learning rate:\t%2.8f\n"%(lr_gen.get_last_lr()[0]))
                file.write("Batch loss:\n")
                file.write("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\tzCritic_loss\t%9.4f\n"%(split.upper(), recon_loss, mask_loss, zCritic_loss))
                file.write("===================================================\n")
    #
    # print the losses for each epoch
    if split == 'train':
        print("Learning rate:\t%2.8f"%(lr_gen.get_last_lr()[0]))
    print("Batch loss:")
    print("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\tzCritic_loss\t%9.4f"%(split.upper(), recon_loss, mask_loss, zCritic_loss))
    if split != "train":
        print("Accumulated loss:")
        print("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\tzCritic_loss\t%9.4f"%(split.upper(), recon_total_loss/iteration, mask_total_loss/iteration, zCritic_total_loss/iteration))
    print()
    with open(log_file, "a+") as file:
        if split == 'train':
            file.write("Learning rate:\t%2.8f\n"%(lr_gen.get_last_lr()[0]))
        file.write("Batch loss:\n")
        file.write("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\tzCritic_loss\t%9.4f\n"%(split.upper(), recon_loss, mask_loss, zCritic_loss))
        if split != "train":
            file.write("Accumulated loss:\n")
            file.write("\t\t%s\trecon_loss\t%9.4f\tmask_loss\t%9.4f\tzCritic_loss\t%9.4f\n"%(split.upper(), recon_total_loss/iteration, mask_total_loss/iteration, zCritic_total_loss/iteration))
        file.write("===================================================\n")
    
    if split == 'train':
        lr_enc.step()
        lr_dec.step()
        lr_diz.step()
        lr_gen.step()

    models = {
        "AE": AE,
        "G": G,
        "Dz": Dz
    }

    return recon_total_loss/iteration, models