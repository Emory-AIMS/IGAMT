import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from utils.train_utils import to_var, sample_start_feature_mask, sample_mask_from_prob, ehr_model_inference as model_inference

from pyvacy import optim, analysis
from pyvacy.optim.dp_optimizer import DPAdam, DPSGD
import pyvacy.analysis.moments_accountant as moments_accountant

from nn.seq2seq_ae import Decoder
from nn.generator import MLP_Generator
from nn.discriminator import CNN_Discriminator


def train_model(args, datasets, prob_mask):
    if not args.test:
        if args.load_model:
            model_path = os.path.join(args.model_path, args.pretrained_model_filename)
            models = load_dgat(model_path)
            Dec = models["Dec"]
            G = models["G"]
            Dx = models["Dx"]
            
        else:
            # model define
            Dx = CNN_Discriminator(
                feature_size=args.feature_size,
                feature_dropout=args.feature_dropout,
                filter_size=args.filter_size,
                window_sizes=args.window_sizes,
                use_spectral_norm = args.use_spectral_norm
                )
            Dec = Decoder(
                rnn=None, 
                rnn_type=args.rnn_type,
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                hidden_factor=None, 
                latent_size=args.latent_size,
                max_length=args.max_length, 
                dropout_rate=args.decoder_dropout, 
                num_layers=args.num_layers,
                bidirectional=args.bidirectional,
                use_prob_mask=args.use_prob_mask
            )

            G = MLP_Generator(
                input_size=args.noise_size,
                output_size=args.latent_size,
                archs=args.gmlp_archs
                )

        if torch.cuda.is_available():
            Dec = Dec.cuda()
            G = G.cuda()
            Dx = Dx.cuda()
        
        

        opt_dec = torch.optim.Adam(Dec.parameters(), lr=args.dec_learning_rate)
        opt_dix = torch.optim.Adam(Dx.parameters(), lr=args.dx_learning_rate)
        opt_gen = torch.optim.Adam(G.parameters(), lr=args.g_learning_rate)
        #
        if args.dp_sgd == True: # ??? why dec, gen?
            opt_dix = DPSGD(params=Dx.parameters(), lr=args.dx_learning_rate, minibatch_size=args.batch_size, microbatch_size=args.batch_size,
                                        l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
            epsilon = moments_accountant.epsilon(len(datasets['train'].data), args.batch_size, args.noise_multiplier, args.epochs, args.delta)

            print('Training procedure satisfies (%f, %f)-DP' % (epsilon, args.delta)) # ?? question, why 2 epsilon?


        
        lr_dec = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dec, gamma=args.dec_lr_decay_rate)
        lr_dix = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dix, gamma=args.dx_lr_decay_rate)
        lr_gen = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_gen, gamma=args.g_lr_decay_rate)

        

        tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        models = {
            "Dec": Dec,
            "G": G,
            "Dx": Dx
        }
        opts = {
            "dec": opt_dec,
            "dix": opt_dix,
            "gen": opt_gen
        }
        lrs = {
            "dec": lr_dec,
            "dix": lr_dix,
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
            _, models = model_evaluation(args, models, opts, lrs, data_loader, prob_mask, "train", log_file)
        
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
                valid_loss, models = model_evaluation(args, models, opts, lrs, data_loader, prob_mask, "valid", log_file)
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
    Dec = models["Dec"]
    G = models["G"]
    Dec.eval()
    G.eval()
    gen_zs, gen_xs, gen_ms = [], [], []
    for i in range(args.gendata_size//args.batch_size):
        zgen = G(batch_size=args.batch_size)
        Pgen, Mgen = model_inference(args, Dec, zgen, prob_mask)
        
        gen_zs.append(zgen)
        gen_xs.append(Pgen)
        gen_ms.append(Mgen)

    gen_zlist = torch.cat(gen_zs).cpu().detach().numpy()
    gen_xlist = torch.cat(gen_xs).cpu().detach().numpy()
    
    
    np.save(os.path.join(args.result_path, '{}_generated_codes.npy'.format(args.model_type)), gen_zlist)
    np.save(os.path.join(args.result_path, '{}_generated_patients.npy'.format(args.model_type)), gen_xlist) 

    if not args.no_mask and not args.use_prob_mask:
        gen_mlist = torch.cat(gen_ms).cpu().detach().numpy()
        np.save(os.path.join(args.result_path, '{}_generated_masks.npy'.format(args.model_type)), gen_mlist)



def save_model(models, path):
    Dec = models["Dec"]
    G = models["G"]
    Dx = models["Dx"]

    torch.save(Dec, "{}_Dec".format(path))
    torch.save(G, "{}_G".format(path))
    torch.save(Dx, "{}_Dx".format(path))


def load_model(path):
    Dec = torch.load("{}_Dec".format(path))
    G = torch.load("{}_G".format(path))
    Dx = torch.load("{}_Dx".format(path))

    models = {
        "Dec": Dec,
        "G": G,
        "Dx": Dx
    }
    return models


def model_evaluation(args, models, opts, lrs, data_loader, prob_mask, split, log_file):
    if args.use_prob_mask:
        assert prob_mask is not None
    Dec = models["Dec"]
    G = models["G"]
    Dx = models["Dx"]
    if split == 'train':
        # opts
        opt_dec = opts["dec"]
        opt_dix = opts["dix"]
        opt_gen = opts["gen"]
        # lr scheduler
        lr_dec = lrs["dec"]
        lr_dix = lrs["dix"]
        lr_gen = lrs["gen"]

    # init
    xCritic_total_loss = 0.0
    
    n_data = 0

    if split == 'train':
        Dec.dropout=args.decoder_dropout
        Dec.train()
        G.train()
        Dx.train()
    else:
        Dec.dropout=0.0
        Dec.eval()
        G.eval()
        Dx.eval()

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


        zgen = G(batch_size=batch_size)
        # make up start feature
        start_feature, start_mask = sample_start_feature_mask(batch_size)
        if args.no_mask:
            Pgen, Mgen = Dec.inference(start_feature=start_feature, start_mask=None, memory=zgen)
        elif args.use_prob_mask:
            Pgen, Mgen = Dec.inference(start_feature=start_feature, start_mask=start_mask, prob_mask=prob_mask, memory=zgen)
        else:
            Pgen, Mgen = Dec.inference(start_feature=start_feature, start_mask=start_mask, memory=zgen)

        Dinput, Dgen = Dx(tgt_tempo, tgt_mask).mean(), Dx(Pgen, Mgen).mean()

        xCritic_loss = - Dinput + Dgen
        
        if split == 'train':
            if iteration % args.critic_freq_base < args.critic_freq_hit:
                # Step 1: Update the Critic_x
                opt_dix.zero_grad()
                Dinput, Dgen = Dx(tgt_tempo, tgt_mask).mean(), Dx(Pgen, Mgen).mean()
                Dinput.backward(mone, retain_graph=True)
                Dgen.backward(one, retain_graph=True)
                Dx.cal_gradient_penalty(tgt_tempo[:, :Pgen.size(1), :], Pgen, Mgen).backward(retain_graph=True)
                opt_dix.step()

            # Step 3, 4: Update the Decoder and generator
            opt_dec.zero_grad()
            opt_gen.zero_grad()
            Dgen = Dx(Pgen, Mgen).mean()
            Dgen.backward(mone, retain_graph=True)
            opt_dec.step()
            opt_gen.step()

        

        xCritic_total_loss += xCritic_loss.data

        if split == 'train' and iteration % args.train_eval_freq == 0:
            # print the losses for each epoch
            print("Learning rate:\t%2.8f"%(lr_gen.get_last_lr()[0]))
            print("Batch loss:")
            print("\t\t%s\txCritic_loss\t%9.4f"%(split.upper(), xCritic_loss))
            print()
            with open(log_file, "a+") as file:
                file.write("Learning rate:\t%2.8f\n"%(lr_gen.get_last_lr()[0]))
                file.write("Batch loss:\n")
                file.write("\t\t%s\txCritic_loss\t%9.4f\n"%(split.upper(), xCritic_loss))
                file.write("===================================================\n")
    #
    # print the losses for each epoch
    if split == 'train':
        print("Learning rate:\t%2.8f"%(lr_gen.get_last_lr()[0]))
    print("Batch loss:")
    print("\t\t%s\txCritic_loss\t%9.4f"%(split.upper(), xCritic_loss))
    if split != "train":
        print("Accumulated loss:")
        print("\t\t%s\txCritic_loss\t%9.4f"%(split.upper(), xCritic_total_loss/iteration))
    print()
    with open(log_file, "a+") as file:
        if split == 'train':
            file.write("Learning rate:\t%2.8f\n"%(lr_gen.get_last_lr()[0]))
        file.write("Batch loss:\n")
        file.write("\t\t%s\txCritic_loss\t%9.4f\n"%(split.upper(), xCritic_loss))
        if split != "train":
            file.write("Accumulated loss:\n")
            file.write("\t\t%s\txCritic_loss\t%9.4f\n"%(split.upper(), xCritic_total_loss/iteration))
        file.write("===================================================\n")
    
    if split == 'train':
        lr_dec.step()
        lr_dix.step()
        lr_gen.step()
    
    models = {
        "Dec": Dec,
        "G": G,
        "Dx": Dx
    }

    return recon_total_loss/iteration, models