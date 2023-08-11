import os
import json
import time
import torch
import argparse
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from datasets.ed import ED
from datasets.syn_ed import ED as SYN_ED
#
from utils.train_utils import ed_model_synthesization as model_syn
from nn.seq2seq_ae import Seq2seq_Autoencoder



def save_model(models, path):
    AE = models["AE"]

    torch.save(AE, "{}_AE".format(path))


def load_model(path):
    AE = torch.load("{}_AE".format(path))

    models = {
        "AE": AE
    }
    return models


def model_evaluation(args, models, opts, lrs, data_loader, prob_mask, split, log_file, **kwargs):
    AE = models["AE"]
    if split == 'train':
        # opts
        opt_enc = opts["enc"]
        opt_dec = opts["dec"]
        # lr scheduler
        lr_enc = lrs["enc"]
        lr_dec = lrs["dec"]

    # init
    recon_total_loss = 0.0
    
    n_data = 0

    if split == 'train':
        AE.encoder_dropout=args.encoder_dropout
        AE.decoder_dropout=args.decoder_dropout
        AE.train()
    else:
        AE.encoder_dropout=0.0
        AE.decoder_dropout=0.0
        AE.eval()

    for iteration, batch in enumerate(data_loader):
        batch_size = batch['src_tempo'].shape[0]
        n_data += batch_size
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = to_var(v)
        
        src_tempo = batch['src_tempo']; tgt_tempo = batch['tgt_tempo']
        src_mask = batch['src_mask']; tgt_mask = batch['tgt_mask']
        #import pdb; pdb.set_trace()

        Poutput, Moutput = AE(src_tempo, None, src_mask, None)
        recon_loss = args.beta_recon * AE.compute_recon_loss(src_tempo, Poutput, src_mask, Moutput)
       
        
        if split == 'train':
            opt_dec.zero_grad()            
            opt_enc.zero_grad()
            recon_loss.backward(retain_graph=True)
            opt_dec.step()
            opt_enc.step()
        
        #import pdb; pdb.set_trace()
        #
        recon_total_loss += recon_loss.data

        if split == 'train' and iteration % args.train_eval_freq == 0:
            # print the losses for each epoch
            print("Learning rate:\t%2.8f"%(lr_enc.get_last_lr()[0]))
            print("Batch loss:")
            print("\t\t%s\trecon_loss\t%9.4f"%(split.upper(), recon_loss))
            print()
            with open(log_file, "a+") as file:
                file.write("Learning rate:\t%2.8f\n"%(lr_enc.get_last_lr()[0]))
                file.write("Batch loss:\n")
                file.write("\t\t%s\trecon_loss\t%9.4f\n"%(split.upper(), recon_loss))
                file.write("===================================================\n")
    #
    # print the losses for each epoch
    if split == 'train':
        print("Learning rate:\t%2.8f"%(lr_enc.get_last_lr()[0]))
    print("Batch loss:")
    print("\t\t%s\trecon_loss\t%9.4f"%(split.upper(), recon_loss))
    if split != "train":
        print("Accumulated loss:")
        print("\t\t%s\trecon_loss\t%9.4f"%(split.upper(), recon_total_loss/iteration))
    print()
    with open(log_file, "a+") as file:
        if split == 'train':
            file.write("Learning rate:\t%2.8f\n"%(lr_enc.get_last_lr()[0]))
        file.write("Batch loss:\n")
        file.write("\t\t%s\trecon_loss\t%9.4f\n"%(split.upper(), recon_loss))
        if split != "train":
            file.write("Accumulated loss:\n")
            file.write("\t\t%s\trecon_loss\t%9.4f\n"%(split.upper(), recon_total_loss/iteration))
        file.write("===================================================\n")
    
    if split == 'train':
        lr_enc.step()
        lr_dec.step()
    
    models = {
        "AE": AE
    }

    return recon_total_loss/iteration, models


def train_ae(args, datasets):
    if not args.test:
        if args.load_model:
            model_path = os.path.join(args.model_path, args.pretrained_model_filename)
            models = load_dgat(model_path)
            AE = models["AE"]
            
        else:
            # model define
            AE = Seq2seq_Variational_Autoencoder(
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
        
        
        opt_enc = torch.optim.Adam(AE.encoder.parameters(), lr=args.enc_learning_rate)
        opt_dec = torch.optim.Adam(AE.decoder.parameters(), lr=args.dec_learning_rate)
        #

        lr_enc = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_enc, gamma=args.enc_lr_decay_rate)
        lr_dec = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_dec, gamma=args.dec_lr_decay_rate)

        tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        models = {
            "AE": AE
        }
        opts = {
            "enc": opt_enc,
            "dec": opt_dec
        }
        lrs = {
            "enc": lr_enc,
            "dec": lr_dec
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
            model_evaluation(args, models, opts, lrs, data_loader, prob_mask, "train", log_file, **kwargs)
        
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
                valid_loss = model_evaluation(args, models, opts, lrs, data_loader, prob_mask, "valid", log_file, **kwargs)
                print("****************************************************")
                print()
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    path = "{}/model_vloss_{}".format(args.model_path, valid_loss)
                    min_valid_path = path

                    models = {
                        "AE": AE
                    }
                    save_model(models, path)

                
            # Generate the synthetic sequences as many as you want 
        model_path = min_valid_path
    else:
        model_path = os.path.join(args.model_path, args.test_model_filename)
    


    

def main(args):
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
    if args.test:
        assert args.test_model_filename != ""
    
    torch.cuda.set_device(args.gpu_devidx) 
    splits = ["train", "valid", "test"]


    infer_file_path = os.path.join(args.data_dir, args.struct_info_file)
    infer_info = np.load(infer_file_path, allow_pickle=True).item()
    max_year = infer_info["max"]["start_year_day_hour"]
    min_year = infer_info["min"]["start_year_day_hour"]
    time_shift = min_year
    time_scale = max_year - min_year

    datasets = OrderedDict()
    if data_type == "ehr":
        for split in splits:
            whole_data = ED(
                data_dir=args.data_dir,
                split=split,
                max_length=args.max_length
            )
            datasets[split] = whole_data
    else:
        model_path = os.path.join(args.model_path, args.pretrained_model_filename)
        models = load_model(model_path)
        kwargs = {
                "time_shift": time_shift, 
                "time_scale": time_scale
            }
        dataset_size = {
            "train": args.syn_train_size,
            "valid": args.syn_valid_size,
            "test": args.syn_test_size
        }
        model_data_path = os.path.join(data_dir, data_type)
        if not os.path.exists(model_data_path):
            os.mkdir(model_data_path)
        for split in splits:
            path_ = os.path.join(model_data_path, split); 
            if not os.path.exists(path_):
                if data_type in ["aae", "daae"]:
                    AE = models["AE"]
                    G = models["G"]
                    AE.eval()
                    G.eval()
                    X, M = model_syn(args, dataset_size[split], G, AE.decoder, prob_mask, **kwargs)

                elif args.data_type == "gan":
                    """ Only one GAN
                    """
                    Dec = models["Dec"]
                    G = models["G"]
                    Dec.eval()
                    G.eval()
                    X, M = model_syn(args, dataset_size[split], G, Dec, prob_mask, **kwargs)
                
                elif args.model_type in ["vae", "seq2seq_vae", "vae_gan"]:
                    AE = models["AE"]
                    AE.eval()
                    X, M = model_syn(args, dataset_size[split], None, AE.decoder, prob_mask, **kwargs)
                
                elif args.model_type in ["dgat", "dgatt", "dgamt", "edgamt"]: 
                    Trans = models["Trans"]
                    G = models["G"]
                    Trans.eval()
                    G.eval()
                    X, M = model_syn(args, dataset_size[split], G, Trans.decoder, prob_mask, **kwargs)

                elif args.model_type == "igamt": # triplet generative adversarial time-embedding transforls
                    """ There are two GANs in daae, one is for output data x, one is for hidden state z and the other one if for imitation of x
                    """
                    Imi = models["Imi"]
                    G = models["G"]
                    Imi.eval()
                    G.eval()
                    X, M = model_syn(args, dataset_size[split], G, Imi, prob_mask, **kwargs)
                
                os.mkdir(path_)
                np.save(os.path.join(path_, "data.npy"), X)
                np.save(os.path.join(path_, "mask.npy"), M)
            else:
                X = np.load(os.path.join(path_, "data.npy"))
                M = np.load(os.path.join(path_, "mask.npy"))
            datasets[split] = SYN_ED({"data": X, "mask": M}, split)


        train_ae(args, datasets)
        
 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, default='daae')
    parser.add_argument('--data_type', type=str, default='ehr')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--pretrained_model_filename', type=str, default="")
    parser.add_argument('--test_model_filename', type=str, default="")
    parser.add_argument('--train_log', type=str, default='train_log_file')
    parser.add_argument('--valid_log', type=str, default='valid_log_file')
    parser.add_argument('--test_log', type=str, default='test_log_file')
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--private_model_path', type=str, default='models')
    parser.add_argument('--private_model_filename', type=str, default="")
    parser.add_argument('--result_path', type=str, default='results')
    parser.add_argument('--struct_info_file', type=str, default='struct_info.npy')
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--train_eval_freq', type=int, default=50)
    parser.add_argument('--valid_eval_freq', type=int, default=1)
    parser.add_argument('--critic_freq_base', type=int, default=5)
    parser.add_argument('--critic_freq_hit', type=int, default=1)
    parser.add_argument('--gen_freq_base', type=int, default=5)
    parser.add_argument('--gen_freq_hit', type=int, default=1)
    parser.add_argument('--use_spectral_norm', type=bool, default=False)
    parser.add_argument('--no_mask', type=bool, default=False)
    parser.add_argument('--use_prob_mask', type=bool, default=False)
    parser.add_argument('--prob_mask_filename', type=str, default='not_nan_prob.npy')
    
    parser.add_argument('-ep', '--epochs', type=int, default=500)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('--enc_learning_rate', type=float, default=0.001)
    parser.add_argument('--dec_learning_rate', type=float, default=0.001)
    parser.add_argument('--imi_learning_rate', type=float, default=0.001)
    parser.add_argument('--uni_learning_rate', type=float, default=0.001)
    parser.add_argument('--dx_learning_rate', type=float, default=0.001)
    parser.add_argument('--dz_learning_rate', type=float, default=0.001)
    parser.add_argument('--dm_learning_rate', type=float, default=0.001)
    parser.add_argument('--di_learning_rate', type=float, default=0.001)
    parser.add_argument('--dmi_learning_rate', type=float, default=0.001)
    parser.add_argument('--g_learning_rate', type=float, default=0.001)
    parser.add_argument('--ax_learning_rate', type=float, default=0.001)
    parser.add_argument('--az_learning_rate', type=float, default=0.001)
    parser.add_argument('--am_learning_rate', type=float, default=0.001)
    parser.add_argument('--gi_learning_rate', type=float, default=0.001)
    #
    parser.add_argument('--enc_lr_decay_rate', type=float, default=0.99)
    parser.add_argument('--dec_lr_decay_rate', type=float, default=0.99)
    parser.add_argument('--imi_lr_decay_rate', type=float, default=0.99)
    parser.add_argument('--uni_lr_decay_rate', type=float, default=0.99)
    parser.add_argument('--dx_lr_decay_rate', type=float, default=0.99)
    parser.add_argument('--dz_lr_decay_rate', type=float, default=0.99)
    parser.add_argument('--dm_lr_decay_rate', type=float, default=0.99)
    parser.add_argument('--di_lr_decay_rate', type=float, default=0.99)
    parser.add_argument('--dmi_lr_decay_rate', type=float, default=0.99)
    parser.add_argument('--g_lr_decay_rate', type=float, default=0.99)
    parser.add_argument('--ax_lr_decay_rate', type=float, default=0.99)
    parser.add_argument('--az_lr_decay_rate', type=float, default=0.99)
    parser.add_argument('--am_lr_decay_rate', type=float, default=0.99)
    parser.add_argument('--gi_lr_decay_rate', type=float, default=0.99)
    parser.add_argument('-beta_r', '--beta_recon', type=float, default=10.0)
    parser.add_argument('-beta_m', '--beta_mask', type=float, default=1.0)
    parser.add_argument('-beta_mt', '--beta_match', type=float, default=1.0)
    parser.add_argument('-beta_mt_g', '--beta_match_g', type=float, default=1.0)
    parser.add_argument('-beta_mt_i', '--beta_match_i', type=float, default=1.0)
    parser.add_argument('-beta_mt_o', '--beta_match_o', type=float, default=1.0)
    parser.add_argument('-beta_ra', '--beta_race', type=float, default=1.0)
    parser.add_argument('-beta_gd', '--beta_gender', type=float, default=1.0)
    parser.add_argument('-beta_k', '--beta_kld', type=float, default=1.0)
    parser.add_argument('-gs','--gendata_size', type=int, default=100000)
    parser.add_argument('-gd', '--gpu_devidx', type=int, default=0)

    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--dx_num_heads', type=int, default=6)
    parser.add_argument('-fts', '--feature_size', type=int, default=9)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=128)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('--dx_num_layers', type=int, default=1)
    parser.add_argument('--dx_hidden_size', type=int, default=128)
    parser.add_argument('--num_dx_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ns', '--noise_size', type=int, default=128)
    parser.add_argument('-ls', '--latent_size', type=int, default=128)
    parser.add_argument('--dx_latent_size', type=int, default=128)
    parser.add_argument('-fis', '--filter_size', type=int, default=16)
    parser.add_argument('-ws', '--window_sizes', nargs='+', type=int, default=[2, 3])
    parser.add_argument('-dz_ws', '--dz_window_sizes', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--dx_dropout', type=float, default=0.5)
    parser.add_argument('--uni_feature_size', type=int, default=9)
    parser.add_argument('--uni_feature_dropout', type=float, default=0.5)
    parser.add_argument('--uni_filter_size', type=int, default=16)
    parser.add_argument('--uni_window_sizes', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--uni_out_size', type=int, default=16)
    parser.add_argument('-ed', '--encoder_dropout', type=float, default=0.5)
    parser.add_argument('-dd', '--decoder_dropout', type=float, default=0.5)
    parser.add_argument('-fd', '--feature_dropout', type=float, default=0.5)
    parser.add_argument('-ga', '--gmlp_archs', nargs='+', type=int, default=[128, 128])
    parser.add_argument('-da', '--dmlp_archs', nargs='+', type=int, default=[256, 128])

    parser.add_argument('--no_recon', type=bool, default=False)
    parser.add_argument('--dp_sgd', type=bool, default=False)
    parser.add_argument('--noise_multiplier', type=float, default=1)
    parser.add_argument('--l2_norm_clip', type=float, default=0.5)
    parser.add_argument('--delta', type=float, default=1e-3)

    args = parser.parse_args()
    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']

    main(args)
