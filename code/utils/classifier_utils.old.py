import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from utils.train_utils import to_var, ed_sample_start_feature_time_mask as sample_start_feature_time_mask, sample_mask_from_prob
from utils.train_utils import ed_model_inference as model_inference, ed_extract_incr_time_from_tempo_step as extract_incr_time_from_tempo_step

from pyvacy import optim, analysis
from pyvacy.optim.dp_optimizer import DPAdam, DPSGD
import pyvacy.analysis.moments_accountant as moments_accountant

from nn.transformers.mixed_embedding_transformer import Transformer, GeneralTransformerDecoder
from nn.generator import MLP_Generator
from nn.discriminator import MLP_Discriminator, CNN_Discriminator, CNN_Auxiliary_Discriminator, CNN_Net



def train_model(args, datasets, prob_mask, **kwargs):
    if not args.test:
        # model define
        if args.load_model:
            model_path = os.path.join(args.model_path, args.classifier_filename)
            Cx = load_model(model_path)
            
        else:
            
            Cx = CNN_Auxiliary_Discriminator(
                feature_size=args.feature_size,
                feature_dropout=args.feature_dropout,
                filter_size=args.filter_size,
                window_sizes=args.window_sizes,
                use_spectral_norm = args.use_spectral_norm，
                output_size = args.output_size
                )
            

        if torch.cuda.is_available():
            Cx = Cx.cuda()

        
        
        opt = torch.optim.Adam(Cx.parameters(), lr=args.cx_learning_rate)
       
        lr = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=args.cx_lr_decay_rate)
        tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        

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
            _, model = model_evaluation(args, Cx, opt, lr, data_loader, prob_mask, "train", log_file, **kwargs)

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
                valid_loss, model = model_evaluation(args, Cx, opt, lr, data_loader, prob_mask, "valid", log_file, **kwargs)
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


def model_evaluation(args, Cx, opt, lr, data_loader, prob_mask, split, log_file, **kwargs):
    loss = 0.0

    if split == 'train':
        Cx.train()
    else:
        Cx.eval()

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

        targets = batch["label"]
        tgt_tempo = batch['tgt_tempo']
        tgt_mask = batch['tgt_mask']

        preds = Cx(tgt_tempo*tgt_mask)
        batch_loss = Cx.cal_xentropy_loss(preds, targets)

        if split == 'train':
            params = list(Cx.parameters())
            opt.zero_grad()
            loss.backward(inputs=params, retain_graph=True)
            opt.step()

        
        loss += batch_loss

        if split == 'train' and iteration % args.train_eval_freq == 0:
            # print the losses for each epoch
            print("Learning rate:\t%2.8f"%(lr_gen.get_last_lr()[0]))
            print("Batch loss:\t%9.4f"%(batch_loss))
            print()
            with open(log_file, "a+") as file:
                file.write("Learning rate:\t%2.8f\n"%(lr_gen.get_last_lr()[0]))
                file.write("Batch loss:\t%9.4f\n"%(batch_loss))
                file.write("===================================================\n")
    #
    # print the losses for each epoch
    if split == 'train':
        print("Learning rate:\t%2.8f"%(lr_gen.get_last_lr()[0]))
    print("Batch loss:\t%9.4f"%(batch_loss))

    if split != "train":
        print("Accumulated loss:\t%9.4f"%(loss))
    print()
    with open(log_file, "a+") as file:
        if split == 'train':
            file.write("Learning rate:\t%2.8f\n"%(lr_gen.get_last_lr()[0]))
        file.write("Batch loss:\t%9.4f\n"%(batch_loss))
        if split != "train":
            file.write("Accumulated loss:\t%9.4f\n"%(loss))
    
    if split == 'train':
        lr.step()
    

    return loss/iteration, Cx