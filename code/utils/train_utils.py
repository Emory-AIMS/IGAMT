import os
import json
import time
import torch
import argparse
import numpy as np
from torch.autograd import Variable
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
    

def ed_model_synthesization(args, data_size, G_0, G_1, prob_mask, path=None, prefix=None, **kwargs,):
    if path is None:
        path = args.result_path
    if prefix is None:
        prefix = args.model_type
    gen_zs, gen_xs, gen_ms = [], [], []
    for i in range(data_size//args.batch_size):
        if G_0 is None or args.model_type in ["vae", "vae_gan"]:
            zgen = torch.randn((args.batch_size, args.latent_size))
        else:
            zgen = G_0(batch_size=args.batch_size*args.max_length)
        zgen = torch.reshape(zgen, (args.batch_size, args.max_length, -1))
        Pgen, Mgen = model_inference(args, G_1, zgen, prob_mask, **kwargs)
        
        gen_xs.append(Pgen)
        gen_ms.append(Mgen)
    //
    size_ = data_size%args.batch_size
    if G_0 is None or args.model_type in ["vae", "vae_gan"]:
        zgen = torch.randn((size_, args.latent_size))
    else:
        zgen = G_0(batch_size=size_*args.max_length)
    zgen = torch.reshape(zgen, (size_, args.max_length, -1))
    Pgen, Mgen = model_inference(args, G_1, zgen, prob_mask, **kwargs)
    
    gen_xs.append(Pgen)
    gen_ms.append(Mgen)
    
    gen_xlist = torch.cat(gen_xs).cpu().detach().numpy()   
    gen_mlist = torch.cat(gen_ms).cpu().detach().numpy()
    
    return gen_xlist, gen_mlist


def ed_model_generation(args, G_0, G_1, prob_mask, path=None, prefix=None, **kwargs,):
    if path is None:
        path = args.result_path
    if prefix is None:
        prefix = args.model_type
    gen_zs, gen_xs, gen_ms = [], [], []
    for i in range(args.gendata_size//args.batch_size):
        if G_0 is None or args.model_type in ["vae", "vae_gan"]:
            zgen = torch.randn((args.batch_size, args.latent_size))
        else:
            zgen = G_0(batch_size=args.batch_size*args.max_length)
        zgen = torch.reshape(zgen, (args.batch_size, args.max_length, -1))
        Pgen, Mgen = model_inference(args, G_1, zgen, prob_mask, **kwargs)
        
        gen_zs.append(zgen)
        gen_xs.append(Pgen)
        gen_ms.append(Mgen)

    gen_zlist = torch.cat(gen_zs).cpu().detach().numpy()
    gen_xlist = torch.cat(gen_xs).cpu().detach().numpy()
    
    np.save(os.path.join(path, '{}_generated_codes.npy'.format(args.model_type)), gen_zlist)
    np.save(os.path.join(path, '{}_generated_patients.npy'.format(args.model_type)), gen_xlist) 
    
    if not args.no_mask and not args.use_prob_mask:
        gen_mlist = torch.cat(gen_ms).cpu().detach().numpy()
        np.save(os.path.join(path, '{}_generated_masks.npy'.format(args.model_type)), gen_mlist)


def ehr_model_inference(args, decoder, zgen, prob_mask, **kwargs):
    # make up start feature
    start_feature, start_time, start_mask, sampled_gender, sampled_race = sample_start_feature_time_mask_gender_race(zgen.size(0))
    model_list = ["dgatt", "dgamt", "edgamt", "tgamt", "etgamt", "igamt", "igamt_v2", "igamt_v3"]
    if args.model_type == "dgatt":
        kwargs["start_time"] = start_time
    elif args.model_type in model_list:
        kwargs["start_time"] = start_time
        kwargs["gender"] = sampled_gender
        kwargs["race"] = sampled_race
    else:
        kwargs = {}

    if args.model_type in model_list:
        if args.no_mask:
            Pgen, Tgen, Mgen = decoder.inference(start_feature=start_feature, start_mask=None, memory=zgen, **kwargs)
        elif args.use_prob_mask:
            Pgen, Tgen, Mgen = decoder.inference(start_feature=start_feature, start_mask=start_mask, prob_mask=prob_mask, memory=zgen, **kwargs)
        else:
            Pgen, Tgen, Mgen = decoder.inference(start_feature=start_feature, start_mask=start_mask, memory=zgen, **kwargs)
    else:
        if args.no_mask:
            Pgen, Mgen = decoder.inference(start_feature=start_feature, start_mask=None, memory=zgen, **kwargs)
        elif args.use_prob_mask:
            Pgen, Mgen = decoder.inference(start_feature=start_feature, start_mask=start_mask, prob_mask=prob_mask, memory=zgen, **kwargs)
        else:
            Pgen, Mgen = decoder.inference(start_feature=start_feature, start_mask=start_mask, memory=zgen, **kwargs)

    return Pgen, Mgen


def ed_model_inference(args, decoder, zgen, prob_mask, **kwargs):
    time_shift = kwargs["time_shift"]
    time_scale = kwargs["time_scale"]
    # make up start feature
    start_feature, start_time, start_mask = ed_sample_start_feature_time_mask(zgen.size(0), time_shift, time_scale)
    model_list = ["dgatt", "dgamt", "edgamt", "tgamt", "etgamt", "igamt", "igamt_v2", "igamt_v3"]
    if args.model_type == "dgatt":
        kwargs["start_time"] = start_time
        kwargs["extract_incr_time_from_tempo_step"] = extract_incr_time_from_tempo_step
    elif args.model_type in model_list:
        kwargs["start_time"] = start_time
        kwargs["use_day"] = True
        kwargs["use_hour"] = True
        kwargs["extract_incr_time_from_tempo_step"] = extract_incr_time_from_tempo_step
    else:
        kwargs = {}

    if args.model_type in model_list:
        if args.no_mask:
            Pgen, Tgen, Mgen = decoder.inference(start_feature=start_feature, start_mask=None, memory=zgen, **kwargs)
        elif args.use_prob_mask:
            Pgen, Tgen, Mgen = decoder.inference(start_feature=start_feature, start_mask=start_mask, prob_mask=prob_mask, memory=zgen, **kwargs)
        else:
            Pgen, Tgen, Mgen = decoder.inference(start_feature=start_feature, start_mask=start_mask, memory=zgen, **kwargs)
    else:
        if args.no_mask:
            Pgen, Mgen = decoder.inference(start_feature=start_feature, start_mask=None, memory=zgen, **kwargs)
        elif args.use_prob_mask:
            Pgen, Mgen = decoder.inference(start_feature=start_feature, start_mask=start_mask, prob_mask=prob_mask, memory=zgen, **kwargs)
        else:
            Pgen, Mgen = decoder.inference(start_feature=start_feature, start_mask=start_mask, memory=zgen, **kwargs)

    return Pgen, Mgen


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def sample_gender_race(batch_size):
    gender = torch.randint(0, 2, (batch_size, 1))
    race = torch.randint(0, 3, (batch_size, 1))
    return gender.int(), race.int()


def sample_start_feature_mask(batch_size):
    start_feature, _, start_mask = sample_start_feature_time_mask(batch_size)

    return start_feature, start_mask

def ed_sample_start_feature_mask(batch_size, shift, scale):
    start_feature, _, start_mask = ed_sample_start_feature_time_mask(batch_size, shift, scale)

    return start_feature, start_mask


def sample_start_feature_time_mask(batch_size):
    padding = torch.zeros(batch_size, 1, dtype=torch.float)
    age = torch.tensor(np.random.uniform(size=(batch_size, 1))*0.9, dtype=torch.float)
    year = torch.tensor(np.random.uniform(size=(batch_size, 1))*0.9, dtype=torch.float)
    start_feature = torch.cat((age, year, age, year, age, year, age, year, padding), 1)
    start_mask = torch.tensor(np.tile(np.expand_dims(np.array([1]*8+[0]), 0), [batch_size, 1]))

    return start_feature, year, start_mask


def extract_day_hour_from_time(year):
    day = (year - torch.floor(year)) * 365
    hour = (day - torch.floor(day)) * 24
    return day, hour


def ed_sample_start_feature_time_mask(batch_size, shift, scale):
    
    year = torch.tensor(np.random.uniform(size=(batch_size, 1))*0.9, dtype=torch.float)
    descaled_year = descale_time(year, shift[0, 0], scale[0, 0])
    descaled_day, descaled_hour = extract_day_hour_from_time(descaled_year)
    day = (descaled_day - shift[0, 1])/scale[0, 1]
    hour = (descaled_hour - shift[0, 2])/scale[0, 2]
    
    padding = torch.zeros(batch_size, 1, dtype=torch.float)
    start_feature = torch.cat((year, day, hour, year, day, hour, year, day, hour, padding), 1)
    start_mask = torch.tensor(np.tile(np.expand_dims(np.array([1]*9+[0]), 0), [batch_size, 1]))
    '''
    padding = torch.zeros(batch_size, 7, dtype=torch.float)
    start_feature = torch.cat((year, day, hour, padding), 1)
    start_mask = torch.tensor(np.tile(np.expand_dims(np.array([1]*3+[0]*7), 0), [batch_size, 1]))
    '''
    return start_feature, year, start_mask



def sample_start_feature_time_mask_gender_race(batch_size):
    padding = torch.zeros(batch_size, 1, dtype=torch.float)
    age = torch.tensor(np.random.uniform(size=(batch_size, 1))*0.9, dtype=torch.float)
    year = torch.tensor(np.random.uniform(size=(batch_size, 1))*0.9, dtype=torch.float)
    gender, race = sample_gender_race(batch_size)
    gender_shape = (batch_size, 2)
    race_shape = (batch_size, 3)
    scale = 0.2; shift = 0.8
    gender_pos = torch.nn.functional.one_hot(gender.squeeze(1).long(), num_classes=2) * (torch.rand(gender_shape) * scale + shift)
    race_pos = torch.nn.functional.one_hot(race.squeeze(1).long(), num_classes=3) * (torch.rand(race_shape) * scale + shift)
    gender_neg = (1 - gender_pos) * (torch.rand(gender_shape) * scale)
    race_neg = (1 - race_pos) * (torch.rand(race_shape) * scale)
    gender_ = gender_pos + gender_neg
    race_ = race_pos + race_neg
    start_feature = torch.cat((age, year, gender_, race_, age, year), 1)
    start_mask = torch.tensor(np.tile(np.expand_dims(np.array([1]*9), 0), [batch_size, 1]))

    return start_feature, year, start_mask, gender, race


def extract_time_from_start_feature(start_feature):
    assert len(start_feature.shape) == 2
    return start_feature[:, 3] # [batch_size, 1]


def extract_incr_time_from_tempo_step(temporal_step_feature):
    assert len(temporal_step_feature.shape) == 2
    return temporal_step_feature[:, -1].unsqueeze(dim=1)  # [batch_size, 1]

def ed_extract_incr_time_from_tempo_step(temporal_step_feature):
    assert len(temporal_step_feature.shape) == 2
    return temporal_step_feature[:, 0].unsqueeze(dim=1)  # [batch_size, 1]

def descale_time(scaled_time, shift, scale):
    return torch.floor(scaled_time * scale + shift)


def sample_mask_from_prob(prob_mask, batch_size, steps):
    prob_mask = torch.tensor(prob_mask, dtype=torch.float).cuda()
    prob_mask = torch.squeeze(prob_mask)
    prob_mask = torch.unsqueeze(prob_mask, 0); prob_mask = torch.unsqueeze(prob_mask, 0)
    prob = torch.tile(prob_mask, (batch_size, steps, 1))
    return torch.bernoulli(prob)
