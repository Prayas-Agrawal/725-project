import os
import torch
import random
import numpy as np
import pickle

def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return None

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

class DistillArgs:
    gpu_no = 0
    train_flag = False
    data_path = "./DATA"
    data = "MNIST"
    num_class = 10 # depends on data
    num_workers = 2
    epoch = 30
    print_interval = 100
    batch_size = 16
    save_path = "./WEIGHTS"
    save_epoch = 10
    valid_interval = 10
    model = "1"
    model_load = None
    teacher_load = None
    teacher = load_pickle(os.path.join(os.path.dirname(teacher_load), 'arguments.pickle')).model if teacher_load is not None else None
    temperature = 1
    distillation_weight = 0.3

def process_args(args):
    if not args.train_flag and args.model_load:
        _model_load = args.model_load
        _train_flag = args.train_flag
        _gpu_no     = args.gpu_no

        loaded_args = load_pickle(os.path.join(os.path.dirname(args.model_load), 'arguments.pickle'))

        args = loaded_args
        args.train_flag = _train_flag
        args.model_load = _model_load
        args.gpu_no     = _gpu_no

    print('*'*30+'\nArguments\n'+ '*'*30)
    for k, v in sorted(vars(args).items()):
        print("%s: %s"%(k, v))

    device = torch.device('cuda:%d'%args.gpu_no) if args.gpu_no >= 0 else torch.device('cpu')

    if args.train_flag:
        os.makedirs(args.save_path, exist_ok=True)
        save_pickle(os.path.join(args.save_path, 'arguments.pickle'), args)

    return args, device
