from argparser import get_option
from utils.visualizer import Visualizer
from model.model import PureRGBNet
from data.dataset import BaseDataset
from torch.utils.data import DataLoader
from functools import partial
from utils.transform import rotate2d, cutout, shift_color, guassian_noise, Transform
from solver import Solver
import random
import numpy as np
import torch
import os
import json

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)

def main():
    opt = get_option()
    set_seed(0)
    if opt.gpu_id != -1:
        torch.backends.cudnn.benchmark = True
    # set up visualizer
    Visualizer.log_path = os.path.join(opt.save_dir, 'log.txt')
    if opt.model_dir is not None:
        file = os.path.join(opt.save_dir, 'opt.txt')
        args = vars(opt)
        Visualizer.log_print('--------------- Options ---------------')
        for k, v in args.items():
            Visualizer.log_print('%s: %s' % (str(k), str(v)))
        Visualizer.log_print('----------------- End -----------------')
        with open(file, 'w') as json_file:
            json.dump(args, json_file)

    network = PureRGBNet(opt.num_res_block, opt.ngf, opt.max_channel, opt.input_dim, opt.num_class)
    solver = Solver(network, opt.gpu_id)
    if opt.train:
        # for debug
        # from pyinstrument import Profiler
        # profiler = Profiler()
        # profiler.start()
        aug_func = None
        if opt.augment:
            t = [rotate2d, partial(cutout, sz=64), shift_color, partial(guassian_noise, std=0.05)]
            aug_func= Transform(t)
        train_dataset = BaseDataset(base_path=opt.data_root,
                                    phase='train',
                                    holdout=opt.holdout,
                                    k_fold=opt.k_fold,
                                    input_size=opt.input_size,
                                    num_class=opt.num_class,
                                    aug_func=aug_func,
                                    augment=opt.augment)
        train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
        val_dataloader = None
        if opt.val:
            val_dataset = BaseDataset(base_path=opt.data_root,
                                      phase='val',
                                      holdout=opt.holdout,
                                      k_fold=opt.k_fold,
                                      input_size=opt.input_size,
                                      num_class=opt.num_class)
            val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size)
        solver.fit(lr=opt.lr,
                   save_dir=opt.save_dir,
                   model_dir=opt.model_dir,
                   max_step=opt.max_step,
                   step_label=opt.step_label,
                   log_interval=opt.log_interval,
                   save_interval=opt.save_interval,
                   train_dataloader=train_dataloader,
                   val_dataloader=val_dataloader,
                   val=opt.val)
        # profiler.stop()
        # print(profiler.output_text())
    if opt.test:
        # create dataloader
        # substitute train with infer dataset you like
        test_dataset = BaseDataset(base_path=opt.data_root,
                                   phase='test',
                                   input_size=opt.input_size,
                                   num_class=opt.num_class)
        test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
        solver.inference(test_dataloader, opt.data_root, opt.save_dir, opt.step_label)

if __name__ == '__main__':
    main()
