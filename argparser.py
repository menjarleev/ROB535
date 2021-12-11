import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--data_root', type=str, default='./')
    parser.add_argument('--num_res_block', type=int, default=5)
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--input_size', default=(512, 1024))
    parser.add_argument('--max_channel', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ckpt_root', type=str, default='./ckpt')
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--max_step', type=int, default=100000)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--step_label', type=str, default='best')
    parser.add_argument('--num_class', type=int, default=3)
    parser.add_argument('--train', action='store_true', dest='train')
    parser.add_argument('--model', type=str, default='simple')
    parser.add_argument('--val', action='store_true', dest='val')
    parser.add_argument('--test', action='store_true', dest='test')
    parser.add_argument('--holdout', type=int, default=-1)
    parser.add_argument('--k_fold', type=int, default=10)
    parser.add_argument('--augment', action='store_true', dest='augment')
    parser.add_argument('--num_workers', type=int, default=8)
    return parser.parse_args()

def get_option():
    opt = parse_args()
    if not os.path.exists(f"{opt.ckpt_root}"):
        os.makedirs(f"{opt.ckpt_root}")   # Added to avoid path not exist error on Win10

    setattr(opt, 'name', f'{opt.model}_lr_{opt.lr}_bs_{opt.batch_size}_maxstep_{opt.max_step}')

    n = len([x for x in os.listdir(opt.ckpt_root) if x.startswith(opt.name)])
    save_dir = os.path.join(opt.ckpt_root, f'{opt.name}_{n + 1}')
    if opt.model_dir is not None:
        save_dir = opt.model_dir
    else:
        os.makedirs(save_dir, exist_ok=False)
    setattr(opt, 'save_dir', save_dir)
    return opt

