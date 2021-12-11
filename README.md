## environment
To install the environment required, run
```bash
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install -U scikit-image
pip3 install tqdm
pip3 install glob2
```

## train
to train the model, make sure you have the environment set up. Then run
```bash
python3 main.py --data_root YOUR_DATA_ROOT_TO_DEPLOY --log_interval 100 --save_interval 500 --train --val --test --holdout 0 --gpu_id 0 --augment
```
## test
to test the model, run
```bash
python3 main.py --data_root YOUR_DATAROOT_TO_DEPLOY --test --gpu_id 0 --model_dir /home/ubuntu/ROB535/ckpt/FPN_lr_1e-05_bs_4_maxstep_10000_1
--step_label best
```
The output is in ```/home/ubuntu/ROB535/ckpt/FPN_lr_1e-05_bs_4_maxstep_10000_1/infer.csv```