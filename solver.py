from utils.visualizer import Visualizer
from torch.utils.data import DataLoader
from data.dataset import BaseDataset
from subprocess import call
from glob import glob
import torch.nn as nn
import torch
import os
import json
import csv
from tqdm import tqdm

class Solver:
    def __init__(self, model, gpu_id=-1):
        self.model = model
        self.device = torch.device(f'cuda:{gpu_id}' if gpu_id != -1 else 'cpu')

    def fit(self,
            lr,
            save_dir,
            model_dir,
            max_step,
            step_label,
            log_interval,
            save_interval,
            train_dataloader,
            val_dataloader=None,
            val=False):
        loss_func = nn.CrossEntropyLoss()
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[int(max_step * 0.5), int(max_step * 0.75)], gamma=0.1)
        Visualizer.log_print("# params of model: {}".format(sum(map(lambda x: x.numel(), self.model.parameters()))))

        log_json = {}
        start = 0
        if model_dir is not None:
            self.load(save_dir, step_label, True)
            json_path = os.path.join(save_dir, 'result.txt')
            if os.path.isfile(json_path):
                with open(json_path) as json_file:
                    log_json = json.load(json_file)
                best_result = log_json['best']
                latest_result = log_json['latest']
                Visualizer.log_print('========== Resuming from iteration {}K ========'
                                     .format(latest_result['step'] // 1000))
                start = latest_result['step'] if step_label == 'latest' else best_result['step']
            else:
                Visualizer.log_print('iteration file at %s is not found' % json_path)

        for step in tqdm(range(start, max_step), desc='train', leave=False):
            try:
                inputs = next(iter_train_data)
            except (UnboundLocalError, StopIteration):
                iter_train_data = iter(train_dataloader)
                inputs = next(iter_train_data)
            img = inputs['image'].to(self.device)
            label = inputs['label'].to(self.device)
            y_hat = self.model(img)
            loss = loss_func(label, y_hat)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if (step + 1) % log_interval == 0:
                call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
                Visualizer.log_print(f"CE loss: {loss.detach().item()}")

            if (step + 1) % save_interval == 0:
                self.summary_and_save(step, max_step, save_dir, log_json, val_dataloader, val)

    @torch.no_grad()
    def test(self,
             save_dir,
             test_dataloader=None,
             step_label='best'):
        self.load(save_dir, step_label, False)
        json_path = os.path.join(save_dir, 'result.txt')
        if os.path.isfile(json_path):
            with open(json_path) as json_file:
                log_result = json.load(json_file)
        else:
            log_result = {}
        test_result = self.evaluate(save_dir, test_dataloader)
        log_result['test'] = test_result
        test_log = 'test result \n'
        for k, v in test_result.items():
            if k != 'step':
                test_log += f'{k}: {v:.2f} '
        Visualizer.log_print(test_log)
        self.save_log_json(log_result, save_dir, 'test')

    def summary_and_save(self, step, max_step, save_dir, log_json, val_dataloader, val):
        best_result = log_json['best'] if 'best' in log_json else None
        if val and val_dataloader is not None:
            curr_result = self.evaluate(dataloader=val_dataloader, phase='val')
            curr_result['step'] = step + 1
            if best_result is None or (curr_result['accuracy'] >= best_result['accuracy']):
                log_json['best'] = curr_result
                best_result = curr_result
                self.save(save_dir, 'best')
                self.save_log_json(log_json, save_dir, 'best')
            log_json['latest'] = curr_result
            curr_lr = self.scheduler.get_lr()[0]
            msg_curr = 'curr result \n'
            for k, v in curr_result.items():
                if k != 'step':
                    msg_curr += f'{k}: {v:.2f} '
            msg_best = 'best result \n'
            for k, v in best_result.items():
                if k != 'step':
                    msg_best += f'{k}: {v:.2f} '
            message = f'[{(step + 1) // 1000}K/{max_step // 1000}K] \n' \
                      f'lr:{curr_lr}\n' \
                      f'{msg_curr}\n' \
                      f'{msg_best}\n'
            Visualizer.log_print(message)
            self.save_log_json(log_json, save_dir, 'latest')
        else:
            'save latest result'
        self.save(save_dir, 'latest')

    @torch.no_grad()
    def inference(self, dataloader, data_root, save_dir, label):
        self.load(save_dir, label, False)
        self.model.eval()
        files = glob(f'{data_root}/test/*/*_image.jpg')
        files.sort()
        tqdm_data_loader = tqdm(dataloader, desc='infer', leave=False)
        dest_infer_path = f'{save_dir}/infer.csv'
        with open(dest_infer_path, 'w') as f:
            writer = csv.writer(f, delimiter=',', lineterminator='\n')
            writer.writerow(['guid/image', 'label'])
            for i, inputs in enumerate(tqdm_data_loader):
                img = inputs['image'].to(self.device)
                out = self.model(img)
                y_hat = torch.argmax(out, 1)
                bs = img.size()[0]
                for j in range(bs):
                    guid = files[i * bs + j].split('/')[-2]
                    idx = files[i * bs + j].split('/')[-1].replace('_image.jpg', '')
                    writer.writerow([f'{guid}/{idx}', y_hat[j].detach().cpu().item()])

    @torch.no_grad()
    def evaluate(self, dataloader, phase='test'):
        self.model.eval()
        tqdm_data_loader = tqdm(dataloader, desc=phase, leave=False)
        total = 0
        correct = 0
        for i, inputs in enumerate(tqdm_data_loader):
            img = inputs['image'].to(self.device)
            label = inputs['label'].to(self.device)
            out = self.model(img)
            y_hat = torch.argmax(out, 1)
            y = torch.argmax(label, 1)
            correct += torch.sum(y_hat == y)
            total += img.shape[0]
        self.model.train()

        return {'accuracy': (correct / total).detach().item()}

    def save_log_json(self, log_json, save_dir, label):
        json_path = os.path.join(save_dir, 'result.txt')
        with open(json_path, 'w') as json_file:
            json.dump(log_json, json_file, indent=4)
        Visualizer.log_print('update [{}] for status file'.format(label))

    def save(self, save_dir, label):
        state_dict = dict()
        state_dict['network'] = self.model.cpu().state_dict()
        self.model.to(self.device)
        if self.optim is not None:
            state_dict['optimizer'] = self.optim.state_dict()
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()

        state_path = os.path.join(save_dir, 'state_{}.pth'.format(label))
        torch.save(state_dict, state_path)

    def load(self, save_dir, label, train):
        state_path = os.path.join(save_dir, 'state_{}.pth'.format(label))
        if not os.path.isfile(state_path):
            Visualizer.log_print('state file store in %s is not found' % state_path)
            return
        ckpt = torch.load(state_path)
        self.model.load_state_dict(ckpt['network'])
        self.model.to(self.device)
        if train:
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.optim.load_state_dict(ckpt['optimizer'])
