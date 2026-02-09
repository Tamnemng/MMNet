import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .processor import Processor # Kế thừa class gốc

class REC_Processor(Processor):
    def load_model(self):
        self.model = self.io.load_model(self.arg.model, **self.arg.model_args)
        self.model = self.model.cuda(self.output_device)
        self.loss = nn.CrossEntropyLoss().cuda(self.output_device)

    def train(self):
        self.model.train()
        self.adjust_learning_rate(self.epoch, self.arg.step, self.arg.base_lr)
        loader = self.data_loader['train']
        loss_value = []

        for data, label, _ in tqdm(loader):
            data = data.float().cuda(self.output_device)
            label = label.long().cuda(self.output_device)

            output = self.model(data)
            loss = self.loss(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.item())

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.io.print_log(f'Training loss: {self.epoch_info["mean_loss"]:.4f}')

    def test(self):
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        with torch.no_grad():
            for data, label, _ in tqdm(loader):
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)

                output = self.model(data)
                loss = self.loss(output, label)
                
                loss_value.append(loss.item())
                result_frag.append(output.data.cpu().numpy())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)
        
        # Tính Top-1 Accuracy
        predict_label = np.argmax(self.result, axis=1)
        acc = np.sum(predict_label == self.label) / len(self.label)
        
        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.epoch_info['val_acc'] = acc
        self.io.print_log(f'Evaluation Acc: {acc:.2%}')

    # Cần override hàm start để gọi đúng train/test loop mới
    def start(self):
        self.io.print_log(f'Parameters:\n{str(vars(self.arg))}')
        self.io.print_log(f'Work dir: {self.arg.work_dir}')
        self.load_model()
        self.load_optimizer()
        self.load_data()
        
        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            self.epoch = epoch
            self.train()
            if epoch % self.arg.eval_interval == 0:
                self.test()