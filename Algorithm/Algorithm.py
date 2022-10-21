import sys
sys.path.append('..')  #加入路径，添加目录

import torch
import torch.nn as nn
import torch.optim as optim
from Model.MRASleepNet import MRASleepNet
from DataAdapter.DataAdapter import TrainAdapter,TestAdapter,BaseAdapter
import torch.utils.data as Data
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import numpy as np
np.set_printoptions(suppress = True)
import os
from Log import Log
from Metric.Metric import Metric

class Algorithm():

    def __init__(self,model_save_path,log_save_path,split_data_file_path,batch_size,learning_rate,epochs,cuda_device,reg_para,parallel = True):
        super(Algorithm,self).__init__()
        self.save_path = model_save_path # The path to save model
        self.parallel = parallel # Whether to use multi GPU calculation
        self.log = Log(log_save_path) # The path to save log file
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reg_parameter = reg_para # L2 regularization parameter
        self.cuda_device = cuda_device # Designated Training GPU
        self.base_adapter = BaseAdapter(split_data_file_path)

        # init model
        self.model = MRASleepNet()
        if self.parallel:
            self.model = nn.DataParallel(self.model)
            self.model.cuda()
        else:
            self.model.cuda(self.cuda_device)
        
        # init optimizer
        self.optimizer = optim.Adam(self.model.parameters(),lr = self.learning_rate)
        self.clr = CosineAnnealingLR(self.optimizer,T_max = 20)
        if self.reg_parameter == 0:
            self.reg_loss = None
        else:
            self.reg_loss = Regularization(self.model,self.reg_parameter)

    def train(self):

        # load data
        train_adapter = TrainAdapter(self.base_adapter)
        train_loader = Data.DataLoader(train_adapter,batch_size=self.batch_size,shuffle=True,num_workers=0)
        test_adapter = TestAdapter(self.base_adapter)
        test_loader = Data.DataLoader(test_adapter,batch_size=self.batch_size,shuffle=False,num_workers=0)

        train_metric = Metric()
        test_metric = Metric()

        # cal weight
        if self.parallel:
            weight = torch.FloatTensor(train_adapter.calc_class_weight()).cuda()
        else:
            weight = torch.FloatTensor(train_adapter.calc_class_weight()).cuda(self.cuda_device)

        self.criterion = nn.CrossEntropyLoss(weight)

        # train
        for epoch in range(1,self.epochs + 1):
            start_time = time.time()
            self.cal_batch(train_loader,train_metric,'train')
            self.cal_batch(test_loader,test_metric,'valid')
            end_time = time.time()
            self.log.log('- Epoch: {:d} - Train_loss: {:.5f} - Train_mean_acc: {:.5f} - Test_loss: {:.5f} - Test_mean_acc: {:.5f} - T_Time: {:.3f}'.format\
                (epoch,train_metric.get_loss(),train_metric.get_accuracy(),test_metric.get_loss(),test_metric.get_accuracy(),end_time - start_time))
            self.log.log('Current learning rate: {:.10f}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))
            self.clr.step()
            train_metric.step()
            test_metric.step()

        # save model
        torch.save(self.model.state_dict(), os.path.join(self.save_path,'last_model.pt'))
        self.log.log('save last model...')       
        self.log.log('train finished...')

    def cal_batch(self,loader,metric,types = 'train'):

        if types == 'train':
            self.model.train()
        elif types == 'valid':
            self.model.eval()

        res_pred = np.zeros(0)
        res_label = np.zeros(0)
        loss_list = []

        for i,data in enumerate(loader,0):
            if self.parallel:
                inputs,labels = data[0].squeeze().cuda(),data[1].squeeze().cuda()
            else:
                inputs,labels = data[0].squeeze().cuda(self.cuda_device),data[1].squeeze().cuda(self.cuda_device)

            if len(inputs.size()) == 1:
                continue

            # Forward Propagation
            outputs = self.model(inputs)
            _,pred = outputs.squeeze().max(1)

            loss = self.criterion(outputs,labels)

            if self.reg_loss:
                loss += self.reg_loss(self.model)

            if types == 'train':
                # Backward Propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            res_pred = np.hstack((res_pred,pred.detach().cpu().numpy()))
            res_label = np.hstack((res_label,labels.detach().cpu().numpy()))

            loss_list.append(loss.item())

        metric.append_loss(np.mean(loss_list))
        metric(res_pred,res_label)
        
    def test(self):
        test_adapter = TestAdapter(self.base_adapter)
        test_loader = Data.DataLoader(test_adapter,batch_size=self.batch_size,shuffle=False,num_workers=0)

        metric = Metric()

        model = MRASleepNet()
        if self.parallel:
            model = nn.DataParallel(model)
            model.cuda()
        else:
            model.cuda(self.cuda_device)
        model.load_state_dict(torch.load(os.path.join(self.save_path,'last_model.pt')))
        model.eval()

        res_predicted = np.zeros(0)
        res_label = np.zeros(0)

        start_time = time.time()
        for i,data in enumerate(test_loader,0):
            if self.parallel:
                inputs,labels = data[0].squeeze().cuda(),data[1].squeeze().cuda()
            else:
                inputs,labels = data[0].squeeze().cuda(self.cuda_device),data[1].squeeze().cuda(self.cuda_device)

            if len(inputs.size()) == 1:
                continue

            outputs = model(inputs)
            _,pred = outputs.squeeze().max(1)

            res_predicted = np.hstack((res_predicted,pred.detach().cpu().numpy()))
            res_label = np.hstack((res_label,labels.detach().cpu().numpy()))
            
        metric(res_predicted,res_label)
        end_time = time.time()
        self.log.log('T_T Time:{:.4f}'.format(end_time - start_time))
        self.log.log('Test Accuracy:{:.5f}'.format(metric.get_accuracy()))
        self.log.log('Test Precision:{}'.format(str(metric.get_precision())))
        self.log.log('Test Recall:{}'.format(str(metric.get_recall())))
        self.log.log('Test F1:{}'.format(str(metric.get_F1())))
        self.log.log('Test MF1:{:.5f}'.format(metric.get_MF1()))
        self.log.log('Test Kappa:{:.5f}'.format(metric.get_kappa()))
        self.log.log('Test confusion matrix')
        self.log.log(str(metric.get_confusion_matric()))


class Regularization(nn.Module):
    def __init__(self,model,weight_decay,p=2):
        # param model: model
        # param weight_decay: Regularization parameters
        # param p: When p=0 for L2 regularization, p=1 for L1 regularization
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        self.weight_info(self.weight_list)

 
    def to(self,device):
        self.device=device
        super().to(device)
        return self
 
    def forward(self, model):
        self.weight_list=self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
 
    def get_weight(self,model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self,weight_list, weight_decay, p=2):
        reg_loss=0

        for name, w in weight_list:
            if p == 2 or p == 0:
                reg_loss += torch.sum(torch.pow(w, 2))
            else:
                reg_loss += torch.sum(torch.abs(w))
 
        reg_loss=weight_decay*reg_loss
        return reg_loss
 
    def weight_info(self,weight_list):
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")