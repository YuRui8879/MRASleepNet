import numpy as np
import torch
from torch.utils.data import Dataset
import os
from scipy.io import loadmat
import math

class BaseAdapter:
    def __init__(self,txt_path):
        self.train_set = self.read_txt(os.path.join(txt_path,'train.txt'))
        self.test_set = self.read_txt(os.path.join(txt_path,'test.txt'))
    
    def get_train_set(self):
        return self.train_set

    def get_test_set(self):
        return self.test_set

    # read split data txt file
    def read_txt(self,txt_path):
        with open(txt_path,'r') as f:
            txt = f.read()
        files = txt.strip().split('\n')
        return files
        
    def get_X_Y(self,sample):
        X = []
        Y = []
        for samp in sample:
            print(samp)
            fea,la = self.fea_label(samp)
            X.append(fea)
            Y.append(la)

        return torch.FloatTensor(np.concatenate(X)),torch.LongTensor(np.concatenate(Y))

    # Standardization Methods
    def normal(self,sig):
        return (sig - np.mean(sig))/np.max(np.abs(sig))

    # Read mat files
    def read_mat(self,path):
        data = loadmat(path)
        segEEG = data['fpz_cz'].squeeze()
        label = data['label'].squeeze()
        return segEEG,label

    # This method is used to generate segment data and labels
    def fea_label(self,samp,win_len = 3000, step = 3000):
        fea = []
        label = []
        segEEG,la = self.read_mat(samp)
        segEEG,la = self.preprocess(segEEG,la)
        segEEG = self.normal(segEEG)
        tmp_fea = []
        tmp_la = []
        for i in range(0,len(segEEG) - win_len + 1,step):
            if i == 0:
                tmp_fea.append(self.expand_data(segEEG[:win_len],segEEG[:win_len],segEEG[win_len:2*win_len],0))
            elif i == len(segEEG) - win_len:
                tmp_fea.append(self.expand_data(segEEG[-2*win_len:-win_len],segEEG[-win_len:],segEEG[-win_len:],2))
            else:
                tmp_fea.append(self.expand_data(segEEG[i-win_len:i],segEEG[i:i+win_len],segEEG[i+win_len:i+2*win_len],1))
            # tmp_fea.append(segEEG[i:i+win_len])
            tmp_la.append(la[i//step])

        assert len(tmp_fea) == len(tmp_la)
        return np.stack(tmp_fea),np.array(tmp_la)

    # This method is used to enhance the data
    def expand_data(self,last_data,data,next_data,mode = 0):
        split_len = 500
        if mode == 0:
            reverse_data = data[::-1]
            tmp = np.hstack((reverse_data[-split_len:],data,next_data[:split_len]))
        elif mode == 1:
            tmp = np.hstack((last_data[-split_len:],data,next_data[:split_len]))
        elif mode == 2:
            reverse_data = data[::-1]
            tmp = np.hstack((last_data[-split_len:],data,reverse_data[:split_len]))
        return tmp

    # Data pre-processing. The labels were merged according to AASM, and only the data before and after 30 s of sleep were retained
    def preprocess(self,segEEG,label,win_len = 3000):
        label = np.array(label)
        label = np.where(label == 4,3,label)
        label = np.where(label == 5,4,label)
        idx = np.where(label > 0)
        max_idx = np.max(idx)
        min_idx = np.min(idx)
        segEEG = segEEG[max(0,min_idx * win_len - 60 * win_len):min(max_idx * win_len + 60 * win_len,len(segEEG))]
        # segEEG = segEEG/10000
        # segEEG = segEEG - np.mean(segEEG)
        label = label[max(0,min_idx - 60):min(max_idx + 60,len(label))]
        idx = np.where(label == -1)
        label = np.delete(label,idx[0])
        del_list = []
        [del_list.extend(list(range(i*win_len,(i+1)*win_len))) for i in idx[0]]
        segEEG = np.delete(segEEG,del_list)
        
        return segEEG,label


class TrainAdapter(Dataset):
    def __init__(self,base_adapter):
        super(TrainAdapter,self).__init__()
        self.train_X,self.train_Y = base_adapter.get_X_Y(base_adapter.get_train_set())

    # Calculate the weight of cross-entropy
    def calc_class_weight(self):
        labels_count = self.count_label(False)
        total = np.sum(labels_count)
        class_weight = dict()
        num_classes = len(labels_count)

        factor = 1 / num_classes
        mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5] # loss function weight

        for key in range(num_classes):
            score = math.log(mu[key] * total / float(labels_count[key]))
            class_weight[key] = score if score > 1.0 else 1.0
            class_weight[key] = round(class_weight[key] * mu[key], 2)

        class_weight = [class_weight[i] for i in range(num_classes)]

        return class_weight

    # Distribution of labels in the statistical training set
    def count_label(self,is_rate = True):
        count = np.zeros(5)
        for i in range(len(count)):
            count[i] += np.sum(np.where(np.array(self.train_Y) == i,1,0))
        if is_rate:
            rate = np.sum(count)/count
        else:
            rate = count
        return rate

    def __getitem__(self,index):
        return self.train_X[index],self.train_Y[index]

    def __len__(self):
        return len(self.train_Y)

class TestAdapter(Dataset):
    def __init__(self,base_adapter):
        super(TestAdapter,self).__init__()
        self.test_X,self.test_Y = base_adapter.get_X_Y(base_adapter.get_test_set())

    def __getitem__(self,index):
        return self.test_X[index],self.test_Y[index]

    def __len__(self):
        return len(self.test_Y)


if __name__ == '__main__':
    import torch.utils.data as Data
    ba = BaseAdapter(r'C:\Users\yurui\Desktop\item\brainstateindex\code\opensource')
    test_adapter = TestAdapter(ba)
    test_loader = Data.DataLoader(test_adapter,batch_size=128,shuffle=False,num_workers=0)
    print(len(test_adapter))