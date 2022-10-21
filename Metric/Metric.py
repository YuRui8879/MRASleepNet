import numpy as np


class Metric:
    def __init__(self):
        self.accuracy_list = {}
        self.precision_list = {}
        self.recall_list = {}
        self.F1_list = {}
        self.MF1_list = {}
        self.Kappa_list = {}
        self.confusion_matric = {}
        self.loss_list = {}
        self.idx = 0

    def __call__(self,pred,real):
        C = self.cal_confusion_matric(pred,real)
        pre = self.cal_precision(pred,real)
        rec = self.cal_recall(pred,real)
        acc = self.cal_accuracy(pred,real)
        F1 = self.cal_F1(pred,real)
        MF1 = self.cal_marco_F1(pred,real)
        kappa = self.cal_kappa(pred,real)
        self.confusion_matric[self.idx] = C
        self.precision_list[self.idx] = pre
        self.recall_list[self.idx] = rec
        self.accuracy_list[self.idx] = acc
        self.F1_list[self.idx] = F1
        self.MF1_list[self.idx] = MF1
        self.Kappa_list[self.idx] = kappa

    def step(self):
        self.idx += 1

    def append_loss(self,loss):
        self.loss_list[self.idx] = loss

    def get_mean_loss(self):
        loss = []
        for k,v in self.loss_list.items():
            loss.append(v)
        return np.mean(loss)

    def get_mean_confusion_matric(self):
        C = np.zeros((5,5))
        for k,v in self.confusion_matric.items():
            C += v
        return C

    def get_mean_precision(self):
        pre = []
        for k,v in self.precision_list.items():
            pre.append(v)
        return np.mean(pre)

    def get_mean_recall(self):
        rec = []
        for k,v in self.recall_list.items():
            rec.append(v)
        return np.mean(rec)

    def get_mean_F1(self):
        F1 = []
        for k,v in self.F1_list.items():
            F1.append(v)
        return np.mean(F1)

    def get_mean_MF1(self):
        MF1 = []
        for k,v in self.MF1_list.items():
            MF1.append(v)
        return np.mean(MF1)

    def get_mean_Kappa(self):
        kappa = []
        for k,v in self.Kappa_list.items():
            kappa.append(v)
        return np.mean(kappa)

    def get_mean_accuracy(self):
        acc = []
        for k,v in self.accuracy_list.items():
            acc.append(v)
        return np.mean(acc)

    def get_loss(self,idx = -1):
        if idx > self.idx:
            raise Exception('idx is out of the length of Metric')
        max_idx = -1
        if idx == -1:
            for k,v in self.loss_list.items():
                if k > max_idx:
                    max_idx = k
        else:
            max_idx = idx
        return self.loss_list[max_idx]

    def get_confusion_matric(self,idx = -1):
        if idx > self.idx:
            raise Exception('idx is out of the length of Metric')
        max_idx = -1
        if idx == -1:
            for k,v in self.confusion_matric.items():
                if k > max_idx:
                    max_idx = k
        else:
            max_idx = idx
        return self.confusion_matric[max_idx]

    def get_precision(self,idx = -1):
        if idx > self.idx:
            raise Exception('idx is out of the length of Metric')
        max_idx = -1
        if idx == -1:
            for k,v in self.precision_list.items():
                if k > max_idx:
                    max_idx = k
        else:
            max_idx = idx
        return self.precision_list[max_idx]

    def get_recall(self,idx = -1):
        if idx > self.idx:
            raise Exception('idx is out of the length of Metric')
        max_idx = -1
        if idx == -1:
            for k,v in self.recall_list.items():
                if k > max_idx:
                    max_idx = k
        else:
            max_idx = idx
        return self.recall_list[max_idx]

    def get_F1(self,idx = -1):
        if idx > self.idx:
            raise Exception('idx is out of the length of Metric')
        max_idx = -1
        if idx == -1:
            for k,v in self.F1_list.items():
                if k > max_idx:
                    max_idx = k
        else:
            max_idx = idx
        return self.F1_list[max_idx]

    def get_MF1(self,idx = -1):
        if idx > self.idx:
            raise Exception('idx is out of the length of Metric')
        max_idx = -1
        if idx == -1:
            for k,v in self.MF1_list.items():
                if k > max_idx:
                    max_idx = k
        else:
            max_idx = idx
        return self.MF1_list[max_idx]

    def get_kappa(self,idx = -1):
        if idx > self.idx:
            raise Exception('idx is out of the length of Metric')
        max_idx = -1
        if idx == -1:
            for k,v in self.Kappa_list.items():
                if k > max_idx:
                    max_idx = k
        else:
            max_idx = idx
        return self.Kappa_list[max_idx]

    def get_accuracy(self,idx = -1):
        if idx > self.idx:
            raise Exception('idx is out of the length of Metric')
        max_idx = -1
        if idx == -1:
            for k,v in self.accuracy_list.items():
                if k > max_idx:
                    max_idx = k
        else:
            max_idx = idx
        return self.accuracy_list[max_idx]

    def cal_confusion_matric(self,pred,real):
        C = np.zeros((5,5))
        for i in range(len(pred)):
            C[int(pred[i]),int(real[i])] += 1
        return C

    def cal_accuracy(self,pred,real):
        C = self.cal_confusion_matric(pred,real)
        sum_matrix = np.sum(np.reshape(C,(C.size,)))
        diag = np.trace(C)
        if sum_matrix == 0:
            return 0
        else:
            return diag/sum_matrix

    def cal_precision(self,pred,real):
        C = self.cal_confusion_matric(pred,real)
        pre = []
        for i in range(C.shape[0]):
            TP = C[i,i]
            TP_FP = np.sum(C[:,i])
            if TP_FP == 0:
                pre.append(0)
            else:
                pre.append(np.round(TP/TP_FP,4))
        return pre

    def cal_recall(self,pred,real):
        C = self.cal_confusion_matric(pred,real)
        rec = []
        for i in range(C.shape[0]):
            TP = C[i,i]
            TP_FN = np.sum(C[i,:])
            if TP_FN == 0:
                rec.append(0)
            else:
                rec.append(np.round(TP/TP_FN,4))
        return rec

    def cal_F1(self,pred,real):
        pre = self.cal_precision(pred,real)
        rec = self.cal_recall(pred,real)
        if len(pre) != len(rec):
            raise Exception('The length of Recall and Precision is not match')
        F1 = []
        for i in range(len(pre)):
            den = pre[i]+rec[i]
            if den == 0:
                F1.append(0)
            else:
                F1.append(np.round(2*pre[i]*rec[i]/(den),4))
        return F1

    def cal_marco_F1(self,pred,real):
        F1 = self.cal_F1(pred,real)
        return np.mean(F1)

    def cal_kappa(self,pred,real):
        C = self.cal_confusion_matric(pred,real)
        acc = self.cal_accuracy(pred,real)
        pe_numer = 0
        sum_matrix = np.array(np.sum(np.reshape(C,(C.size,))),dtype = np.float64)
        for i in range(C.shape[0]):
            c1 = np.array(np.sum(C[:,i]),dtype = np.float64)
            c2 = np.array(np.sum(C[i,:]),dtype = np.float64)
            pe_numer += c1 * c2
        pe = pe_numer/(sum_matrix*sum_matrix)
        if pe == 1:
            return 0
        else:
            return (acc - pe)/(1 - pe)