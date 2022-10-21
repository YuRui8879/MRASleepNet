import argparse
import os
import random

def combine_sample(path,types = 'ALL'):
    sample = []
    dirs = os.listdir(path)
    for files in dirs:
        if types == 'ALL':
            if files[5] == '1':
                tmp_name = files[:5] + '2' + files[6:]
                if tmp_name in dirs:
                    sample.append([files,tmp_name])
                else:
                    sample.append([files])
        elif types == 'SC':
            if files[5] == '1' and files[:2] == 'SC':
                tmp_name = files[:5] + '2' + files[6:]
                if tmp_name in dirs:
                    sample.append([files,tmp_name])
                else:
                    sample.append([files])
        elif types == 'ST':
            if files[5] == '1' and files[:2] == 'ST':
                tmp_name = files[:5] + '2' + files[6:]
                if tmp_name in dirs:
                    sample.append([files,tmp_name])
                else:
                    sample.append([files])
    return sample

def split_dataset(data_path,output_path,sample,folds,fold_count = 0):
    step = max(1,int(len(sample)/folds))
    test_sample = sample[fold_count*step:(fold_count+1)*step]
    all_sample = []
    for samp in sample:
        all_sample.extend(samp)
    res_test_sample = []
    for samp in test_sample:
        res_test_sample.extend(samp)
    res_train_sample = list(set(all_sample) - set(res_test_sample))

    with open(os.path.join(output_path,'train.txt'),'w') as f:
        for train_sample in res_train_sample:
            f.write(os.path.join(data_path,train_sample) + '\n')

    with open(os.path.join(output_path,'test.txt'),'w') as f:
        for test_sample in res_test_sample:
            f.write(os.path.join(data_path,test_sample) + '\n')

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', type=int, default=10, help='random seed')
parser.add_argument('-f', '--file_type', choices=['SC','ST','All'], default='SC', help = 'Select whether to read sc files or st files')
parser.add_argument('-d', '--data_path', help = 'The path to store data')
parser.add_argument('-o', '--output_path', help = 'The path to output train data file and test data file')
parser.add_argument('-fd', '--folds', type=int, default=10, help = 'K folds cross validation')
parser.add_argument('-fi', '--fold_idx', type=int, default=0, help = 'The fold_idx fold in the K-fold cross validation')
args = parser.parse_args()

file_type = args.file_type
data_path = args.data_path
seed = args.seed
folds = args.folds
fold_idx = args.fold_idx
output_path = args.output_path

sample = combine_sample(data_path,types=file_type)
random.seed(seed)
random.shuffle(sample)
split_dataset(data_path,output_path,sample,folds,fold_idx)

print('split data finished!')