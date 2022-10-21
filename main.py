from Algorithm.Algorithm import Algorithm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--parallel', type=bool, default=False, help = 'Whether to use multi-GPU training')
parser.add_argument('-lp', '--log_save_path', help = 'The path to save log file')
parser.add_argument('-mp', '--model_save_path', help = 'The path to save model file')
parser.add_argument('-tp', '--split_data_file_path', help = 'The path to load split data file list')
parser.add_argument('-b', '--batch_size', type=int, default=128, help = 'Batch size')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help = 'Learning rate')
parser.add_argument('-c', '--cuda_device', type=int, default=0, help = 'If you do not use multi-GPU training, you need to specify the GPU')
parser.add_argument('-r', '--reg_parameter', type=float, default=0.001, help = 'Parameter for L2 regularization. If set to 0, L2 regularization is not used')
parser.add_argument('-e', '--epochs', type=int, default=100, help='Max iteration')
args = parser.parse_args()

parallel = args.parallel
log_save_path = args.log_save_path
model_save_path = args.model_save_path
split_data_file_path = args.split_data_file_path
batch_size = args.batch_size
learning_rate = args.learning_rate
cuda_device = args.cuda_device
reg_parameter = args.reg_parameter
epochs = args.epochs

algorithm = Algorithm(model_save_path,log_save_path,split_data_file_path,batch_size = batch_size, learning_rate = learning_rate, epochs = epochs, 
                cuda_device = cuda_device,reg_para = reg_parameter, parallel = parallel)
algorithm.train()
algorithm.test()