import os
from mne.io import read_raw_edf
from scipy.io import savemat,loadmat
import datetime
import argparse

def read_data(name,load_data_path,lead = 'Fp2-F4'):
    data_path = os.path.join(load_data_path, name + '.edf')
    raw = read_raw_edf(data_path,preload = True)
    data,times = raw.get_data(return_times=True)
    fs = raw.info['sfreq']
    start_time = raw.info['meas_date'].replace(tzinfo=None)
    label = raw.info['ch_names']
    label_idx = label.index(lead)

    return data[label_idx,:],fs,start_time

def read_label(name,load_data_path,load_label_path):
    time_path = os.path.join(load_data_path,name + '.txt')
    label_path = os.path.join(load_label_path,'hyp' + name + '.mat')
    times = []
    time_flag = 0
    with open(time_path,'r') as f:
        for line in f.readlines():
            line = line.split('\t')
            if line[0] == 'Recording Date:':
                rddate = line[-1] + ' '
            if line[0] == 'Sleep Stage':
                time_idx = line.index('Time [hh:mm:ss]')
                time_flag = 1
                continue
            if time_flag:
                times.append(rddate + line[time_idx])

    start_time = datetime.datetime.strptime(times[0],'%d/%m/%Y %H:%M:%S')
    end_time = datetime.datetime.strptime(times[-1],'%d/%m/%Y %H:%M:%S') + datetime.timedelta(days=1) + datetime.timedelta(seconds=30)
    mat = loadmat(label_path)
    label = mat['hyp'][:,0]
    duration_time = (end_time - start_time).seconds
    return label,start_time,duration_time

def gen_data(load_data_path,load_label_path,out_path,sel_ch):
    fs_list = []
    for files in os.listdir(data_path):
        if files.endswith('.edf'):
            files = files.split('.edf')[0]
            try:
                data,fs,edf_start_time = read_data(files,load_data_path,sel_ch)
                fs_list.append(fs)
                label,txt_start_time,duration_time = read_label(files,load_data_path,load_label_path)
            except Exception as f:
                print(files,'read faild!')
                print(f)
                continue
            time_diff = (txt_start_time - edf_start_time).seconds
            data = data[int(time_diff*fs):int((time_diff+duration_time)*fs)]
            if len(data) != len(label) * 30 *fs:
                print(files,'data != label')
                continue
            savemat(os.path.join(out_path,files + '.mat'),{'data':data,'label':label,'fs':fs})
            print('save',files,'success!')
    print(fs_list)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', help='The path to load data')
    parser.add_argument('-l', '--label_path', help = 'The path to load label')
    parser.add_argument('-o', '--output_path', help = 'The path to output processed mat files')
    parser.add_argument('-s', '--select_channel', default = 'Fp2-F4', help = 'Selective channel')
    args = parser.parse_args()

    gen_data(args.data_path,args.label_path,args.output_path,args.select_channel)

    


