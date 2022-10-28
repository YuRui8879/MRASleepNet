'''
https://github.com/akaraspt/deepsleepnet
Copyright 2017 Akara Supratak and Hao Dong.  All rights reserved.
'''

import argparse
import glob
import math
import ntpath
import os
import shutil


from datetime import datetime

import numpy as np
import pandas as pd

from mne.io import concatenate_raws, read_raw_edf

import dhedfreader

from scipy.io import savemat


# Label values
UNKNOWN = -1

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
    "Sleep stage ?": -1,
    "Movement time": -1
}

EPOCH_SEC_SIZE = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_edf_20",
                        help="File path to the PSG and annotation files.")
    parser.add_argument("--output_dir", type=str, default="data_edf_20_npz/fpzcz",
                        help="Directory where to save numpy files outputs.")
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    # Read raw and annotation EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*Hypnogram.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for i in range(len(psg_fnames)):
        raw = read_raw_edf(psg_fnames[i], preload=True, stim_channel=None)
        sampling_rate = raw.info['sfreq']
        raw_ch_df = raw.to_data_frame(scalings=100.0)['EEG Fpz-Cz']
        raw_ch_df = raw_ch_df.to_frame()
        raw_ch_df.set_index(np.arange(len(raw_ch_df)))

        fpz_cz_df = raw_ch_df

        pz_oz_df = raw.to_data_frame(scalings=100.0)['EEG Pz-Oz']
        pz_oz_df = pz_oz_df.to_frame()
        pz_oz_df.set_index(np.arange(len(pz_oz_df)))

        eog_df = raw.to_data_frame(scalings=100.0)['EOG horizontal']
        eog_df = eog_df.to_frame()
        eog_df.set_index(np.arange(len(eog_df)))

        # Get raw header
        f = open(psg_fnames[i], 'r', errors='ignore')
        reader_raw = dhedfreader.BaseEDFReader(f)
        reader_raw.read_header()
        h_raw = reader_raw.header
        f.close()
        raw_start_dt = datetime.strptime(h_raw['date_time'], "%Y-%m-%d %H:%M:%S")

        # Read annotation and its header
        f = open(ann_fnames[i], 'r', errors='ignore')
        reader_ann = dhedfreader.BaseEDFReader(f)
        reader_ann.read_header()
        h_ann = reader_ann.header
        _, _, ann = zip(*reader_ann.records())
        f.close()
        ann_start_dt = datetime.strptime(h_ann['date_time'], "%Y-%m-%d %H:%M:%S")

        # Assert that raw and annotation files start at the same time
        assert raw_start_dt == ann_start_dt

        # Generate label and remove indices
        remove_idx = []    # indicies of the data that will be removed
        labels = []        # indicies of the data that have labels
        label_idx = []
        for a in ann[0]:
            onset_sec, duration_sec, ann_char = a
            ann_str = "".join(ann_char)
            label = ann2label[ann_str[2:-1]]
            if label != UNKNOWN:
                if duration_sec % EPOCH_SEC_SIZE != 0:
                    raise Exception("Something wrong")
                duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
                label_epoch = np.ones(duration_epoch, dtype=np.int) * label
                labels.append(label_epoch)
                idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=np.int)
                label_idx.append(idx)

                print ("Include onset:{}, duration:{}, label:{} ({})".format(
                    onset_sec, duration_sec, label, ann_str
                ))
            else:
                idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=np.int)
                remove_idx.append(idx)

                print ("Remove onset:{}, duration:{}, label:{} ({})".format(
                    onset_sec, duration_sec, label, ann_str))
        labels = np.hstack(labels)
        
        print ("before remove unwanted: {}".format(np.arange(len(raw_ch_df)).shape))
        if len(remove_idx) > 0:
            remove_idx = np.hstack(remove_idx)
            select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)
        else:
            select_idx = np.arange(len(raw_ch_df))
        print ("after remove unwanted: {}".format(select_idx.shape))

        # Select only the data with labels
        print ("before intersect label: {}".format(select_idx.shape))
        label_idx = np.hstack(label_idx)
        select_idx = np.intersect1d(select_idx, label_idx)
        print ("after intersect label: {}".format(select_idx.shape))

        # Remove extra index
        if len(label_idx) > len(select_idx):
            print("before remove extra labels: {}, {}".format(select_idx.shape, labels.shape))
            extra_idx = np.setdiff1d(label_idx, select_idx)
            # Trim the tail
            if np.all(extra_idx > select_idx[-1]):
                # n_trims = len(select_idx) % int(EPOCH_SEC_SIZE * sampling_rate)
                # n_label_trims = int(math.ceil(n_trims / (EPOCH_SEC_SIZE * sampling_rate)))
                n_label_trims = int(math.ceil(len(extra_idx) / (EPOCH_SEC_SIZE * sampling_rate)))
                if n_label_trims!=0:
                    # select_idx = select_idx[:-n_trims]
                    labels = labels[:-n_label_trims]
            print("after remove extra labels: {}, {}".format(select_idx.shape, labels.shape))

        # Remove movement and unknown stages if any
        fpz_cz = fpz_cz_df.values[select_idx]
        pz_oz = pz_oz_df.values[select_idx]
        eog = eog_df.values[select_idx]

        # Get epochs and their corresponding labels
        fpz_cz = fpz_cz.astype(np.float64)
        pz_oz = pz_oz.astype(np.float64)
        eog = eog.astype(np.float64)
        y = labels.astype(np.float64)

        assert len(fpz_cz) == len(y) * sampling_rate * 30

        # Save
        filename = ntpath.basename(psg_fnames[i]).replace("-PSG.edf", ".mat")
        save_dict = {
            "fpz_cz": fpz_cz, 
            "pz_oz": pz_oz, 
            "eog": eog,
            "fs": sampling_rate,
            "label": y
        }
        savemat(os.path.join(args.output_dir, filename), save_dict)

        print ("\n=======================================\n")


if __name__ == "__main__":
    main()
