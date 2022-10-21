# MRASleepNet

## Cite

This repository provides the code for "MRASleepNet: A multi-resolution attention network for sleep stage classification using single-channel EEG"

Our work now was Accepted by the Journal of Neural Engineering. If you want to cite this article, please use ""

## Description

### Overall

The signal pre-processing was to normalize and cut the EEG data into 30-s segments. For data enhancement, we used a contextual enhancement method to enhance the data by merging information from adjacent samplessegments. After that, the enhanced data were fed into the MRASleepNet model and different metrics were used to evaluate the performance of the model.

![Overall](https://github.com/YuRui8879/MRASleepNet/blob/master/fig/overall.png)

### Data Enhancement

We have augmented the data using the adjacent sleep states.

![DataEnhancement](https://github.com/YuRui8879/MRASleepNet/blob/master/fig/dataenhance.jpg)

### Model

Our MRASleepNet comprised three major modules: a feature extraction (FE) module, an MRA module, and a gMLP module. 

The FE module used convolution to abstract and refine the features in the enhanced EEG signals to obtain a feature vector. 

The MRA module further abstracted the features and modeled the importance of the features so that the model focused on key features. 

The gMLP module was used to extract temporal information from the EEG signals and model the temporal relationship of the features. 

![Model](https://github.com/YuRui8879/MRASleepNet/blob/master/fig/model.jpg)

### Result

Comparison of results with other models

![Result](https://github.com/YuRui8879/MRASleepNet/blob/master/fig/result.jpg)

## Usage

### Database

The open source database used in this repository is from

* https://www.physionet.org/content/sleep-edfx/1.0.0/
* https://www.physionet.org/content/capslpdb/1.0.0/

You need to download the database locally, and Physionet offers different ways to do so.

### Requirements

Switch to this repository directory, download the required dependencies via pip

```python
pip install -r requirements
```

### Convert data format

By running
```python
python prepare_physionet.py --data_dir . --output_dir .
```
Convert edf format data in the database to mat format. **data_dir** is the path of the original data downloaded, **output_dir** is the path to save the mat file after conversion.

### Data set partition

By running
```python
python split_data.py --data_path . --output_path .
```
partitioning the data set, Where **data_path** is the generated mat file directory and **output_path** is the output dataset divided txt file directory.

split_data.py also offers the following optional parameters
```python
--seed          Random seed
--file_type     choice = ['SC','ST','ALL'], Select whether to use SC files, ST files or all files
--folds         The K for K-fold cross validation
--fold_idx      The fold_idx round in the K-fold cross validation
```

### Train and Test

Run
```python
python main.py --log_save_path . --model_save_path . --split_data_file_path .
```
for training and testing, where **log_save_path** is the path to save log file, **model_save_path** is the path to save trained model and **split_data_file_path** is the path to save txt file of dataset partition results.

main.py provides the following optional parameters
```python
--parallel        Whether to use multi-GPU training
--batch_size      Batch size
--learning_rate   Learning_rate
--epochs          Max iteration
--cuda_device     If you do not use multi-GPU training, you need to specify the GPU
-- reg_parameter  Parameter for L2 regularization. If set to 0, L2 regularization is not used
```