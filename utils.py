import numpy as np
import csv
import ipdb
import os


def load_dataset(main_path = './',preprocess = 'dct'):
    train_y = []
    test_y = []
    val_y  = []
    train_x = []
    test_x = []
    val_x = []
    with open(main_path+'processed_data/train.csv', 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        for row in reader:
            train_y.append(int(row[0]))
            train_x.append(np.load('{}processed_data/{}/{}_{}{}.npy'.format(main_path,row[0],row[0],row[1],preprocess)))
    with open(main_path+'processed_data/test.csv', 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        for row in reader:
            test_y.append(int(row[0]))
            test_x.append(np.load('{}processed_data/{}/{}_{}{}.npy'.format(main_path,row[0],row[0],row[1],preprocess)))
    if os.path.exists(main_path+'processed_data/val.csv'):
        with open(main_path+'processed_data/val.csv', 'r') as file:
            reader = csv.reader(file, delimiter = ',')
            for row in reader:
                val_y.append(int(row[0]))
                val_x.append(np.load('{}processed_data/{}/{}_{}{}.npy'.format(main_path,row[0],row[0],row[1],preprocess)))
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    train_x = np.array(train_x)
    test_x = np.array(test_x)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    # if psd take log
    #if preprocess == 'psd': 
    #    train_x = np.log(train_x)
    #    test_x = np.log(test_x)
    
    # standard normalization
    x_mean = np.mean(train_x,axis=0)
    x_std = np.std(train_x,axis=0)
    train_x = (train_x-x_mean)/x_std
    test_x = (test_x-x_mean)/x_std

    # also normalize validation dataset 
    if len(val_y)>0:
        if preprocess=='psd': val_x = np.log(val_x)
        val_x = (val_x-x_mean)/x_std
    return train_x, test_x, val_x, train_y, test_y, val_y, x_mean, x_std
