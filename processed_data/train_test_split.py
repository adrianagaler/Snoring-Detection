import numpy as np
import csv
import os

def train_test_split(sizes='80_20_0',seed = 66):
    np.random.seed(seed)
    for i in ['train','test','val']:
        fp = '{}.csv'.format(i)
        if os.path.exists(fp):os.remove(fp)
    N = 500
    split_sizes = sizes.split('_')
    train_s, test_s,val_s = int(split_sizes[0]), int(split_sizes[1]), int(split_sizes[2])
    assert train_s+test_s+val_s==100,"Split sizes must sum to 100"
    y0 = np.random.permutation(N)
    y1 = np.random.permutation(N)
    with open('train.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(int(train_s*1./100*N)):
            writer.writerow([0,y0[i]])
        for i in range(int(train_s*1./100*N)):
            writer.writerow([1,y1[i]])
    with open('test.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(int(train_s*1./100*N),int(train_s*1./100*N)+int(test_s*1./100*N)):
            writer.writerow([0,y0[i]])
        for i in range(int(train_s*1./100*N),int(train_s*1./100*N)+int(test_s*1./100*N)):
            writer.writerow([1,y1[i]])
    if val_s>0:
        with open('val.csv','w') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(int(train_s*1./100*N)+int(test_s*1./100*N),N):
                writer.writerow([0,y0[i]])
            for i in range(int(train_s*1./100*N)+int(test_s*1./100*N),N):
                writer.writerow([1,y1[i]])
train_test_split('80_10_10')

