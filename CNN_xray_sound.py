import re
import numpy
import os
import glob
import os, shutil
import math
import json
import librosa, librosa.display
import matplotlib.pyplot as plt

rawtrain_dir = r'C:\Users\hirom\Documents\GitHub\Group5_DeepLearning\genres_train'
rawtest_dir = r'C:\Users\hirom\Documents\GitHub\Group5_DeepLearning\genres_test'
train_dir = r'C:\Users\hirom\Documents\GitHub\Group5_DeepLearning\CNN_example\data\train\track'
test_dir = r'C:\Users\hirom\Documents\GitHub\Group5_DeepLearning\CNN_example\data\test\track'

SAMPLE_RATE = 22050

def adjust_data(path,finaldir):
    #access the files in the directory 
    lista_dir = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            track = os.path.join(root, name)
            lista_dir.append(track)

    #apply MFCC in each file 
    for n in range(0,len(lista_dir)+1): 
        name = lista_dir[n].split("\\")[-1]

        signal, sr = librosa.load(lista_dir[n], sr=SAMPLE_RATE)
        MFCCs = librosa.feature.mfcc(signal, n_fft= 2048,hop_length = 512, n_mfcc = 13)

        librosa.display.specshow(MFCCs, sr=sr, hop_length=512)
        plt.xlabel('Time')
        plt.ylabel('MFCCs')
        filename  = name + '.jpg'
        plt.savefig(finaldir + filename)


#create the train data dir
adjust_data(rawtrain_dir,train_dir)
#create the test data dir
adjust_data(rawtest_dir,test_dir)




        

