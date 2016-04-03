#!/usr/bin/python
import numpy as np
import sys,os
import scipy.io.wavfile
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA

from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

def write_mfcc(ceps, fn):
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".mfcc"
    np.save(data_fn, ceps)
    print("Written %s" % data_fn)
def create_mfcc(fn):
    sample_rate, X = scipy.io.wavfile.read(fn)
    ceps, mspec, spec = mfcc(X)
    write_mfcc(ceps, fn)



if __name__ == '__main__':
    filepath = "/mnt/hgfs/vmfiles/genres/"
    for genrefolder in os.listdir(filepath):
        genrefolder = filepath + genrefolder
        newgenrefolder = genrefolder.replace("genres","genresmfcc")
        os.makedirs(newgenrefolder)
        files = genrefolder
        for file in os.listdir(files):
            file = files + "/" + file
            create_mfcc(file)
