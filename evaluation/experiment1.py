#!/usr/bin/python
from yaafelib import *
import numpy as np
import sys,os,glob
import scipy.io.wavfile
import sklearn.svm
from sklearn.preprocessing import MinMaxScaler
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
from sklearn.metrics import confusion_matrix
from matplotlib import pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import librosa

def get_mfcc(file):
    #audiofile = "/mnt/hgfs/vmfiles/genres/pop/pop.00001.wav"
    fp = FeaturePlan(sample_rate=22050)
    fp.addFeature('mfcc: MFCC blockSize=512 stepSize=256')
    #fp.addFeature('sr: SpectralRolloff blockSize=512 stepSize=256')
    #fp.addFeature('sf: SpectralFlux blockSize=512 stepSize=256')
    engine = Engine()
    engine.load(fp.getDataFlow())
    afp = AudioFileProcessor()
    afp.processFile(engine,file)
    feats = engine.readAllOutputs()
    ceps = feats['mfcc']
    num_ceps = len(ceps)
    
    #return np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0)
    return np.mean(ceps, axis=0)

def test_feature():
    file = "/mnt/hgfs/vmfiles/genres/pop/pop.00002.wav"
    fp = FeaturePlan(sample_rate=22050)
    fp.addFeature('mfcc: MFCC blockSize=512 stepSize=256')#13
    fp.addFeature('sr: SpectralRolloff blockSize=512 stepSize=256')#1
    fp.addFeature('sf: SpectralFlux blockSize=512 stepSize=256')#1
    fp.addFeature('scfp: SpectralCrestFactorPerBand FFTLength=0  FFTWindow=Hanning  blockSize=512  stepSize=256')#19
    fp.addFeature('sf1: SpectralFlatness FFTLength=0  FFTWindow=Hanning  blockSize=512  stepSize=256')#1
    fp.addFeature('sc: SpectralShapeStatistics FFTLength=0  FFTWindow=Hanning  blockSize=512  stepSize=256') #4
    fp.addFeature('sfp: SpectralFlatnessPerBand FFTLength=0  FFTWindow=Hanning  blockSize=512  stepSize=256') #19
    fp.addFeature('energy: Energy blockSize=512  stepSize=256')#1
    fp.addFeature('loudness: Loudness FFTLength=0  FFTWindow=Hanning  LMode=Relative  blockSize=512  stepSize=256')#24
    fp.addFeature('ms: MagnitudeSpectrum FFTLength=0  FFTWindow=Hanning  blockSize=512  stepSize=256') #257
    fp.addFeature('ps: PerceptualSharpness FFTLength=0  FFTWindow=Hanning  blockSize=512  stepSize=256')#1
    fp.addFeature('zcr:ZCR blockSize=512  stepSize=256')#1
    engine = Engine()
    engine.load(fp.getDataFlow())
    afp = AudioFileProcessor()
    afp.processFile(engine,file)
    feats = engine.readAllOutputs()
    ceps = feats['zcr']
    #print ceps.shape
    #num_ceps = len(ceps)
    
    y, sr = librosa.load(file)
    print y.shape
    print sr

    hop_length = 256
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])#384
    ac_global = librosa.util.normalize(ac_global)
    print ac_global.shape
    tempo = librosa.beat.estimate_tempo(oenv, sr=sr, hop_length=hop_length) #1
    print "tempo" , tempo
    print "tempogram" , tempogram.shape#384

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    print "tempo", tempo
    print "beat_frames", beat_frames.shape
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    print "beat_times" , beat_times.shape
    print beat_times
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    # Compute MFCC features from the raw signal
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    print "mfcc" , mfcc.shape
    # And the first-order differences (delta features)
    mfcc_delta = librosa.feature.delta(mfcc)
    print "mfcc_delta" , mfcc_delta.shape
    # Stack and synchronize between beat events
    # This time, we'll use the mean value (default) instead of median
    beat_mfcc_delta = librosa.feature.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

    print "beat_mfcc_delta" , beat_mfcc_delta.shape

    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr) #12

    c = np.mean(chromagram,axis=1)
    print "c", c.shape
    print "chromagram" , chromagram.shape
    r = calc_statistical_features(chromagram)
    print r.shape

    
    beat_chroma = librosa.feature.sync(chromagram, beat_frames, aggregate=np.median)
    print "beat_chroma" , beat_chroma.shape
    #print beat_chroma
    # Finally, stack all beat-synchronous features together
    beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
    print "beat_features" , beat_features.shape
    beat_feature_set = np.mean(beat_features,axis =1)
    print beat_feature_set.shape
    print beat_feature_set
    #print np.mean(ceps,axis =0)
    #return np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0)

    

def get_features1(file):
    #audiofile = "/mnt/hgfs/vmfiles/genres/pop/pop.00001.wav"
    fp = FeaturePlan(sample_rate=22050)
    fp.addFeature('mfcc: MFCC blockSize=512 stepSize=256')
    #fp.addFeature('sr: SpectralRolloff blockSize=512 stepSize=256')
    #fp.addFeature('sf: SpectralFlux blockSize=512 stepSize=256')
    engine = Engine()
    engine.load(fp.getDataFlow())
    afp = AudioFileProcessor()
    afp.processFile(engine,file)
    feats = engine.readAllOutputs()
    ceps = feats['mfcc']
    num_ceps = len(ceps)
    
    

    #return np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0)
    return np.mean(ceps, axis=0)

    #tempo_feature
def get_features2(file):
    #audiofile = "/mnt/hgfs/vmfiles/genres/pop/pop.00001.wav"
    print file
    y, sr = librosa.load(file)
    #print y.shape
    #print sr

    hop_length = 256
    '''
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])#384
    ac_global = librosa.util.normalize(ac_global)
    print ac_global.shape
    
    tempo = librosa.beat.estimate_tempo(oenv, sr=sr, hop_length=hop_length) #1
    '''

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    # Compute MFCC features from the raw signal
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)    
    # And the first-order differences (delta features)
    mfcc_delta = librosa.feature.delta(mfcc)
    # Stack and synchronize between beat events
    # This time, we'll use the mean value (default) instead of median
    beat_mfcc_delta = librosa.feature.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr) #12
    beat_chroma = librosa.feature.sync(chromagram, beat_frames, aggregate=np.median)
    #print beat_chroma
    # Finally, stack all beat-synchronous features together
    beat_features = np.vstack([beat_chroma, beat_mfcc_delta]) #38
    print "beat_features" , beat_features.shape
    beat_feature_set = np.mean(beat_features,axis =1)
    print beat_feature_set.shape
    #return np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0)
    return beat_feature_set

#mfcc+chroma
def get_features3(file):
    #audiofile = "/mnt/hgfs/vmfiles/genres/pop/pop.00001.wav"
    print file
    y, sr = librosa.load(file)
    #print y.shape
    #print sr

    hop_length = 256
    '''
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])#384
    ac_global = librosa.util.normalize(ac_global)
    print ac_global.shape
    
    tempo = librosa.beat.estimate_tempo(oenv, sr=sr, hop_length=hop_length) #1
    '''

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    # Compute MFCC features from the raw signal
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    # And the first-order differences (delta features)
    mfcc_delta = librosa.feature.delta(mfcc)
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr) #12
    c = np.mean(chromagram,axis=1)
    m = np.mean(mfcc,axis=1)
    md = np.mean(mfcc_delta,axis=1)
    feature3_set = np.concatenate((m, md), axis=0)
    feature3_set = np.concatenate((feature3_set, c), axis=0)
    print feature3_set.shape
    
    
    

    #return np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0)
    return feature3_set

def get_features4(file):
    #file = "/mnt/hgfs/vmfiles/genres/pop/pop.00002.wav"
    fp = FeaturePlan(sample_rate=22050)
    fp.addFeature('mfcc: MFCC blockSize=512 stepSize=256')#13
    fp.addFeature('sr: SpectralRolloff blockSize=512 stepSize=256')#1
    #[0:15]
    fp.addFeature('sf: SpectralFlux blockSize=512 stepSize=256')#1
    fp.addFeature('scfp: SpectralCrestFactorPerBand FFTLength=0  FFTWindow=Hanning  blockSize=512  stepSize=256')#19
    fp.addFeature('sf1: SpectralFlatness FFTLength=0  FFTWindow=Hanning  blockSize=512  stepSize=256')#1
    fp.addFeature('sc: SpectralShapeStatistics FFTLength=0  FFTWindow=Hanning  blockSize=512  stepSize=256') #4
    fp.addFeature('sfp: SpectralFlatnessPerBand FFTLength=0  FFTWindow=Hanning  blockSize=512  stepSize=256') #19
    fp.addFeature('energy: Energy blockSize=512  stepSize=256')#1
    fp.addFeature('loudness: Loudness FFTLength=0  FFTWindow=Hanning  LMode=Relative  blockSize=512  stepSize=256')#24
    fp.addFeature('ms: MagnitudeSpectrum FFTLength=0  FFTWindow=Hanning  blockSize=512  stepSize=256') #257

    #[22:25]
    fp.addFeature('ps: PerceptualSharpness FFTLength=0  FFTWindow=Hanning  blockSize=512  stepSize=256')#1
    fp.addFeature('zcr:ZCR blockSize=512  stepSize=256')#1
    engine = Engine()
    engine.load(fp.getDataFlow())
    afp = AudioFileProcessor()
    afp.processFile(engine,file)
    feats = engine.readAllOutputs()
    a1 = np.mean(feats['mfcc'],axis=0)
    a2 = np.mean(feats['sr'],axis=0)
    a3 = np.mean(feats['sf'],axis=0)
    a4 = np.mean(feats['sf1'],axis=0)
    a5 = np.mean(feats['sc'],axis=0)
    a6 = np.mean(feats['energy'],axis=0)
    a7 = np.mean(feats['loudness'])
    a8 = np.mean(feats['ps'],axis=0)
    a9 = np.mean(feats['zcr'],axis=0)
    #print ceps.shape
    #num_ceps = len(ceps)
    print a1.shape
    print a2.shape
    print a3.shape
    print a4.shape
    print a5.shape
    print a6.shape
    print a7.shape
    print a8.shape
    print a9.shape
    y, sr = librosa.load(file)
    
    hop_length = 256   

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    a10 = tempo
    feature4_set = np.hstack((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10))
    return feature4_set

def calc_statistical_features(mat):

    result = np.zeros((mat.shape[0],7))
    
    result[:,0] = np.mean(mat, axis=1)
    result[:,1] = np.var(mat, axis=1)
    result[:,2] = scipy.stats.skew(mat, axis=1)
    result[:,3] = scipy.stats.kurtosis(mat, axis=1)
    result[:,4] = np.median(mat, axis=1)
    result[:,5] = np.min(mat, axis=1)
    result[:,6] = np.max(mat, axis=1)
    
    result = np.nan_to_num(result)
    
    return result
def getData(genre_list):
    X = []
    y = []
    i = 0
    j = 0
    x = []
    filepath = "/mnt/hgfs/vmfiles/genres/"
    for genrefolder in os.listdir(filepath):
    	if genrefolder in genre_list:
    	    i = genre_list.index(genrefolder)
            genrefolder = filepath + genrefolder
            #os.makedirs(genrefolder)
        

            files = genrefolder
            for file in os.listdir(files):
                file = os.path.join(files,file)
                x = get_features4(file)
                X.append(x)
                y.append(i)
            
    return np.array(X), np.array(y)

def getSplitData(featureset = 0):
    genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]
    #X, y = getData(genre_list)
    #test feature 4
    
    mfcctestX = "/home/tyc/Downloads/machine learning/mlproject/feature4X"
    mfcctesty = "/home/tyc/Downloads/machine learning/mlproject/feature4y"
    if os.path.isfile(mfcctestX + ".npy"):

        X = np.load(mfcctestX + ".npy")
        y = np.load(mfcctesty + ".npy")
    else:
        X, y = getData(genre_list)
        np.save(mfcctestX, X)
        np.save(mfcctesty, y)
    print "X.shape", X.shape
    #16,17 unrelated feature; [0:15]+[22:] = 0.656666
    Xtemp = X[:,:15]
    Xtemp1 = X[:,22:25]

    X = np.hstack((Xtemp,Xtemp1))
    #test feature 4
    '''
    
    mfcctestX = "/home/tyc/Downloads/machine learning/mlproject/mfcctestX"
    mfcctesty = "/home/tyc/Downloads/machine learning/mlproject/mfcctesty"
    if os.path.isfile(mfcctestX + ".npy"):

    	X = np.load(mfcctestX + ".npy")
    	y = np.load(mfcctesty + ".npy")
    else:
    	X, y = getData(genre_list)
    	np.save(mfcctestX, X)
        np.save(mfcctesty, y)
    
    
    #test feature 2
    mfcctestX = "/home/tyc/Downloads/machine learning/mlproject/feature2X"
    mfcctesty = "/home/tyc/Downloads/machine learning/mlproject/feature2y"
    if os.path.isfile(mfcctestX + ".npy"):

        X = np.load(mfcctestX + ".npy")
        y = np.load(mfcctesty + ".npy")
    else:
        X, y = getData(genre_list)
        np.save(mfcctestX, X)
        np.save(mfcctesty, y)
    #test feature 2
    '''

    '''
    mfcctestX = "/home/tyc/Downloads/machine learning/mlproject/mfcctestX"
    mfcctesty = "/home/tyc/Downloads/machine learning/mlproject/mfcctesty"
    if os.path.isfile(mfcctestX + ".npy"):

        X1 = np.load(mfcctestX + ".npy")
        y1 = np.load(mfcctesty + ".npy")
    else:
        X1, y1 = getData(genre_list)
        np.save(mfcctestX, X1)
        np.save(mfcctesty, y1)
    X = np.concatenate((X, X1), axis=1)
    '''
    


    print X.shape[0]
    print X.shape[1]
    print "X", X.shape
    print len(X[0])
    print y.shape
    train = []
    trainlabel = []
    test = []
    testlabel = []
    xnum = X.shape[0]
    temp = len(genre_list)
    j = 0
    for i in range(0,xnum):
        if (temp != y[i]):
            temp = y[i]
            j = 0
        if j < 50:
            train.append(X[i])
            trainlabel.append(y[i])
        else:
            test.append(X[i])
            testlabel.append(y[i])
        j = j + 1
    return np.array(train),np.array(trainlabel),np.array(test),np.array(testlabel)

def getSplitData1(p=40,featureset = 0):
    genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]

    #X, y = getData(genre_list)
    #test feature 4
    '''
    mfcctestX = "/home/tyc/Downloads/machine learning/mlproject/feature5X"
    mfcctesty = "/home/tyc/Downloads/machine learning/mlproject/feature5y"
    if os.path.isfile(mfcctestX + ".npy"):

        X = np.load(mfcctestX + ".npy")
        y = np.load(mfcctesty + ".npy")
    else:
        X, y = getData(genre_list)
        np.save(mfcctestX, X)
        np.save(mfcctesty, y)
    print "X.shape", X.shape
    #16,17 unrelated feature; [0:15]+[22:] = 0.67
    Xtemp = X[:,:434] # 0.66
    #Xtemp1 = X[:,22:25]
    X = Xtemp
    #X = np.hstack((Xtemp,Xtemp1))
    #test feature 4
    '''
    '''
    mfcctestX = "/home/tyc/Downloads/machine learning/mlproject/mfcctestX"
    mfcctesty = "/home/tyc/Downloads/machine learning/mlproject/mfcctesty"
    if os.path.isfile(mfcctestX + ".npy"):

        X = np.load(mfcctestX + ".npy")
        y = np.load(mfcctesty + ".npy")
    else:
        X, y = getData(genre_list)
        np.save(mfcctestX, X)
        np.save(mfcctesty, y)
    
    
    #test feature 2
    mfcctestX = "/home/tyc/Downloads/machine learning/mlproject/feature2X"
    mfcctesty = "/home/tyc/Downloads/machine learning/mlproject/feature2y"
    if os.path.isfile(mfcctestX + ".npy"):

        X = np.load(mfcctestX + ".npy")
        y = np.load(mfcctesty + ".npy")
    else:
        X, y = getData(genre_list)
        np.save(mfcctestX, X)
        np.save(mfcctesty, y)
    #test feature 2
    '''

    '''
    mfcctestX = "/home/tyc/Downloads/machine learning/mlproject/mfcctestX"
    mfcctesty = "/home/tyc/Downloads/machine learning/mlproject/mfcctesty"
    if os.path.isfile(mfcctestX + ".npy"):

        X1 = np.load(mfcctestX + ".npy")
        y1 = np.load(mfcctesty + ".npy")
    else:
        X1, y1 = getData(genre_list)
        np.save(mfcctestX, X1)
        np.save(mfcctesty, y1)
    X = np.concatenate((X, X1), axis=1)
    '''
    #test feature 6
    mfcctestX = "/home/tyc/Downloads/machine learning/mlproject/feature6X"
    mfcctesty = "/home/tyc/Downloads/machine learning/mlproject/feature6y"
    if os.path.isfile(mfcctestX + ".npy"):

        X = np.load(mfcctestX + ".npy")
        y = np.load(mfcctesty + ".npy")
    else:
        X, y = getData(genre_list)
        np.save(mfcctestX, X)
        np.save(mfcctesty, y)
    print "X.shape", X.shape
    #16,17 unrelated feature; [0:15]+[22:] = 0.67
    #Xtemp = X[:,:434] # 0.66
    #Xtemp1 = X[:,22:25]
    #X = Xtemp
    #X = np.hstack((Xtemp,Xtemp1))
    #test feature 6
    #print "X........",X
    for i in range(0,len(X)):
        l = len(X[i])
        if(l < 2538):
            X[i] = np.hstack((X[i],np.zeros(2538 - l)))
            for j in range(l,2538):
                X[i][j] = X[i][l-1]
            print len(X[i])

    
    print X.shape[0]
    #print X.shape[1]
    print "X", X.shape
    print len(X[0])
    print X[0]
    print len(X[1])
    print y.shape
    train = []
    trainlabel = []
    test = []
    testlabel = []
    xnum = X.shape[0]
    temp = len(genre_list)
    j = 0
    for i in range(0,xnum):
        #if (y[i]==0):
            #continue
        if (temp != y[i]):
            temp = y[i]
            j = 0
        if j < 50:
            train.append(list(X[i]))
            trainlabel.append(y[i])
        else:
            test.append(list(X[i]))
            testlabel.append(y[i])
        j = j + 1
    '''
    a = np.array(train).shape[0]
    b = np.array(train).shape[1]
    train1 = []
    test1 = []
    f1 = []
    f2 = []
    temp = 0
    x = np.zeros(4)
    for i in range(0,a):
        f1 = []
        f2 = []
        temp = 0
        for j in range(1,b):
            if(train[i][j]!=train[i][j-1]):
                f2.append(np.abs(train[i][j] - train[i][j-1]))
                f1.append(temp)
                temp = 0
            else:
                temp = temp + 1
        x[0] = np.mean(f1)
        x[1] = np.var(f1)
        x[2] = np.mean(f2)
        x[3] = np.var(f2)
        train1.append(x)
    f1 = []
    f2 = []
    temp = 0
    x = np.zeros(4)
    for i in range(0,a):
        f1 = []
        f2 = []
        temp = 0
        for j in range(1,b):
            if(test[i][j]!=test[i][j-1]):
                f2.append(np.abs(test[i][j] - test[i][j-1]))
                f1.append(temp)
                temp = 0
            else:
                temp = temp + 1
        print f1
        x[0] = np.mean(f1)
        x[1] = np.var(f1)
        x[2] = np.mean(f2)
        x[3] = np.var(f2)
        test1.append(x)
    '''
    a = np.array(train).shape[0]
    b = np.array(train).shape[1]
    train1 = []
    test1 = []
    f1 = []
    #x = np.zeros(4)
    for i in range(0,a):
        f1 = []
        f1.append(train[i][0])
        for j in range(1,b):
            if(train[i][j]!=train[i][j-1]):                
                f1.append(train[i][j]) 
        train1.append(f1)
        #print "%d   %d"%(len(f1),trainlabel[i])
    a = np.array(test).shape[0]
    b = np.array(test).shape[1]

    f1 = []
    #x = np.zeros(4)
    for i in range(0,a):
        f1 = []
        count = 0
        f1.append(test[i][0])
        for j in range(1,b):
            if(test[i][j]!=test[i][j-1]):                
                if(count>p):
                    f1.append(test[i][j])
                count = 0
            else:
                count = count + 1
        test1.append(f1)
    return np.array(train1),np.array(trainlabel),np.array(test1),np.array(testlabel)
def getScaledData(traindata,testdata):
    print "Scale..."
    min_max_scaler = MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(traindata)
    X_test_minmax = min_max_scaler.transform(testdata)
    return X_train_minmax,X_test_minmax

def plot_confusion_matrix(cm, genre_list, name="", title=""):
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap=plt.cm.Blues)
    ax = pylab.axes()
    ax.set_xticks(range(len(genre_list)))
    ax.set_xticklabels(genre_list)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(len(genre_list)))
    ax.set_yticklabels(genre_list)
    #pylab.title(title)
    pylab.colorbar()
    pylab.grid(False)
    pylab.xlabel('Predicted class')
    pylab.ylabel('True class')
    pylab.grid(False)
    pylab.show()

def testsvm():
    genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]
    Xtrain,ytrain,Xtest,ytest = getSplitData()
    Xtrain, Xtest = getScaledData(Xtrain, Xtest)
    traindata = Xtrain
    trainlabel = ytrain
    testdata = Xtest
    testlabel = ytest
    '''
    for k in range(2,4,1):
            degree = k
            for i in range(5,12):
                C=2**(i-5)
                for j in range(6,12):
                    gamma=2**(j-8)            
                    print ("kernel = 'poly' C = %f   gamma = %f  degree = %f") %(C,gamma,degree)
                    clf = SVC(C=C,gamma=gamma,kernel='poly',degree=degree)
    
                    clf.fit(traindata, trainlabel)
                    pred = clf.predict(testdata)
        
                    print "svm classification accuracy: ", accuracy_score(testlabel,pred)
                    cm = confusion_matrix(testlabel, pred)
                    print cm
                    #plot_confusion_matrix(cm,genre_list)
    
    #print ("kernel = 'poly' C = %f   gamma = %f  degree = %f") %(C,gamma,degree)
    '''
    '''
    clf = SVC(C = 2, gamma = 1, kernel = 'poly', degree = 3)
    clf.fit(traindata, trainlabel)
    pred = clf.predict(testdata)
    print "svm classification accuracy: ", accuracy_score(testlabel,pred)
    cm = confusion_matrix(testlabel, pred)
    print cm
    plot_confusion_matrix(cm,genre_list)
    '''


    clf = SVC(C = 2, gamma = 2, kernel = 'poly', degree = 2)
    #clf = SVC(C = 2, gamma = 1, kernel = 'poly', degree = 3)
    clf.fit(traindata, trainlabel)
    pred = clf.predict(testdata)
    cm = confusion_matrix(testlabel, pred)
    print cm
    print "svm classification accuracy: ", accuracy_score(testlabel,pred)
    Xtrain1,ytrain1,Xtest1,ytest1 = getSplitData1(p = 175) #175 0.68
    print "abc"
    ntest = Xtest1.shape[0]
    ntrain = Xtrain1.shape[0]
    yPredict = []
    for i in range(0,ntest):
        if (pred[i]!=1 and pred[i]!=2):
            continue

        dtemp = 0
        ptemp = 0
        for j in range(0,ntrain):
            #if(trainlabel[j] == 0 or trainlabel[j] == 3 or trainlabel[j] == 4 or trainlabel[j] == 5):
                #continue
            if(trainlabel[j] == 0):
                continue
            d = lcs(Xtest1[i],Xtrain1[j])
            if(d>dtemp):
                dtemp = d
                ptemp = ytrain1[j]
        #print Xtest[i]
        #print dtemp
        #print ptemp
        if(ptemp==1 and pred[i]==1):
            pred[i] = ptemp
        elif (ptemp==1 and pred[i]==2):
            pred[i] = ptemp
        #else:
            #pred[i] = ptemp
        #print i
    print "svm classification accuracy: ", accuracy_score(testlabel,pred)
    cm = confusion_matrix(testlabel, pred)
    print cm
    plot_confusion_matrix(cm,genre_list)
 
def testsvm1():
    genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]
    Xtrain,ytrain,Xtest,ytest = getSplitData()
    Xtrain, Xtest = getScaledData(Xtrain, Xtest)
    traindata = Xtrain
    trainlabel = ytrain
    testdata = Xtest
    testlabel = ytest
    
    for k in range(2,4,1):
            degree = k
            for i in range(5,12):
                C=2**(i-5)
                for j in range(6,12):
                    gamma=2**(j-8)            
                    print ("kernel = 'poly' C = %f   gamma = %f  degree = %f") %(C,gamma,degree)
                    clf = SVC(C=C,gamma=gamma,kernel='poly',degree=degree)
    
                    clf.fit(traindata, trainlabel)
                    pred = clf.predict(testdata)
        
                    print "svm classification accuracy: ", accuracy_score(testlabel,pred)
                    cm = confusion_matrix(testlabel, pred)
                    print cm
                    #plot_confusion_matrix(cm,genre_list)
    
    #print ("kernel = 'poly' C = %f   gamma = %f  degree = %f") %(C,gamma,degree)
def gridSearch(Xtrain,ytrain,Xtest,ytest,option = 0):
    Xtrain, Xtest = getScaledData(Xtrain, Xtest)
    traindata = Xtrain
    trainlabel = ytrain
    testdata = Xtest
    testlabel = ytest
    if(option==0):
        for i in range(5,12):
            C=2**(i-5)
            for j in range(5,12):
                gamma=2**(j-8)            
                print ("kernel='rbf' C = %f   gamma = %f") %(C,gamma)
                clf = SVC(C=C,gamma=gamma,kernel='rbf')
    
                clf.fit(traindata, trainlabel)
                pred = clf.predict(testdata)
        
                #print(classification_report(testlabel, pred))
                print "svm classification accuracy: ", accuracy_score(testlabel,pred)
    elif (option==1):
        for i in range(5,12):
            C=2**(i-5)                   
            print ("kernel='linear' C = %f ") %(C)
            clf = SVC(C=C,kernel='linear')
    
            clf.fit(traindata, trainlabel)
            pred = clf.predict(testdata)
        
            print "svm classification accuracy: ", accuracy_score(testlabel,pred)
    elif (option==2):
        for k in range(2,4,1):
            degree = k
            for i in range(5,12):
                C=2**(i-5)
                for j in range(5,12):
                    gamma=2**(j-8)            
                    print ("kernel='poly' C = %f   gamma = %f  degree = %f") %(C,gamma,degree)
                    clf = SVC(C=C,gamma=gamma,kernel='poly',degree=degree)
    
                    clf.fit(traindata, trainlabel)
                    pred = clf.predict(testdata)
        
                    print "svm classification accuracy: ", accuracy_score(testlabel,pred)
def testfeature2():
    Xtrain,ytrain,Xtest,ytest = getSplitData()
    #Xtrain, Xtest = getScaledData(Xtrain, Xtest):
    gridSearch(Xtrain,ytrain,Xtest,ytest,0)
    gridSearch(Xtrain,ytrain,Xtest,ytest,1)
    gridSearch(Xtrain,ytrain,Xtest,ytest,2)

def testfeature3():
    Xtrain,ytrain,Xtest,ytest = getSplitData(3)
    #Xtrain, Xtest = getScaledData(Xtrain, Xtest):
    gridSearch(Xtrain,ytrain,Xtest,ytest,0)
    gridSearch(Xtrain,ytrain,Xtest,ytest,1)
    gridSearch(Xtrain,ytrain,Xtest,ytest,2)
def testfeature4():
    Xtrain,ytrain,Xtest,ytest = getSplitData(4)
    #Xtrain, Xtest = getScaledData(Xtrain, Xtest):
    gridSearch(Xtrain,ytrain,Xtest,ytest,0)
    gridSearch(Xtrain,ytrain,Xtest,ytest,1)
    gridSearch(Xtrain,ytrain,Xtest,ytest,2)
def dtree():

    Xtrain,ytrain,Xtest,ytest = getSplitData()
    Xtrain, Xtest = getScaledData(Xtrain, Xtest)
    ntest = Xtest.shape[0]
    clf = DecisionTreeClassifier(criterion='gini')
    clf.fit(Xtrain,ytrain)
    yPredict = clf.predict(Xtest)
    
    print "parameter: criterion='gini' "
    print "decision_tree classification accuracy: ", accuracy_score(ytest,yPredict)

    #Your code here
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(Xtrain,ytrain)
    yPredict = clf.predict(Xtest)
    print "parameter: criterion='entropy' "
    print "decision_tree classification accuracy: ", accuracy_score(ytest,yPredict)

def knn():
    
    Xtrain,ytrain,Xtest,ytest = getSplitData()
    Xtrain, Xtest = getScaledData(Xtrain, Xtest)
    ntest = Xtest.shape[0]
    ntrain = Xtrain.shape[0]
    for n in range(1,7):
        neigh = KNeighborsClassifier(n_neighbors=n)
        neigh.fit(Xtrain, ytrain) 
    
       

        yPredict1 = neigh.predict(Xtrain)
    
        print "parameter: n_neighbors = ",n
        print "knn train classification accuracy: ", accuracy_score(ytrain,yPredict1)


        yPredict = neigh.predict(Xtest)
        print "knn test classification accuracy: ", accuracy_score(ytest,yPredict)

def neural_net():
    Xtrain,ytrain,Xtest,ytest = getSplitData()
    Xtrain, Xtest = getScaledData(Xtrain, Xtest)
    ntest = Xtest.shape[0]
    #Your code here
    clf = Perceptron()
    clf.fit(Xtrain, ytrain) 
    
    yPredict = clf.predict(Xtest)
    
    #print "parameter: n_neighbors = ",n
    print "neural_net classification accuracy: ", accuracy_score(ytest,yPredict)

def pca_knn():
    Xtrain,ytrain,Xtest,ytest = getSplitData()
    Xtrain, Xtest = getScaledData(Xtrain, Xtest)
    ntest = Xtest.shape[0]
    #Your code here
    for n in range(5,8):
        pca = RandomizedPCA(n_components=n)
        pca_Xtrain = pca.fit_transform(Xtrain, ytrain)
        pca_Xtest = pca.fit_transform(Xtest) 
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(pca_Xtrain, ytrain)
        
        yPredict = neigh.predict(pca_Xtest)
     
        print "parameter: n_components = ",n
        print "parameter: n_neighbors = 5"
        print "pca_knn classification accuracy: ", accuracy_score(ytest,yPredict)

def kmeans(X, k, maxIter):
    #X = X[:,:-1]
    print X.shape
    d0 = X.shape[0]
    d1 = X.shape[1]
    a = []
    
    label = np.zeros(d0,dtype=np.int)
    reversel = np.empty((k, 0)).tolist()
    for j1 in range(0,maxIter):     
        if j1 == 0:
            for i in range(0,k):
                a.append(np.random.uniform(0,1,d1))
        else:
            for i in range(0,k):
                if (reversel[i]==[]):
                    a[i] = np.random.uniform(0,1,d1)
                else:
                    a[i] = np.mean(X[reversel[i]],axis=0)
        reversel = np.empty((k, 0)).tolist()
        label = np.zeros(d0,dtype=np.int)
        for j2 in range(0,d0):
            mindistance = np.sum((X[j2] - a[0])**2)
            minindex = 0
            for j3 in range(1,k):
                distance = np.sum((X[j2]-a[j3])**2)
                if(distance < mindistance):
                    mindistance = distance
                    minindex = j3
            label[j2] = minindex
            reversel[minindex].append(j2)
    kscatter(X,reversel)
    return label 

def kmeans1(X, k, maxIter):
    #X = X[:,:-1]
    print X.shape
    d0 = X.shape[0]
    d1 = X.shape[1]
    a = [] # a is a list of center with length k
    
    label = np.zeros(d0,dtype=np.int)
    reversel = np.empty((k, 0)).tolist() #reversel is a map from clusterindex to a list of sampleindex
    for j1 in range(0,maxIter):     
        if j1 == 0:
            for i in range(0,k):
                a.append(np.random.uniform(0,1,d1))
        else:
            for i in range(0,k):
                if (reversel[i]==[]):
                    a[i] = np.random.uniform(0,1,d1)
                else:
                    a[i] = np.mean(X[reversel[i]],axis=0)
        reversel = np.empty((k, 0)).tolist()
        label = np.zeros(d0,dtype=np.int)
        for j2 in range(0,d0):
            mindistance = np.sum((X[j2] - a[0])**2)
            minindex = 0
            for j3 in range(1,k):
                distance = np.sum((X[j2]-a[j3])**2)
                if(distance < mindistance):
                    mindistance = distance
                    minindex = j3
            label[j2] = minindex
            reversel[minindex].append(j2)
    #kscatter(X,reversel)
    return a, reversel   
def kscatter(X,reversel):
    colors = cm.rainbow(np.linspace(0, 1, len(reversel)))
    for r, c in zip(reversel, colors):
        x = X[r]
        plt.scatter(x[:,0], x[:,1], color=c)
    plt.show()    
   



def purity(labels, trueLabels):

    k = labels.max(axis=0) + 1
    reversel = np.empty((k, 0)).tolist()

    for i in range(0, labels.shape[0]):
        reversel[labels[i]].append(i)
    #print reversel[1]
    p = {}
    for i in range(0, k):
        q = {}
        t1 = trueLabels[reversel[i]]
        for j in t1:
            if(j in q):
                q[j] = q[j] + 1
            else:
                q[j] = 0
        maxn = 0
        maxi = 0
        for j1 in q:
            if (q[j1] >= maxn):
                maxi = j1
                maxn = q[j1]
        if(q=={}):
            print q
            p[i] = 1
        else:
            print q
            p[i] = q[maxi]*1.0 / len(t1)
        print ('cluster %d has purity %f' % (i,p[i]))
    sum = 0
    for i in p:
        sum = sum + p[i]
    return sum

def kneeFinding(X):
    x = [1,2,3,4,5,6,7,8,9,10,11,12]
    y = [0,0,0,0,0,0,0,0,0,0,0,0]
    X1 = X
    for k in range(1,13):
        obj = 0
        a, reversel = kmeans1(X,k,50)
        for i in range(0, len(a)):
            for j in reversel[i]:
                obj = obj + np.sum((X1[j] - a[i])**2)
        y[k-1] = obj
    plt.plot(x, y)
    plt.show()
def testkmeans():
    Xtrain,ytrain,Xtest,ytest = getSplitData()
    Xtrain, Xtest = getScaledData(Xtrain, Xtest)
    ntest = Xtest.shape[0]   
    labels = kmeans(Xtrain,9,50)
    purityMetric = purity(labels, ytrain)
    #print purityMetric
    #kneeFinding(Xtrain)
def lcs(a,b):
    m = len(a)
    n = len(b)
    c = np.zeros((m+1,n+1))
    for i in range(1,m+1):
        for j in range(1,n+1):
            if(a[i-1] == b[j-1]):
                c[i][j] = c[i-1][j-1]+1
            else:
                c[i][j] = max(c[i][j-1], c[i-1][j])
    return c[m][n]

def knn1(p = 80):
    #genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]
    genre_list = ["classical","jazz", "country", "pop", "rock", "metal"]
    Xtrain,ytrain,Xtest,ytest = getSplitData1(p = p)
    #Xtrain, Xtest = getScaledData(Xtrain, Xtest)
    ntest = Xtest.shape[0]
    ntrain = Xtrain.shape[0]
    print "knn1 ntest", ntest
    print "knn1 ntrain", ntrain
    yPredict = []
    for i in range(0,ntrain):
        dtemp = 0
        ptemp = 0
        for j in range(0,ntrain):
            d = lcs(Xtrain[i],Xtrain[j])
            if(d>dtemp):
                dtemp = d
                ptemp = ytrain[j]
        yPredict.append(ptemp)
    print "1-nn lcs train classification accuracy: ", accuracy_score(ytrain,yPredict)


    ntest = Xtest.shape[0]
    ntrain = Xtrain.shape[0]
    yPredict = []
    for i in range(0,ntest):
        dtemp = 0
        ptemp = 0
        for j in range(0,ntrain):
            d = lcs(Xtest[i],Xtrain[j])
            if(d>dtemp):
                dtemp = d
                ptemp = ytrain[j]
        #print Xtest[i]
        #print dtemp
        yPredict.append(ptemp)
    print "1-nn lcs test classification accuracy: ", accuracy_score(ytest,yPredict)
    cm = confusion_matrix(ytest, yPredict)
    print cm
    #plot_confusion_matrix(cm,genre_list)

if __name__ == '__main__':
    #testmfcc()
    #dtree()
    #knn() 
    #neural_net()
    #pca_knn()
    #test_feature()
    testsvm()
    #testfeature2()
    #testfeature3()
    #testkmeans()
    #testfeature4()