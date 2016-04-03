#!/usr/bin/python
#from yaafelib import *
import numpy as np
import sys,os,glob
import scipy.io.wavfile
import sklearn.svm
from matplotlib import pylab

import numpy as np

import wave
import struct
import sys
from scikits.audiolab import *
import random
from datetime import datetime
import operator
import scipy
from scipy import signal
import math
from pylab import*
import cmath
import operator
from tempfile import TemporaryFile

def read_audio(filename):
	spf = wave.open(filename,'r')
	signal = spf.readframes(-1)
	signal = np.fromstring(signal, 'Int16')
	p = spf.getnchannels()
	f = spf.getframerate()
	sound_info = np.zeros(len(signal),dtype=float)
	signal = signal.astype(np.float)
	sound_info = signal/max(signal)

	#sound_info = sound_info[1:len(sound_info):2]
	if p==2:
		sound_info = scipy.signal.decimate(sound_info,2)

	return p ,f , sound_info


def spectrogram(sound_info,f,nfft,hop):
	Pxx, freqs, bins, im = specgram(sound_info, Fs = f, NFFT = nfft, noverlap=nfft-hop, scale_by_freq=True,sides='default')
	return Pxx, freqs, bins, im


def hz_to_oct(freq):
	#A6 = 27.5*(2^6)
	#A0 = 27.5
	#A4 = 440
	fmin = 27.5
	b = 24
	return np.log2(freq/fmin)*b


def oct_to_hz(oct):
	fmin = 27.5
	b = 24.0
	return fmin*(2**(oct/b))


def generate_filterbank(NFFT,fs,b,z):
	#b is bins per octave
	#z is number of octaves
	#b = 24
	#z = 6
	#fs(downsampled) = 44100/4 = 11025

	octmax = b*z

	octpts = np.arange(-1,octmax+1)
	#print 'octpts',octpts
	#print len(octpts)

	ctrfrq = oct_to_hz(octpts)
	#print "ctrfrq",ctrfrq
	#print len(ctrfrq)

	ctrrep = np.floor((NFFT+2)*ctrfrq/((fs/2)))
	#print "ctrrep",ctrrep
	#print len(ctrrep)

	bank = np.zeros([len(octpts)-2,NFFT/2+1],dtype=float)

	for j in xrange(0,len(octpts)-2):
		y = np.hamming(ctrrep[j+2]-ctrrep[j])
		area = trapz(y, dx=5)
		if area==0:
			area=1
		y2 = (y/area)
		bank[j,ctrrep[j]:ctrrep[j+2]] = y2
		#plot(bank[j,:])

	#show()
	return bank


def generate_template():
	b = np.zeros(288).reshape(12,24)

	a = [0,0,0,1,0,0,0,1,0,0,1,0]
	b[:,0] = a

	for i in range(1,12):
		b[:,i] = np.roll(b[:,i-1],1)

	minor = [0,0,0,1,0,0,1,0,0,0,1,0]
	b[:,12] = minor

	for i in range(13,24):
		b[:,i] = np.roll(b[:,i-1],1)
	
	return b


def ground_truth(labfile):
	dict = {'N':None,'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11,
	'c':12,'c#':13,'d':14,'d#':15,'e':16,'f':17,'f#':18,'g':19,'g#':20,'a':21,'a#':22,'b':23}

	ground = np.zeros(1250)
	list = []
	with open(labfile,'r') as g:
	            f=g.readlines()
	            for line in f:
	        		a=(line.split(' '))
	        		list.append(a)
	        		ground[np.floor(float(a[0])*86.13):np.floor(float(a[1])*86.13)] = dict[a[2].rstrip("\n")]
	        		if float(a[1])>15:
	        			break
	return ground

def generate_template():
	b = np.zeros(288).reshape(12,24)

	a = [0,0,0,1,0,0,0,1,0,0,1,0]
	b[:,0] = a

	for i in range(1,12):
		b[:,i] = np.roll(b[:,i-1],1)

	minor = [0,0,0,1,0,0,1,0,0,0,1,0]
	b[:,12] = minor

	for i in range(13,24):
		b[:,i] = np.roll(b[:,i-1],1)
	
	return b

def viterbi(chrm):

	temp = generate_template()
	#print "template shape is",shape(temp)
	trans = np.dot(temp.T,temp)

	trans = np.ones(shape(trans))
	trans = trans + np.eye(24)*0.1
	#print trans
	print "transition matrix shape is",shape(trans)


	V = np.zeros(shape(chrm))
	path = np.zeros(shape(chrm))

	b,col = shape(chrm)
	print "shape of V is",shape(V)

	#same prior probabilities
	initial = 1.00

	#initilization
	V[:,0] = np.log(chrm[:,0]) + np.log(initial)
	#V[:,0] = np.zeros(b)

	#chrm = chromgram

	for t in range(1,col):
		for j in range(b):
			sam = [(V[i,t-1]+np.log(trans[i,j])+np.log(chrm[j,t]),i) for i in range(b)]
			#find index and value that maximize sum
			(prob,state) = max(sam)
			V[j,t] = prob
			path[j,t-1] = state

	#print path

	return V,path

def process0(final_path):
    n =  final_path.shape[0]
    print n
    print final_path
    temp = final_path[0]
    tempnum = 1
    previous = 0
    boundary = 40
    for i in range(0,n) :

	    if(temp==final_path[i]):
		    tempnum = tempnum + 1
	    else:
		if(tempnum < boundary):
		    for j in range(0,tempnum):
		        final_path[i-1-j] = previous
                    temp = final_path[i]
		    tempnum = 1
                else:
                    previous = temp
            	    temp = final_path[i]
		    tempnum = 1
    return final_path

def process1(final_path):
    n =  final_path.shape[0]
    print n
    print final_path
    temp = final_path[0]
    tempnum = 1
    boundary = 40
    for i in range(0,n) :

	    if(temp==final_path[i]):
		    tempnum = tempnum + 1
	    else:
		    if(tempnum < boundary):
			    for j in range(0,tempnum):
				    final_path[i-1-j] = 0
		    temp = final_path[i]
		    tempnum = 1
    return final_path

def genChroma(file):
	file1 = file.replace("genres","genreschord")
	file1 = file1.replace(".wav",".chroma")
	file2 = file.replace("genres","genreschord")
	file2 = file2.replace(".wav",".tempo")
	print file
	print file1
	"""
	Read input audio wav file
	"""
	p, f, sound_info_old = read_audio(file)
	print "frequency is",f
	print "channels are",p
	print len(sound_info_old)
	#30 seconds is 1323000
	sound_info_deci = scipy.signal.decimate(sound_info_old,2)
	print len(sound_info_deci)
	#165375 is 15 sec 330750 is 30 sec for downsampled signal
	sound_info = sound_info_deci[0:330750]
	print "length of audio is ",len(sound_info)
	f = f/2

	#plot(sound_info)
	#show()


	"""
	Compute spectrogram
	"""

	Pxx, freqs, bins, im = spectrogram(sound_info, f, 6000, 128)
	print "shape of Spectrogram is",shape(Pxx)
	#plot(Pxx,freqs)
	#show()


	"""
	Generate CQ filterabank
	"""

	bank = generate_filterbank(NFFT=6000,fs=f,b=24,z=6)
	print "shape of bank",shape(bank)
	#im = imshow(bank,aspect='auto',interpolation='nearest',origin='lower')
	#show()


	"""
	Generate CQ spectrogram as the dot product of spectrogram and CQ filterabank
	"""

	sal = np.dot(bank,Pxx)
	b,col = shape(sal)

	#Replace 0 by 1 in matrix before taking log
	for i in range(0,len(sal)):
		for j in range(0,len(sal[0])):
			if sal[i][j] == 0:
				sal[i][j]+=1

	sal = 10*np.log10(sal)
	#Normalize
	salm = np.zeros(shape(sal))
	salm = (sal - sal.min())/(sal.max()-sal.min())
	#salm = sal

	"""
	for i in range(col):
		salm[:,i] = sal[:,i]/sum(sal[:,i])
	"""

	print "shape of CQ Spectrogram is",shape(salm)
	#subplot(3,1,2)
	#title('CQ spectrogram (dB)')
	#xlim(0,(len(sound_info)/11025.00))
	#im = imshow(salm,aspect='auto',interpolation='nearest',origin='lower')
	#xticks(np.arange(0,col,172),[0,2,4,6,8,10,12,14])

	#show()


	"""
	Generate Chroma
	"""
	row,col = shape(salm)
	#print row
	#print col
	b = 24

	chrm = np.zeros(b*col).reshape(b,col)

	for i in range(col):
		c = salm[:,i]
		for j in range(b):
			chrm[j,i] = sum(c[j:row:b])

	#print chrm

	chrm_new = np.zeros(b*col).reshape(b,col)

	#median
	"""
	for i in range(b):
		for j in range(col):
			chrm_new[i,j] = np.median(chrm[i,j-3:j+3])
	"""

	chrm_new = chrm

	chrm_new_norm = np.zeros(b*col).reshape(b,col)

	#normalise
	for i in range(col):
		chrm_new_norm[:,i] = chrm_new[:,i]/sum(chrm_new[:,i])
		#chrm_new_norm[i,j] = np.median(chrm_new[i,j-10:j+10])
	#print chrm_new_norm


	print "shape of Chroma is",shape(chrm_new_norm)
	#subplot(3,1,3)
	#title('Chroma (dB)')

	#save chroma in a .npy file
	chromaa = TemporaryFile()
	np.save(file1,chrm_new_norm)

	#im2 = imshow(chrm_new_norm,aspect='auto',interpolation='nearest',origin='lower')
	#xlim(0,(len(sound_info)/11025.00))
	#yticks([1,3,5,7,9,11,13,15,17,19,21,23], ['A','A#','B','C','C#','D','D#','E','F','F#','G','G#'])
	#xticks(np.arange(0,col,172),[0,2,4,6,8,10,12,14])

	#show()

	#only for plotting
	#subplot(3,1,1)
	#title('Spectrogram (dB)')
	#Pxx, freqs, bins, im = spectrogram(sound_info, f, 6000, 128)
	#plot(Pxx,freqs)
	#xlim(0,(np.ceil(len(sound_info)/11025)))
	#show()


	"""
	Generate Template
	"""

	temp = generate_template()
	#xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], 
		#['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','c','c#','d','d#','e','f','f#','g','g#','a','a#','b'])
	#yticks([0,1,2,3,4,5,6,7,8,9,10,11,12], ['A','A#','B','C','C#','D','D#','E','F','F#','G','G#'])
	#title('Template')
	#im = imshow(temp,aspect='auto',interpolation='nearest',origin='lower')
	#show()

	print "template shape is",shape(temp)


	"""
	Fitness matrix computation by taking
	dot product of Template and Chroma
	"""

	#Since b=24, we need to take mean of 2 rows
	lol = zip(*chrm_new_norm[::-1])

	print shape(lol)
	lol = np.asarray(lol)
	grand = lol.reshape(-1,2).mean(axis=1).reshape(lol.shape[0],-1)
	#print shape(grand)

	grand2 = zip(*grand)[::-1]
	#print shape(grand2)

	match = np.dot(temp.T,grand2)

	match_norm = np.zeros(shape(match))

	#normalize

	for i in range(col):
		match_norm[:,i] = match[:,i]/sum(match[:,i])

	print "matched array is",shape(match_norm)
	#title('Fitness Matrix')
	#im = imshow(match_norm,aspect='auto',interpolation='nearest',origin='lower')
	#yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], 
		#['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','c','c#','d','d#','e','f','f#','g','g#','a','a#','b'])
	#show()

	#save tempogram in a .npy file
	match22 = TemporaryFile()
	np.save(file2,match_norm)

	print file + ' chroma generation end'

def genChord(file):
	file1 = file.replace("genres","genreschord")
	file1 = file1.replace(".wav",".chroma")
	file1 = file1 + ".npy"
	file2 = file.replace("genres","genreschord")
	file2 = file2.replace(".wav",".tempo")
	file2 = file2 + ".npy"
	file3 = file.replace("genres","genreschord")
	file3 = file3.replace(".wav",".chord")
	"""
	Read chroma file and plot it
	"""

	chroma = np.load(file1)
	#subplot(3,1,1)
	#title('Chroma')

	#im = imshow(chroma,aspect='auto',interpolation='nearest',origin='lower',extent=[0,15,0,24])
	#yticks([1,3,5,7,9,11,13,15,17,19,21,23], ['A','A#','B','C','C#','D','D#','E','F','F#','G','G#'])


	"""
	Read the Distace matrix file and plot it
	"""
	chrm = np.load(file2)
	b,col = shape(chrm)
	print chrm.shape
	whereAreNaNs = np.isnan(chrm);
	chrm[whereAreNaNs] = 0;

	#subplot(3,1,2)
	#title('Templates')

	#im = imshow(chrm,aspect='auto',interpolation='nearest',origin='lower',extent=[0,15,0,24])
	#yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], 
		#['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','c','c#','d','d#','e','f','f#','g','g#','a','a#','b'])


	"""
	Viterbi Algorithm
	"""

	V,path = viterbi(chrm)



	Vnorm = np.zeros(shape(V))

	#normalize
	for i in range(col):
		Vnorm[:,i] = V[:,i]/(sum(V[:,i]))

	Vm = np.zeros(shape(V))


	#median filter
	for i in range(b):
		for j in range(col):
			Vm[i,j] = np.mean(V[i,max(0,j-100):min(col,j+100)])


	#picking chords with max value
	inde = np.argmax(Vm,axis=0)
	final_path = np.zeros(col)

	for i in range(col):
		final_path[i] = path[inde[i],i]


	#ground = ground_truth('Let It Be_new.lab')

	#print "ground truth length is",len(ground)


	#plot detected and ground thruth chords
	#subplot(3,1,3)
	#title('Chords')
	#yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], 
		#['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','c','c#','d','d#','e','f','f#','g','g#','a','a#','b'])
	#l = plot(ground,'ro',alpha=0.3)
	#setp(l, 'markersize', 15)

	final_path = process0(final_path)
	#final_path = process1(final_path)
	#plot(final_path,'ko',alpha=0.2)



	#yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], 
		#['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','c','c#','d','d#','e','f','f#','g','g#','a','a#','b'])
	#xlim(0,col)
	#xticks(np.arange(0,col,172),[0,2,4,6,8,10,12,14])
	#show()


	np.save(file3, final_path)
	print file + ' chord extraction end'

def chordextract():
	filepath = "/mnt/hgfs/vmfiles/genres/"
	for genrefolder in os.listdir(filepath):
		genrefolder = filepath + genrefolder
        newgenrefolder = genrefolder.replace("genres","genreschord")
        if(not os.path.isdir(newgenrefolder)):
            os.makedirs(newgenrefolder)
        files = genrefolder
        for file in os.listdir(files):
            file = files + "/" + file
            genChroma(file)
            #genChord(file)

if __name__ == '__main__':
	#genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]
	#chordextract()
	genChroma(sys.argv[1])
	genChord(sys.argv[1])