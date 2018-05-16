import os
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import pylab
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#plt.rcParams['figure.figsize'] = (26.3, 1.28)# (17, 5)
tracks = pd.read_csv("metadata/set.csv",usecols=["fname","label"])
wavfiles=tracks['fname'].tolist()
for i in wavfiles:
	x, sr = librosa.load(i)   
	stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))

	mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
	kwargs = {'cmap': 'gray'}
	log_mel = librosa.amplitude_to_db(mel)

        # adjusting
	log_mel = log_mel[:, :2580]
        # zero padding
	log_mel = np.pad(log_mel, ((0, 0), (25, 25)), 'constant', constant_values=0)

        # mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        # mfcc = sklearn.preprocessing.StandardScaler().fit_transform(mfcc)
	fig = plt.figure(frameon=False)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	fig.add_axes(ax)
	librosa.display.specshow(log_mel, sr=sr, hop_length=512, x_axis='time', y_axis='mel', **kwargs)
        # display.specshow(mfcc, sr=sr, x_axis='time', **kwargs)
	#Crea una carpeta llamada imagenes
	j=i[3:]
	fig.savefig('imagenes/'+j[:-3]+'jpg')
	fig.clf()
	plt.close()
