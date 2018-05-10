import numpy as np
import matplotlib.pyplot as plt
# import sklearn.preprocessing
import librosa
from librosa import display

import utils

DIR = "./FMA/fma_medium/"
DIR_IM = "./FMA/fma_data/"


path = "./FMA/"

tracks = utils.load(path)


med = tracks['set', 'subset'] <= 'medium'

train = tracks['set', 'split'] == 'training'
val = tracks['set', 'split'] == 'validation'
test = tracks['set', 'split'] == 'test'

df_train = tracks.loc[med & train, ('track', 'genre_top')]
df_valid = tracks.loc[med & val, ('track', 'genre_top')]
df_test = tracks.loc[med & test, ('track', 'genre_top')]

plt.rcParams['figure.figsize'] = (26.3, 1.28)# (17, 5)

audio_train, y_train = utils.get_audio_files_v2(df_train, DIR)
audio_valid, y_valid = utils.get_audio_files_v2(df_valid, DIR)
audio_test, y_test = utils.get_audio_files_v2(df_test, DIR)


def gen_im(files, path):

    for raw_audio in utils.get_audio_gen(files):
        # Duration: 30.00s, 1323119 samples   ... MAX
        # Power Spectogram

        x, sr = raw_audio[0]
        name_file = raw_audio[1].split("/")[-1].split(".")[0]
        dir = raw_audio[2]

        try:
            stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        except:
            print("ERROR Value Error")
            print("file, ", raw_audio[1])
            print("x, sr", x, sr)
            continue
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

        display.specshow(log_mel, sr=sr, hop_length=512, x_axis='time', y_axis='mel', **kwargs)
        # display.specshow(mfcc, sr=sr, x_axis='time', **kwargs)

        fig.savefig(path+str(dir)+"/"+name_file)
        fig.clf()
        plt.close()


print("saving data")
gen_im(list(zip(audio_train, y_train)), DIR_IM+"train/")
gen_im(list(zip(audio_valid, y_valid)), DIR_IM+"valid/")
gen_im(list(zip(audio_test, y_test)), DIR_IM+"test/")
print("data stored")






