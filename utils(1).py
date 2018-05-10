import pandas as pd
import os
import librosa
import numpy as np
import glob
from keras.preprocessing import image


def load(path):
    tracks = pd.read_csv(path + "fma_metadata/tracks.csv",
                         index_col=0,
                         header=[0, 1])

    COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
               ('track', 'genres'), ('track', 'genres_all'),
               ('track', 'genre_top')]

    SUBSETS = ('small', 'medium', 'large')
    tracks['set', 'subset'] = tracks['set', 'subset'].astype(
        'category', categories=SUBSETS, ordered=True)
    return tracks


def _get_num_p(file_name):
    file_name = file_name.split(".")[0]
    return int(file_name)


def _get_num_p2(file_name):
    t_str = file_name.split(".")[1]
    number = t_str.split("/")[-1]
    return int(number)


def get_audio_files(y, DIR):
    keys = y.keys()
    vec = []

    dir_f = sorted(os.listdir(DIR))

    for dir in dir_f:
        dir_data = sorted(os.listdir(DIR+dir+"/"))

        data = list(map(_get_num_p, dir_data))

        i = 0
        for k in data:
            if k in keys:
                vec.append(DIR+dir+"/"+dir_data[i])
            i = i + 1

    return vec


def get_audio_files_v2(serie, DIR):
    keys = serie.keys()
    vec = []

    dir_f = sorted(os.listdir(DIR))
    for dir in dir_f:
        dir_data = sorted(os.listdir(DIR+dir+"/"))

        data = list(map(_get_num_p, dir_data))

        i = 0
        for k in data:
            if k in keys:
                vec.append(DIR+dir+"/"+dir_data[i])
            i = i + 1

    # Some audio files where missing
    index_s = keys
    index_f = list(map(_get_num_p2, vec))

    index = sorted(list(set(index_s) & set(index_f)))

    y_labels = serie.loc[index].values

    mapl = {}
    for l in y_labels:
        if l not in mapl.keys():
            mapl[l] = len(mapl)

    y = [mapl[k] for k in y_labels]

    return vec, y


def get_audio_gen(list_files):
    for file in list_files:
        yield (librosa.load(file[0], sr=None, mono=True), file[0], file[1])


def get_vectors(serie, path):

    files = sorted(os.listdir(path))

    # Some audio files where missing
    index_s = serie.axes[0]
    index_f = list(map(_get_num_p, files))

    index = sorted(list(set(index_s) & set(index_f)))

    y_labels = serie.loc[index].values

    mapl = {}
    for l in y_labels:
        if l not in mapl.keys():
            mapl[l] = len(mapl)

    y = [mapl[k] for k in y_labels]
    x = sorted(glob.glob(path+"*"))

    return x, y


def get_shape(x_file):
    i = image.load_img(x_file[0])
    v = image.img_to_array(i)
    return v.shape


def get_shape_v2(path):
    path_i = path
    t_dir = os.listdir(path)
    while len(t_dir) > 0:
        path_i += t_dir[0] + "/"
        if not os.path.exists(path_i):
            break
        t_dir = os.listdir(path_i)
    path_i = path_i[:-1]
    i = image.load_img(path_i, True)
    v = image.img_to_array(i)
    return v.shape


def get_im_test(path):
    path_i = path
    t_dir = os.listdir(path)
    while len(t_dir) > 0:
        path_i += t_dir[0] + "/"
        if not os.path.exists(path_i):
            break
        t_dir = os.listdir(path_i)
    path_i = path_i[:-1]
    i = image.load_img(path_i, True)
    v = image.img_to_array(i)
    return v


def datagen(x, y, batches=1):
    # Todo split the images..?
    vx = []
    vy = []
    batch = 0
    for index in range(len(x)):
        i = image.load_img(x[index])
        vx.append(image.img_to_array(i))
        vy.append(y[index])

        batch += 1
        if batch == batches:
            yield np.array(vx), np.array(vy)
            batch = 0
            vx.clear()
            vy.clear()
