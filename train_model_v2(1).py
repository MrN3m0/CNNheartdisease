import utils
import os
import pickle
import sys
from models import FCCNN
from keras.preprocessing.image import ImageDataGenerator

# from keras.utils import np_utils

if __name__ == "__main__":
    DIR = "./FMA/"
    DIR_IM = "./FMA/fma_data/"
    PATH_W = "model_w.h5"
    OBJ_HISTORY = "history"

    tensor_shape = utils.get_shape_v2(DIR_IM)
    target_size = (tensor_shape[0], tensor_shape[1])

    model = FCCNN(tensor_shape)

    if os.path.exists(PATH_W):
        model.load_weights(PATH_W)

    datagen_i = ImageDataGenerator()

    train_datagen = datagen_i.flow_from_directory(
        DIR_IM + "train/",
        batch_size=4,
        target_size=target_size,
        class_mode='categorical',
        color_mode='grayscale'
    )
    #
    valid_datagen = datagen_i.flow_from_directory(
        DIR_IM + "test/",
        batch_size=4,
        target_size=target_size,
        class_mode='categorical',
        color_mode='grayscale'
    )
    history = []
    try:
        history = model.fit_generator(train_datagen,
                                      2000,
                                      200,
                                      valid_datagen,
                                      500)

        model.save_weights(PATH_W)

        pickle.dump(history, open(OBJ_HISTORY, 'wb'))
    except KeyboardInterrupt:
        print("\n\n-----Interruption-----\nSaving weights")
        model.save_weights(PATH_W)
        print("Weights saved")

        print("\n\nSaving tmp History")
        pickle.dump(history, open(OBJ_HISTORY, 'wb'))
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

