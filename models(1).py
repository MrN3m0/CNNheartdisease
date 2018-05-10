import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
import keras.backend as K

class FCCNN:

    def __init__(self, tensor_shape):
        self.model = self._create_model(tensor_shape)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.adam(),
                           metrics=[self.metric_auc_roc_score])
        print(self.model.summary())

    @staticmethod
    def _create_model(tensor_shape):
        x = Input(shape=tensor_shape)
        h = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
        h = BatchNormalization()(h)
        h = MaxPooling2D(pool_size=(2, 4))(h)
        h = Dropout(0.4)(h)
        h = Conv2D(filters=384, kernel_size=(3, 3), activation='relu')(h)
        h = BatchNormalization()(h)
        h = MaxPooling2D(pool_size=(2, 4))(h)
        h = Dropout(0.4)(h)
        h = Conv2D(filters=768, kernel_size=(3, 3), activation='relu')(h)
        h = BatchNormalization()(h)
        h = MaxPooling2D(pool_size=(2, 4))(h)
        h = Dropout(0.4)(h)
        h = Conv2D(filters=2048, kernel_size=(3, 3), activation='relu')(h)
        h = BatchNormalization()(h)
        h = MaxPooling2D(pool_size=(2, 4))(h)
        h = Dropout(0.4)(h)
        h = Flatten()(h)
        o = Dense(units=16, activation='softmax')(h)

        return Model(inputs=x, outputs=o)

    def fit(self, x_train, y_train, epochs=10, batch_size=30):
        self.model.fit(x_train, y_train, # batch_size=batch_size,
                       # epochs=epochs,
                       verbose=1)

    def fit_generator(self, gen, steps_per_epoch, epochs, valid_data, valid_steps):
        self.model.fit_generator(generator=gen,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 validation_data=valid_data,
                                 validation_steps=valid_steps,
                                 verbose=1)

    # Desc: (loss, accuracy)
    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=0)

    def save_weights(self, path):
        self.model.save_weights(filepath=path)

    def load_weights(self, path):
        self.model.load_weights(filepath=path, by_name=False)

    def predict(self, x):
        return self.model.predict(x)

    # ToDo fix this
    # https://github.com/keras-team/keras/issues/6050
    # https://github.com/keras-team/keras/issues/4402
    @staticmethod
    def metric_auc_roc_score(y_true, y_pred):
        """
                Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
                from prediction scores.
        """

        # sess = K.get_session()
        # init_l = tf.local_variables_initializer()
        # init_g = tf.global_variables_initializer()
        # sess.run(init_l)
        # sess.run(init_g)

        value, update_op = tf.metrics.auc(y_true, y_pred)

        metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

        for v in metric_vars:
            tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
            return value



