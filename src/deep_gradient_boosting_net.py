import os

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten
from keras.layers import Add, Lambda
from keras import backend as K
import tensorflow as tf
import numpy as np


class DeepGBnet:
    def __init__(self, boosting_num, shrinkage, l2):
        
        self.BOOSTING_NUM = boosting_num
        self.SHRINKAGE = shrinkage
        self.L2 = l2

        # assume cifar10 image
        self.INPUT_SHAPE = (32, 32, 3)
        self.OUP_CLASS_NUM = 10

        self.classify_model = None

        # datagen
        self.datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0.,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
        
        return

    def train_boosting_model(self, x_tr, y_tr, x_ts, y_ts, epochs, batch_size, do_subsampling=True, mask_rate=0.0, save_result_dir=None):

        self.train_scores = []
        self.test_scores = []

        weak_models = []

        # weak model作成
        # weak modelsからboosting model作成
        # boosting model学習

        for iboost in range(self.BOOSTING_NUM):
            print('\nboosting no. {0} / {1}'.format(iboost+1, self.BOOSTING_NUM))
            # subsample
            if do_subsampling:
                subsamp_idx = self.subsumpling_index(len(x_tr))
                x_subsamp = x_tr[subsamp_idx]
                y_subsamp = y_tr[subsamp_idx]
            else:
                subsamp_idx = np.arange(len(x_tr))
                x_subsamp = x_tr[subsamp_idx]
                y_subsamp = y_tr[subsamp_idx]

            # add new weak model
            weak_models.append(self.build_weak_model(mask_rate))
            # freeze
            for iwm in range(iboost - 1):
                weak_models[iwm].trainable = False
            # make boosting model
            boosting_model = self.build_boosting_model(weak_models, do_shrink=False)
            
            # train boosting model
            opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
            boosting_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            boosting_model.fit_generator(self.datagen.flow(x_subsamp, y_subsamp, batch_size=batch_size),
                                        epochs=epochs, steps_per_epoch=int(len(x_subsamp) / batch_size),
                                        validation_data=(x_ts, y_ts),
                                        workers=3,
                                        )

            # score
            # train
            scores = boosting_model.evaluate(x_tr, y_tr, verbose=0)
            self.train_scores.append([scores[0], scores[1]])
            print('\nTrain loss, accuracy: {0:.4f}, {1:.4f}'.format(scores[0], scores[1]))
            # test
            scores = boosting_model.evaluate(x_ts, y_ts, verbose=0)
            self.test_scores.append([scores[0], scores[1]])
            print('Test loss, accuracy: {0:.4f}, {1:.4f}'.format(scores[0], scores[1]))
            
        # make classify model
        self.classify_model = self.build_boosting_model(weak_models, do_shrink=True)
        self.classify_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        if save_result_dir is not None:
            self.save_plot_model(weak_models[0], os.path.join(save_result_dir, 'weak_model_structure.png'))

        return

    def build_weak_model(self, mask_rate=0.0):
        """
        return weak learner which is base of gradient boosting.
        """

        # model structure
        # input
        input_img = Input(self.INPUT_SHAPE)
        h = input_img
        
        # mask layer
        mask_filter = self.make_mask_filter(mask_rate)
        h = Lambda(lambda x: tf.math.multiply(x, tf.convert_to_tensor(mask_filter, dtype=tf.float32)))(h)

        # conv
        h = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = MaxPooling2D(pool_size=(2, 2))(h)

        h = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = MaxPooling2D(pool_size=(2, 2))(h)

        h = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = MaxPooling2D(pool_size=(2, 2))(h)
        
        # flatten
        #h = Flatten()(h)
        h = GlobalAveragePooling2D()(h)
        # fully conection
        h = Dense(256, kernel_regularizer=keras.regularizers.l2(self.L2), kernel_initializer='he_normal')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Dense(64, kernel_regularizer=keras.regularizers.l2(self.L2), kernel_initializer='he_normal')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Dense(self.OUP_CLASS_NUM, kernel_regularizer=keras.regularizers.l2(self.L2), kernel_initializer='he_normal')(h)

        model = Model(inputs=input_img, outputs=h)

        return model

    def make_mask_filter(self, mask_rate):
        """
        return mask filter of image pixcel.
        """
        # mask
        pix_num = self.INPUT_SHAPE[0] * self.INPUT_SHAPE[1]
        
        # masked pixcel. 0 is masked.
        mask_pix_idx = np.random.choice(np.arange(pix_num), int(mask_rate * pix_num), replace=False)        
        mask_pix_filter = np.full(pix_num, 1.0)
        if len(mask_pix_idx) != 0:
            mask_pix_filter[mask_pix_idx] = 0.0
        mask_pix_filter = np.resize(mask_pix_filter, (self.INPUT_SHAPE[0], self.INPUT_SHAPE[1]))[:,:,np.newaxis]
        
        # mask filter
        mask_filter = mask_pix_filter * np.full(self.INPUT_SHAPE, 1.0)

        return mask_filter

    def build_boosting_model(self, weak_models, do_shrink):
        """
        return boosting model
        """
        # input
        input_img = Input(self.INPUT_SHAPE)        
        
        # boosting
        weak_oups = []
        for iwm, weak_model in enumerate(weak_models):
            # shinkage
            if (iwm == len(weak_models) - 1) and (not do_shrink):
                shrinkage = 1.0
            else:
                shrinkage = self.SHRINKAGE
            # weak model output
            weak_oup = weak_model(input_img)
            weak_oup = Lambda(lambda x: x * shrinkage)(weak_oup)
            weak_oups.append(weak_oup)

        # add
        if len(weak_models) is not 1:
            add_oup = Add()(weak_oups)
        else:
            add_oup = weak_oups[0]
        # classify
        oup = Activation('softmax')(add_oup)
            
        model = Model(inputs=input_img, outputs=oup)

        return model

    def subsumpling_index(self, index_num):
        """
        bagging
        """
        subsamp_idx = np.unique(np.random.choice(np.arange(index_num), size=index_num, replace=True))
        return subsamp_idx

    def save_model(self, model, save_file_name):
        """
        save model
        """
        # dir
        dir_name = os.path.dirname(save_file_name)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        # visualize
        keras.utils.plot_model(model, to_file=os.path.join(dir_name, 'model_structure.png'), show_shapes=True, show_layer_names=False)

        # save model
        model.save(save_file_name)
        print('Saved trained model at %s ' % save_file_name)
        
        return

    def save_plot_model(self, model, save_file_name):
        """
        save model
        """
        # dir
        dir_name = os.path.dirname(save_file_name)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        # visualize
        keras.utils.plot_model(model, to_file=save_file_name, show_shapes=True, show_layer_names=False)
        
        return