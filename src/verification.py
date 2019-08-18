import os

import keras
import numpy as np

import deep_gradient_boosting_net as dgbn

def get_cifar10_data():
    num_classes = 10

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    # 
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = (x_train - 122.5) / 255
    x_test = (x_test - 122.5) / 255

    return x_train, y_train, x_test, y_test

def veri_dgbn_cifar10():
    SAVE_RESULT_DIR = os.path.join(os.getcwd(),'result_dgbn')

    # cifar10 data
    x_train, y_train, x_test, y_test = get_cifar10_data()

    # train model
    boosting_num = 2
    shrinkage = 1.0
    #shrinkage = 0.06
    print('shrinkage : {0}'.format(shrinkage))
    l2 = 0.0
    deep_gb_net = dgbn.DeepGBnet(boosting_num=boosting_num, shrinkage=shrinkage, l2=l2)

    epochs = 100
    batch_size = 256
    #do_subsampling = True
    do_subsampling = False
    mask_rate = 0.0
    #mask_rate = 0.1
    deep_gb_net.train_boosting_model(x_train, y_train, x_test, y_test, epochs, batch_size, do_subsampling, mask_rate, SAVE_RESULT_DIR)
    
    # save result
    deep_gb_net.save_model(deep_gb_net.classify_model, os.path.join(SAVE_RESULT_DIR, 'trained_model.h5'))

    #
    print('\nscore')
    for iboost in range(boosting_num):
        print('i, tr_acc, ts_acc : {0}, {1}, {2}'.format(iboost+1, deep_gb_net.train_scores[iboost][1], deep_gb_net.test_scores[iboost][1]))

    return

veri_dgbn_cifar10()