""" Alexnet.

References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.

"""

# import tflearn
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.estimator import regression
# from tflearn.layers.normalization import local_response_normalization

# def alexnet(width, height, lr):
#     network = input_data(shape=[None, width, height, 1], name='input')
#     network = conv_2d(network, 96, 11, strides=4, activation='relu')
#     network = max_pool_2d(network, 3, strides=2)
#     network = local_response_normalization(network)
#     network = conv_2d(network, 256, 5, activation='relu')
#     network = max_pool_2d(network, 3, strides=2)
#     network = local_response_normalization(network)
#     network = conv_2d(network, 384, 3, activation='relu')
#     network = conv_2d(network, 384, 3, activation='relu')
#     network = conv_2d(network, 256, 3, activation='relu')
#     network = max_pool_2d(network, 3, strides=2)
#     network = local_response_normalization(network)
#     network = fully_connected(network, 4096, activation='tanh')
#     network = dropout(network, 0.5)
#     network = fully_connected(network, 4096, activation='tanh')
#     network = dropout(network, 0.5)
#     network = fully_connected(network, 3, activation='softmax')
#     network = regression(network, optimizer='momentum',
#                          loss='categorical_crossentropy',
#                          learning_rate=lr, name='targets')

#     model = tflearn.DNN(network, checkpoint_path='model_alexnet',
#                         max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')

#     return model

# import tflearn
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.estimator import regression

# def alexnet(width, height, lr):
#     network = input_data(shape=[None, width, height, 1], name='input')
#     network = conv_2d(network, 96, 11, strides=4, activation='relu')
#     network = max_pool_2d(network, 3, strides=2)
#     network = conv_2d(network, 256, 5, activation='relu')
#     network = max_pool_2d(network, 3, strides=2)
#     network = conv_2d(network, 384, 3, activation='relu')
#     network = conv_2d(network, 384, 3, activation='relu')
#     network = conv_2d(network, 256, 3, activation='relu')
#     network = max_pool_2d(network, 3, strides=2)
#     network = fully_connected(network, 4096, activation='relu')
#     network = dropout(network, 0.5)
#     network = fully_connected(network, 4096, activation='relu')
#     network = dropout(network, 0.5)
#     network = fully_connected(network, 4, activation='softmax')  # Change this to 4 for 4 classes
#     network = regression(network, optimizer='adam', learning_rate=lr,
#                          loss='categorical_crossentropy', name='targets')
#     model = tflearn.DNN(network, tensorboard_dir='log')

#     return model

# import tflearn
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.estimator import regression

# def alexnet(width, height, lr):
#     network = input_data(shape=[None, width, height, 1], name='input')
#     network = conv_2d(network, 96, 11, strides=4, activation='relu')
#     network = max_pool_2d(network, 3, strides=2)
#     network = conv_2d(network, 256, 5, activation='relu')
#     network = max_pool_2d(network, 3, strides=2)
#     network = conv_2d(network, 384, 3, activation='relu')
#     network = conv_2d(network, 384, 3, activation='relu')
#     network = conv_2d(network, 256, 3, activation='relu')
#     network = max_pool_2d(network, 3, strides=2)
#     network = fully_connected(network, 4096, activation='relu')
#     network = dropout(network, 0.5)
#     network = fully_connected(network, 4096, activation='relu')
#     network = dropout(network, 0.5)
#     network = fully_connected(network, 3, activation='softmax')  # Change this to 3 for 3 classes
#     network = regression(network, optimizer='adam', learning_rate=lr,
#                          loss='categorical_crossentropy', name='targets')
#     model = tflearn.DNN(network, tensorboard_dir='log')

#     return model

# import tflearn
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.estimator import regression

# def alexnet(width, height, lr):
#     network = input_data(shape=[None, width, height, 1], name='input')
#     network = conv_2d(network, 96, 11, strides=4, activation='relu')
#     network = max_pool_2d(network, 3, strides=2)
#     network = conv_2d(network, 256, 5, activation='relu')
#     network = max_pool_2d(network, 3, strides=2)
#     network = conv_2d(network, 384, 3, activation='relu')
#     network = conv_2d(network, 384, 3, activation='relu')
#     network = conv_2d(network, 256, 3, activation='relu')
#     network = max_pool_2d(network, 3, strides=2)
#     network = fully_connected(network, 4096, activation='relu')
#     network = dropout(network, 0.5)
#     network = fully_connected(network, 4096, activation='relu')
#     network = dropout(network, 0.5)
#     network = fully_connected(network, 3, activation='softmax')  # Changed to 3 classes
#     network = regression(network, optimizer='adam', learning_rate=lr,
#                          loss='categorical_crossentropy', name='targets')
#     model = tflearn.DNN(network, tensorboard_dir='log')

#     return model

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

def alexnet(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='input')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=lr, name='targets')

    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')

    return model