# #From https://github.com/Sentdex/pygta5
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING, 3 = ERROR)

# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# import numpy as np
# from alexnet import alexnet

# WIDTH = 80
# HEIGHT = 60
# LR = 1e-3
# EPOCHS = 8
# MODEL_NAME = 'autonomous_car-{}-{}-{}-epochs.model'.format(LR, 'alexnet',EPOCHS)

# model = alexnet(WIDTH, HEIGHT, LR)

# train_data = np.load('training_data.npy', allow_pickle=True)

# train = train_data[:-200]
# test = train_data[-200:]

# X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
# Y = [i[1] for i in train]

# test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
# test_y = [i[1] for i in test]

# #Now we can actually train the model with:
# model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
#     snapshot_step=500, show_metric=True, run_id=MODEL_NAME)



# model.save(MODEL_NAME)


# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING, 3 = ERROR)

# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# import numpy as np
# from alexnet import alexnet

# WIDTH = 80
# HEIGHT = 60
# LR = 1e-3
# EPOCHS = 8
# MODEL_NAME = 'autonomous_car_1-{}-{}-{}-epochs.model'.format(LR, 'alexnet', EPOCHS)

# model = alexnet(WIDTH, HEIGHT, LR)

# train_data = np.load('training_data.npy', allow_pickle=True)

# train = train_data[:-200]
# test = train_data[-200:]

# X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
# Y = [i[1] for i in train]

# test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
# test_y = [i[1] for i in test]

# # Print shapes of Y and test_y
# print("Shape of Y:", np.array(Y).shape)
# print("Shape of test_y:", np.array(test_y).shape)

# # Now we can actually train the model with:
# model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
#           snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

# model.save(MODEL_NAME)

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING, 3 = ERROR)

# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# import numpy as np
# import pickle
# from alexnet import alexnet

# WIDTH = 80
# HEIGHT = 60
# LR = 1e-3
# EPOCHS = 8
# MODEL_NAME = 'autonomous_car_1-{}-{}-{}-epochs.model'.format(LR, 'alexnet', EPOCHS)

# model = alexnet(WIDTH, HEIGHT, LR)

# # Load training data using pickle
# file_name = 'training_data.pkl'
# with open(file_name, 'rb') as f:
#     train_data = pickle.load(f)

# train = train_data[:-200]
# test = train_data[-200:]

# X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
# Y = [i[1] for i in train]

# test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
# test_y = [i[1] for i in test]

# # Print shapes of Y and test_y
# print("Shape of Y:", np.array(Y).shape)
# print("Shape of test_y:", np.array(test_y).shape)

# # Now we can actually train the model with:
# model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
#           snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

# model.save(MODEL_NAME)

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# import numpy as np
# from alexnet import alexnet

# WIDTH = 80
# HEIGHT = 60
# LR = 1e-3
# EPOCHS = 8
# MODEL_NAME = 'autonomous_car_1-{}-{}-{}-epochs.model'.format(LR, 'alexnet', EPOCHS)

# model = alexnet(WIDTH, HEIGHT, LR)

# # Load training data
# train_data = np.load('training_data.npy', allow_pickle=True)

# # Split data into training and testing sets
# train = train_data[:-200]
# test = train_data[-200:]

# # Prepare training data
# X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
# Y = np.array([i[1] for i in train])

# # Prepare testing data
# test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
# test_y = np.array([i[1] for i in test])

# # Print shapes for debugging
# print("Shape of X:", X.shape)
# print("Shape of Y:", Y.shape)
# print("Shape of test_x:", test_x.shape)
# print("Shape of test_y:", test_y.shape)

# # Train the model
# model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS,
#           validation_set=({'input': test_x}, {'targets': test_y}),
#           snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

# # Save the model
# model.save(MODEL_NAME)

# #From https://github.com/Sentdex/pygta5
# import numpy as np
# from alexnet import alexnet

# WIDTH = 80
# HEIGHT = 60
# LR = 1e-3
# EPOCHS = 8
# MODEL_NAME = 'autonomous_car_1-{}-{}-{}-epochs.model'.format(LR, 'alexnet',EPOCHS)

# model = alexnet(WIDTH, HEIGHT, LR)

# train_data = np.load('training_data.npy', allow_pickle=True)

# train = train_data[:-200]
# test = train_data[-200:]

# X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
# Y = [i[1] for i in train]

# test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
# test_y = [i[1] for i in test]

# #Now we can actually train the model with:
# model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
#     snapshot_step=500, show_metric=True, run_id=MODEL_NAME)



# model.save(MODEL_NAME)

# #From https://github.com/Sentdex/pygta5
# import numpy as np
# from alexnet import alexnet

# WIDTH = 80
# HEIGHT = 60
# LR = 1e-3
# EPOCHS = 8
# MODEL_NAME = 'autonomous_car_1-{}-{}-{}-epochs.model'.format(LR, 'alexnet',EPOCHS)

# model = alexnet(WIDTH, HEIGHT, LR)

# train_data = np.load('balanced.npy', allow_pickle=True)

# train = train_data[:-200]
# test = train_data[-200:]

# X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
# Y = [i[1] for i in train]

# test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
# test_y = [i[1] for i in test]

# #Now we can actually train the model with:
# model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
#     snapshot_step=500, show_metric=True, run_id=MODEL_NAME)



# model.save(MODEL_NAME)

import numpy as np
from alexnet import alexnet

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'autonomous_car_1-{}-{}-{}-epochs.model'.format(LR, 'alexnet', EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

train_data = np.load('balanced.npy', allow_pickle=True)

# Dynamic splitting based on dataset size
split_index = int(0.8 * len(train_data))  # 80% for training, 20% for testing
train = train_data[:split_index]
test = train_data[split_index:]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = [i[1] for i in test]

print(f"Training samples: {len(X)}, Testing samples: {len(test_x)}")

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)
