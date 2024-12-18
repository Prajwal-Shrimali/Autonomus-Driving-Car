import numpy as np 
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

# train_data = np.load('balanced.npy', allow_pickle=True)
train_data = np.load('training_data.npy', allow_pickle=True)




for data in train_data:
    img = data[0]
    choice = data[1]
    img2 = cv2.resize(img, (400, 300))
    cv2.imshow('test', img2)
    print(choice)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break


# import numpy as np 
# import pandas as pd
# from collections import Counter
# from random import shuffle
# import cv2
# import pickle

# # Load the training data using pickle
# file_name = 'training_data.pkl'
# with open(file_name, 'rb') as f:
#     train_data = pickle.load(f)

# # Display the data frames
# for data in train_data:
#     img = data[0]
#     choice = data[1]
#     img2 = cv2.resize(img, (400, 300))
#     cv2.imshow('test', img2)
#     print(choice)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break