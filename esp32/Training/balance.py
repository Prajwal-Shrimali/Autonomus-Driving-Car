# #from  https://github.com/Sentdex/pygta5

# import numpy as np 
# import pandas as pd
# from collections import Counter
# from random import shuffle
# import cv2

# train_data = np.load('training_data.npy', allow_pickle=True) 

# df = pd.DataFrame(train_data)
# print(df.head())
# print(Counter(df[1].apply(str)))

# lefts = []
# rights = []
# forwards = []

# shuffle(train_data)

# for data in train_data:
#     img = data[0]
#     choice = data[1]
#     #print(choice)
#     if choice == [1, 1, 0]:
#         lefts.append([img, choice])
#     elif choice == [0, 1, 1]:
#         rights.append([img, choice])
#     elif choice == [0, 1, 0]:
#         forwards.append([img, choice])
#     else:
#         pass#print('no matches')


# forwards = forwards[:len(lefts)][:len(rights)]
# rights = rights[:len(forwards)]
# lefts = lefts[:len(forwards)]
# print(len(forwards), len(rights), len(lefts))
# final_data = forwards + lefts + rights

# shuffle(final_data)
# print(len(final_data))
# np.save('balanced.npy', final_data)

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

# Load the training data (as a NumPy array)
train_data = np.load('training_data.npy', allow_pickle=True)

# Convert the numpy array to a regular Python list for processing
train_data = train_data.tolist()

# Convert to DataFrame for analysis (optional, for inspecting data)
df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

# Prepare lists for each action
lefts = []
rights = []
forwards = []

# Shuffle data
shuffle(train_data)

# Sort data into respective lists based on key presses (choices)
for data in train_data:
    img = data[0]  # Image data
    choice = data[1].tolist()  # Convert NumPy array to Python list for comparison

    # Depending on key states, categorize the data
    if choice == [1, 1, 0]:
        lefts.append([img, choice])
    elif choice == [0, 1, 1]:
        rights.append([img, choice])
    elif choice == [0, 1, 0]:
        forwards.append([img, choice])
    else:
        pass  # Ignore invalid data, or print if needed

# Balancing the dataset by trimming excess data to match the smallest class
forwards = forwards[:len(lefts)][:len(rights)]
rights = rights[:len(forwards)]
lefts = lefts[:len(forwards)]

# Printing the lengths of each class after balancing
print(len(forwards), len(rights), len(lefts))

# Combine the classes into a final dataset and shuffle
# final_data = forwards + lefts + rights
# shuffle(final_data)

# # Print the final size of the dataset
# print(len(final_data))
# Combine the classes into a final dataset and shuffle
final_data = forwards + lefts + rights
shuffle(final_data)
print(len(final_data))


# Convert final_data to a NumPy object array
final_data_array = np.array(final_data, dtype=object)

# Save the balanced dataset
np.save('balanced.npy', final_data_array)


# # Save the balanced dataset
# np.save('balanced.npy', final_data)
