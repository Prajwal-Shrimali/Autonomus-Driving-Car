# import cv2
# import numpy as np
# import socket
# from log import key_check
# import os
# import time
# s = socket.socket()         # Create a socket object
# host = '192.168.233.180'            
# port = 12345                # Reserve a port for your service.
# s.connect((host, port))


# cap = cv2.VideoCapture('http://192.168.233.82:81/stream') #esp32cam ip stream. Check on esp32cam serial for ip
# time.sleep(5)               #wait for 5 sec
# print('start')

# #creating file store data
# file_name = 'training_data.npy'
# if os.path.isfile(file_name):
#     print("File exists , loading previous data")
#     training_data = list(np.load(file_name, allow_pickle=True))
# else:
#     print('file does not exist, starting fresh')
#     training_data = []

# def key_out(key):
#     output = [0, 0, 0, 0]
#     if 'A' in key:
#         output[0] = 1
#     if 'D' in key:
#         output[2] = 1
#     if 'W' in key:
#         output[1] = 1
#     if 'S' in key:
#         output[3] = 1
#     return output
    


# while(True):
#     key = key_check()
#     a = key_out(key)

#     # Create a serialized dict
#     x = '{"a":'
#     x += str(a[0])
#     x += ',"d":'
#     x += str(a[2])
#     x += ',"w":'
#     x += str(a[1])
#     x += ',"s":'
#     x += str(a[3])
#     x += "}"
#     msg = str.encode(x, 'utf-8')
#     s.send(msg)
#     data1 = s.recv(1024)
    
#     ret, frame = cap.read()
#     rotate = cv2.rotate(frame, cv2.ROTATE_180)  # Align the video feed
#     screen = cv2.cvtColor(rotate, cv2.COLOR_BGR2GRAY)
#     screen = cv2.resize(screen, (80, 60))
    
#     # Convert 'a' to a NumPy array
#     a = np.array(a, dtype=np.int8)
#     training_data.append([screen, a])
    
#     cv2.imshow('rotate', rotate)
#     cv2.imshow('screen', screen)
    
#     # Save data to file after every 500 data points
#     if len(training_data) % 500 == 0:
#         print(len(training_data))
#         np.save(file_name, training_data)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# cap.release()
# cv2.destroyAllWindows()
















# import cv2
# import numpy as np
# import socket
# from log import key_check
# import os
# import time

# # Create a socket object
# s = socket.socket()
# host = '192.168.233.180'  # Update with your ESP32-CAM IP
# port = 12345  # Reserve a port for your service
# s.connect((host, port))

# cap = cv2.VideoCapture('http://192.168.233.82:81/stream')  # ESP32-CAM stream
# time.sleep(5)  # Wait for the stream to stabilize
# print('start')

# # Creating file to store data
# file_name = 'training_data.npy'
# if os.path.isfile(file_name):
#     print("File exists, loading previous data")
#     training_data = list(np.load(file_name, allow_pickle=True))
# else:
#     print('File does not exist, starting fresh')
#     training_data = []

# def key_out(key):
#     output = [0, 0, 0, 0]
#     if 'A' in key:
#         output[0] = 1
#     if 'D' in key:
#         output[2] = 1
#     if 'W' in key:
#         output[1] = 1
#     if 'S' in key:
#         output[3] = 1
#     return output

# while True:
#     key = key_check()  # Get keyboard input
#     a = key_out(key)  # Map key to control output

#     # Create a serialized dict to send over socket
#     x = '{"a":' + str(a[0]) + ',"d":' + str(a[2]) + ',"w":' + str(a[1]) + ',"s":' + str(a[3]) + '}'
#     msg = str.encode(x, 'utf-8')
#     s.send(msg)
#     data1 = s.recv(1024)  # Receive confirmation from server

#     # Read video frame from the ESP32-CAM
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

#     rotate = cv2.rotate(frame, cv2.ROTATE_180)  # Align video feed
#     screen = cv2.cvtColor(rotate, cv2.COLOR_BGR2GRAY)
#     screen = cv2.resize(screen, (80, 60))  # Resize for storage

#     # Convert key output 'a' to NumPy array
#     a = np.array(a, dtype=np.int8)

#     # Append tuple of (screen, key input) to training_data
#     training_data.append((screen, a))

#     # Display frames
#     cv2.imshow('rotate', rotate)
#     cv2.imshow('screen', screen)

#     # Save data to file after every 500 data points
#     if len(training_data) % 500 == 0:
#         print(f"Saving {len(training_data)} frames")
#         np.save(file_name, np.array(training_data, dtype=object), allow_pickle=True)

#     # Break loop when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()


######################################################################################################################################

# import cv2
# import numpy as np
# import socket
# from log import key_check
# import os
# import time
# s = socket.socket()         # Create a socket object
# host = '192.168.233.180'            
# port = 12345                # Reserve a port for your service.
# s.connect((host, port))


# cap = cv2.VideoCapture('http://192.168.233.82:81/stream') #esp32cam ip stream. Check on esp32cam serial for ip
# time.sleep(5)               #wait for 5 sec
# print('start')

# #creating file store data
# file_name = 'training_data.npy'
# if os.path.isfile(file_name):
#     print("File exists , loading previous data")
#     training_data = list(np.load(file_name, allow_pickle=True))
# else:
#     print('file does not exist, starting fresh')
#     training_data = []

# def key_out(key):
#     output = [0, 0, 0, 0]
#     if 'A' in key:
#         output[0] = 1
#     if 'D' in key:
#         output[2] = 1
#     if 'W' in key:
#         output[1] = 1
#     if 'S' in key:
#         output[3] = 1
#     return output
    


# while(True):
#     key = key_check()
#     a = key_out(key)
    
#     #create a serialized dict
#     x = '{"a":'
#     x += str(a[0])
#     x += ',"d":'
#     x += str(a[2])
#     x += ',"w":'
#     x += str(a[1])
#     x += ',"s":'
#     x += str(a[3])
#     x += "}"
#     msg = str.encode(x, 'utf-8')
#     #print(msg)
#     s.send(msg)
#     data1 = s.recv(1024)
#     ret, frame = cap.read()
#     rotate = cv2.rotate(frame, cv2.ROTATE_180) #align the video feed 
#     screen = cv2.cvtColor(rotate, cv2.COLOR_BGR2GRAY)
#     screen = cv2.resize(screen, (80, 60))
#     a.pop()
#     print(a)
#     training_data.append([screen, a])
#     cv2.imshow('rotate', rotate)
#     cv2.imshow('screen', screen)
#     #save data to file after every 500 data point collection
#     if len(training_data) % 500 == 0:
#         print(len(training_data))
#         np.save(file_name, training_data)
    
   
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import socket
# import os
# import time
# import win32api as wapi

# # Define the keys you want to check
# keyList = ["\b"]
# for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
#     keyList.append(char)

# # Function to check the pressed keys
# def key_check():
#     keys = []
#     for key in keyList:
#         if wapi.GetAsyncKeyState(ord(key)):
#             keys.append(key)
#     return keys

# # Function to map keys to the output
# def key_out(key):
#     output = [0, 0, 0, 0]
#     if 'A' in key:
#         output[0] = 1
#     if 'D' in key:
#         output[3] = 1
#     if 'W' in key:
#         output[1] = 1
#     if 'S' in key:
#         output[2] = 1  # Send S key but don't save it
#     return output

# # Create a socket object
# s = socket.socket()         
# host = '192.168.233.180'  # ESP32 IP address
# port = 12345              # Port for the service
# s.connect((host, port))

# # Open the video stream from the ESP32 camera
# cap = cv2.VideoCapture('http://192.168.233.82:81/stream')  # ESP32 CAM IP stream

# # Create or load the training data file
# file_name = 'training_data.npy'
# if os.path.isfile(file_name):
#     print("File exists, loading previous data")
#     training_data = list(np.load(file_name, allow_pickle=True))
# else:
#     print('File does not exist, starting fresh')
#     training_data = []

# print('Start capturing video')

# while True:
#     keys = key_check()  # Check the key states
#     key_output = key_out(keys)  # Map keys to output (S key is included in output but will not be saved)
    
#     # Create a serialized dict to send to the ESP32
#     x = '{"a":' + str(key_output[0]) + ',"d":' + str(key_output[3]) + ',"w":' + str(key_output[1]) + ',"s":' + str(key_output[2]) + '}'
#     msg = str.encode(x, 'utf-8')
#     s.send(msg)  # Send the key states to the ESP32
#     data1 = s.recv(1024)  # Receive any response from ESP32 (optional)

#     ret, frame = cap.read()  # Capture frame from the video stream
#     screen = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#     screen = cv2.resize(screen, (80, 60))  # Resize to match model input size
    
#     # Append the frame and key states (without the S key) to the training data
#     # We exclude key_output[2] (the S key) from being saved in training data
#     training_data.append([screen, [key_output[0], key_output[1], key_output[3]]])  # Exclude the S key in saved data

#     # Display the video frames
#     cv2.imshow('Original Frame', frame)
#     cv2.imshow('Processed Frame', screen)

#     # Save the training data after every 500 samples
#     if len(training_data) % 500 == 0:
#         print(f"Collected {len(training_data)} samples")
#         np.save(file_name, training_data)

#     # Exit on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import numpy as np
import socket
import os
import time
import win32api as wapi

# Define the keys you want to check
keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

# Function to check the pressed keys
def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys

# Function to map keys to the output
def key_out(key):
    output = [0, 0, 0, 0]
    if 'A' in key:
        output[0] = 1
    if 'D' in key:
        output[3] = 1
    if 'W' in key:
        output[1] = 1
    if 'S' in key:
        output[2] = 1  # Send S key but don't save it
    return output

# Create a socket object
s = socket.socket()         
host = '192.168.233.180'  # ESP32 IP address
port = 12345              # Port for the service
s.connect((host, port))

# Open the video stream from the ESP32 camera
cap = cv2.VideoCapture('http://192.168.233.82:81/stream')  # ESP32 CAM IP stream

# Create or load the training data file
file_name = 'training_data.npy'
if os.path.isfile(file_name):
    print("File exists, loading previous data")
    training_data = list(np.load(file_name, allow_pickle=True))
else:
    print('File does not exist, starting fresh')
    training_data = []

print('Start capturing video')

while True:
    keys = key_check()  # Check the key states
    key_output = key_out(keys)  # Map keys to output (S key is included in output but will not be saved)
    
    # Create a serialized dict to send to the ESP32
    x = '{"a":' + str(key_output[0]) + ',"d":' + str(key_output[3]) + ',"w":' + str(key_output[1]) + ',"s":' + str(key_output[2]) + '}'
    msg = str.encode(x, 'utf-8')
    s.send(msg)  # Send the key states to the ESP32
    data1 = s.recv(1024)  # Receive any response from ESP32 (optional)

    ret, frame = cap.read()  # Capture frame from the video stream
    screen = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    screen = cv2.resize(screen, (80, 60))  # Resize to match model input size
    
    # Ensure the frame and key output have consistent shape
    frame_data = np.array(screen)  # Convert the frame to a NumPy array
    key_data = np.array([key_output[0], key_output[1], key_output[3]])  # Exclude the S key in saved data

    # Append the frame and key states to the training data
    training_data.append([frame_data, key_data])

    # Display the video frames
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Processed Frame', screen)

    # Save the training data after every 500 samples
    if len(training_data) % 500 == 0:
        print(f"Collected {len(training_data)} samples")
        np.save(file_name, np.array(training_data, dtype=object))  # Save as a NumPy array

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


















# import cv2
# import numpy as np
# import socket
# from log import key_check
# import os
# import time
# import pickle

# s = socket.socket()         # Create a socket object
# host = '192.168.233.180'            
# port = 12345                # Reserve a port for your service.
# s.connect((host, port))

# cap = cv2.VideoCapture('http://192.168.233.82:81/stream') #esp32cam ip stream. Check on esp32cam serial for ip
# time.sleep(5)               #wait for 5 sec
# print('start')

# #creating file store data
# file_name = 'training_data.pkl'
# if os.path.isfile(file_name):
#     print("File exists , loading previous data")
#     with open(file_name, 'rb') as f:
#         training_data = pickle.load(f)
# else:
#     print('file does not exist, starting fresh')
#     training_data = []

# def key_out(key):
#     output = [0, 0, 0, 0]
#     if 'A' in key:
#         output[0] = 1
#     if 'D' in key:
#         output[2] = 1
#     if 'W' in key:
#         output[1] = 1
#     if 'S' in key:
#         output[3] = 1
#     return output

# while(True):
#     key = key_check()
#     a = key_out(key)
    
#     #create a serialized dict
#     x = '{"a":'
#     x += str(a[0])
#     x += ',"d":'
#     x += str(a[2])
#     x += ',"w":'
#     x += str(a[1])
#     x += ',"s":'
#     x += str(a[3])
#     x += "}"
#     msg = str.encode(x, 'utf-8')
#     s.send(msg)
#     data1 = s.recv(1024)
#     ret, frame = cap.read()
#     rotate = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) #align the video feed 
#     screen = cv2.cvtColor(rotate, cv2.COLOR_BGR2GRAY)
#     screen = cv2.resize(screen, (80, 60))
#     a.pop()
#     training_data.append((screen, np.array(a)))  # Store as tuple
#     cv2.imshow('rotate', rotate)
#     cv2.imshow('screen', screen)
#     #save data to file after every 500 data point collection
#     if len(training_data) % 500 == 0:
#         print(len(training_data))
#         # Debug: Print shapes of elements in training_data
#         for i, data in enumerate(training_data):
#             print(f"Element {i} shape: {data[0].shape}, {data[1].shape}")
#         with open(file_name, 'wb') as f:
#             pickle.dump(training_data, f)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import socket
# from log import key_check  # Ensure log.py is updated for Windows compatibility
# import os
# import time

# # Create a socket object
# s = socket.socket()
# host = '192.168.233.180'  # Replace with your ESP32-CAM IP address
# port = 12345              # Port for your service
# s.connect((host, port))

# # Video stream from ESP32-CAM
# cap = cv2.VideoCapture('http://192.168.233.82:81/stream')  # Replace with your stream URL
# print('Starting...')

# # Check if training data file exists
# file_name = 'training_data.npy'
# if os.path.isfile(file_name):
#     print("File exists, loading previous data")
#     training_data = list(np.load(file_name, allow_pickle=True))
# else:
#     print('File does not exist, starting fresh')
#     training_data = []

# while True:
#     # Get key presses
#     keys = key_check()
    
#     # Convert keys to output array
#     output = [0, 0, 0]
#     if 'A' in keys:
#         output[0] = 1
#     if 'W' in keys:
#         output[1] = 1
#     if 'D' in keys:
#         output[2] = 1
    
#     # Create a serialized JSON message
#     x = '{"a":' + str(int('A' in keys))
#     x += ',"d":' + str(int('D' in keys))
#     x += ',"w":' + str(int('W' in keys))
#     x += ',"s":' + str(int('S' in keys)) + '}'
#     msg = str.encode(x, 'utf-8')
#     s.send(msg)
#     data1 = s.recv(1024)
    
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame")
#         continue

#     # Rotate and preprocess the frame
#     rotate = cv2.rotate(frame, cv2.ROTATE_180)
#     screen = cv2.cvtColor(rotate, cv2.COLOR_BGR2GRAY)
#     screen = cv2.resize(screen, (80, 60))
    
#     # Append data to training set
#     training_data.append([screen, output])

#     # Display the images
#     cv2.imshow('Rotate', rotate)
#     cv2.imshow('Screen', screen)

#     # Save data every 500 frames
#     if len(training_data) % 500 == 0:
#         print(len(training_data))
#         np.save(file_name, np.array(training_data, dtype=object))
    
#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()