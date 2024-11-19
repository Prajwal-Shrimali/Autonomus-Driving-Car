import win32api as wapi
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys

#To get AWSD key positions
def key_out(key):
    output = [0, 0, 0, 0]
    if 'A' in key:
        output[0] = 1
    if 'D' in key:
        output[3] = 1
    if 'W' in key:
        output[1] = 1
    if 'S' in key:
        output[2] = 1
    return output





# log.py
# import ctypes

# # Key codes for 'A', 'W', 'D', 'S' keys
# key_list = ['A', 'W', 'D', 'S']
# VK_CODE = {'A': 0x41, 'W': 0x57, 'D': 0x44, 'S': 0x53}

# def key_check():
#     keys = []
#     for key in key_list:
#         if ctypes.windll.user32.GetAsyncKeyState(VK_CODE[key]):
#             keys.append(key)
#     return keys

#for testing
#while True:
#    key = key_check()
#    a = key_out(key)
#    print(a)