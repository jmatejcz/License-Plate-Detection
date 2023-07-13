import os

def gmkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)