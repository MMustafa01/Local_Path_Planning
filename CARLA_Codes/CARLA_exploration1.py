'''
This folder will be following the CARLA documentation given at:
https://carla.readthedocs.io/en/latest/foundations/
'''
import glob
import os
import sys
import random
import time
import numpy as np
import cv2
try:
    sys.path.append(glob.glob('C:/CARLA/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla



try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)

    world = client.get_world()



finally:
    print("The simulation is done")