RL_Carla_env.py
import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


from stable_baselines3 import ppo
import carla
import random
import matplotlib.pyplot as plt
from time import sleep



print(f'This is PPO {type(ppo)} \n\n\n\n\n\n')


# Connect to the CARLA simulator
client = carla.Client('localhost', 2000)
client.set_timeout(200.0)

# Get the world and blueprint library
world = client.get_world()
blueprint_library = world.get_blueprint_library()

#get spawn points
spawn_points = world.get_map().get_spawn_points()

# Spawn a vehicle
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))

for v in world.get_actors().filter('*vehicle*'):
    v.set_autopilot(True)
    sleep(10)
    print("Autopiolet")

print("This gets done")

for v in world.get_actors().filter('*vehicle*'):
    v.destroy()
    print("Destroy this")
    
# import glob
# import os
# import sys
# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass
# import carla

# import random
# import time
# import numpy as np
# import cv2

# im_width = 640
# im_height = 480


# def process_img(image):
#     i = np.array(image.raw_data)
#     i2 = i.reshape((im_height, im_width, 4))
#     i3 = i2[:, :, :3]
#     cv2.imshow("", i3)
#     cv2.waitKey(1)
#     return i3/255.0


# actor_list = []
# try:
#     client = carla.Client('localhost', 2000)
#     client.set_timeout(2.0)

#     world = client.get_world()

#     blueprint_library = world.get_blueprint_library()

#     bp = blueprint_library.filter('model3')[0]
#     print(bp)

#     spawn_point = random.choice(world.get_map().get_spawn_points())

#     vehicle = world.spawn_actor(bp, spawn_point)
#     vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
#     actor_list.append(vehicle)

#     # sleep for 5 seconds, then finish:
#     time.sleep(5)

# finally:

#     print('destroying actors')
#     for actor in actor_list:
#         actor.destroy()
#     print('done.')