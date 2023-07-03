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

    '''
    To change map use the following command
    Note: 
    1. It takes time to change the map, thus increase the timeout from 20 to 200
    '''
    # print(client.get_available_maps()) #This is to to check what maps are available
    # world = client.load_world('/Game/Carla/Maps/Town10HD')

    ###### Getting the world map #######
    world = client.get_world()
    ###### blueprint is needed to get any actor #####
    bp_lib = world.get_blueprint_library()
    
    spawn_points = world.get_map().get_spawn_points()
    #### Try to get the coordinates through spawn points ####
    print(f"The spawn points are given at: \t{spawn_points[0].location}")
    '''Note to self:
    1. __dir__() is a useful python method to get attributes
    2. spawn points have an attribute .localtion that gives location information
    '''

    
    ########################### Loading Modeles ################################
    
    vehicle = bp_lib.find('vehicle.lincoln.mkz_2020')
    vehicle = world.try_spawn_actor(vehicle, random.choice(spawn_points)) # use the .try_spawn_actor method always since sometimes the actor may collide at that spawn point
    

    ############################ Specter Mode ##################################

    t0 = time.clock()
    while True:
        spectator = world.get_spectator() 
        transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation) 
        spectator.set_transform(transform) 

        vehicle.set_autopilot(True)
        tn = time.clock()
        t_diff = tn - t0
        print(t_diff)
        if t_diff >= 60:
            break

finally:

    try:
        vehicle.destroy()
    except:
        print("could not delete vehice")
    print("The simulation is done")



    '''
    Things that will be used later
    print(world.get_actors().find(471))
    world.get_actors().find(471).destroy()
    

    if cv2.waitkey(1) == ord('q')
    


    ### Vey important piece of code
    for bp in bp_lib.filter("vehicle"):
        print(bp.id)

    '''