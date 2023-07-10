import gymnasium as gym
import stable_baselines3
import sys
import math 
import random 
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import carla
from gym.spaces import Discrete, Box
sys.path.append(("C:/CARLA/WindowsNoEditor/PythonAPI/carla"))
from agents.navigation.global_route_planner import GlobalRoutePlanner

'''
Notes:
Each episode should be 10 seconds long?


'''
############# Declaring Global Variables ################
SHOW_PREVIEW = True 
VEHICLE_MODEL = 'vehicle.tesla.model3'
HOST = 'localhost'
PORT = 2000
IMG_WIDTH = 640
IMG_HEIGHT = 480
FOV = 110
NO_RENDER_MODE = False
TIMEOUT = 10.0
SHOW_LOCAL_VIEW= True
SECONDS_PER_EP = 100
SAMPLING_RESOLUTION = 1


class CarENV(gym.Env):
    
    def __init__(self, Vehicle_model = VEHICLE_MODEL, host = HOST, port = PORT,no_render_mode = NO_RENDER_MODE, time_out  = TIMEOUT, IMG_HEIGHT = IMG_HEIGHT , IMG_WIDTH =IMG_WIDTH, show_local_view = SHOW_LOCAL_VIEW, sampling_resolution = SAMPLING_RESOLUTION):
        super().__init__()
        try: 
            print("Please wait as we attempt to connect to the CARLA server.")
            self.client = carla.Client(host, port)        
            self.client.set_timeout(time_out)
            print("You have successfully connected to the CARLA server.")
        except:
            raise Exception("Sorry you were unable to connect to the CARLA server. Check if you have a CARLA server running.")
        self.world = self.client.get_world()
        self.bp_lip = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points() 
        self.vehicle_model = self.bp_lip.filter(Vehicle_model)[0]
        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = False
        self.world.apply_settings(self.settings)
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH =IMG_WIDTH
        self.SHOW_LOCAL_VIEW = show_local_view 
        self.camera_flag = None
        self.camera_observation = None
        self.Global_Route_Planner = GlobalRoutePlanner(self.world.get_map(), sampling_resolution)
    

        
        

        self.action_space = Discrete(3) #Left, Up, right
        self.observation_space = Box(
            low=0, high=255, shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

        '''
        Need to do:
            1. Complete the Reset method, i.e. 
                a) The me

        
        
        '''

    def path_visualizor(self):
         for waypoint in self.route:
            self.world.debug.draw_string(waypoint[0].transform.location, '^', draw_shadow=False,
                color=carla.Color(r=0, g=0, b=255), life_time=60.0,
                persistent_lines=True)


    def reset(self, seed=None, options=None, start_point = None, end_point = None):
        
        self.collision = {'collision':False}  #check the collision sensor
        self.actor_lst = []
        if not start_point:        
            self.start_transform = np.random.choice(self.spawn_points)
        self.vehicle = self.world.spawn_actor(self.vehicle_model, self.start_transform)
        if  self.vehicle == None:
            print("Failed to ")
        self.actor_lst.append(self.vehicle)

        ###### Spawning a top view camera #######
        '''
        Need to:
                1. Adjust location and rotation of the camera so that the camera has a top down view of the CAR: For this you can check the PP code
                2. The start and end points should come in this

        '''
        self.Camera_bp = self.bp_lip.find('sensor.camera.rgb')
        self.Camera_bp.set_attribute("image_size_x", f'{self.IMG_WIDTH}')
        self.Camera_bp.set_attribute("image_size_y", f'{self.IMG_HEIGHT}')
        self.Camera_bp.set_attribute('fov', f"{FOV}")
        
        cam_transform = carla.Transform(self.vehicle.get_transform().transform(carla.Location(x=-5,z=20)), carla.Rotation(yaw=90, pitch=-90))
        transform = carla.Transform(carla.Location(x = 2.5, z = 0.7), carla.Rotation(pitch = 0))
        

        self.Camera  =self.world.try_spawn_actor(self.Camera_bp, cam_transform, attach_to= self.vehicle)
        self.actor_lst.append(self.Camera)
        
        ###### Spawn Collision Sensor ######## sensor.other.collision
        collision_Sensor_bp = self.bp_lip.find("sensor.other.collision")

        
        self.collision_Sensor = self.world.try_spawn_actor(collision_Sensor_bp, transform,  attach_to =  self.vehicle)
        self.actor_lst.append(self.collision_Sensor)
        
        

        
        spectator = self.world.get_spectator() 
        transform = carla.Transform(self.vehicle.get_transform().transform(carla.Location(x=-5,z=20)), carla.Rotation(yaw=90, pitch=-90)) 
        spectator.set_transform(transform)
        

        ''' set sensor recording'''
        self.Camera.listen(lambda data:self.process_img(data))
        self.collision_Sensor.listen(lambda event: self.collision_detector(event))


        '''Apparently this makes the spawning quicker'''

        
        self.control = carla.VehicleControl()
        self.control.throttle = 0.0
        self.control.steer = 0.0
        self.vehicle.apply_control(self.control)
        
        ############ Tracing a Global Route ############
        if not end_point:
            end_point = random.choice(self.spawn_points)

        self.route = self.Global_Route_Planner.trace_route(self.start_transform.location, end_point.location)
        
        ''' This is just to visualize the route
        Make this in
        '''
        self.path_visualizor()


        

        print("The environmnet has been reset")
        
        while self.camera_flag is None:
            time.sleep(0.01)
            print('.',end='')

        self.episode_start_time = time.time()
        info = None
        return self.camera_observation, None





    def step(self, action):
        '''
        The action state is A = {right, center, left}

        What this means is that the 
        '''
        if action == 0:
            # Navigation mark at right
            pass
        if action == 1:
            # Navigation mark at center
            pass
        if action == 2:
            # Navigation mark at left
            pass


        v = self.vehicle.get_velocity()
        v_kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))  

        if self.collision['collision'] == True:
            done = True
            reward  = -10
        else: 
            done = False
            reward = 1
        if self.episode_start_time + SECONDS_PER_EP < time.time():
            done = True
            reward = 0


        truncated = None
        info = None
        return self.camera_observation, reward, done, truncated, None





    def render(self):
        pass

    def close(self):
        pass


    def collision_detector(self, event):
        self.collision['collision'] = True
        print("A collision has occured")
    
        

    def process_img(self, image):
        i = np.array(image.raw_data)
        print("This is called")
        i2 = i.reshape((self.IMG_HEIGHT, self.IMG_WIDTH, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_LOCAL_VIEW:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.camera_observation = i3
        if  self.camera_observation is None:
            self.camera_flag = False
            print(f'This is to show that camera image hasn\'t loaded yet: {i3}')
        else:
            self.camera_flag = True
    

    def destroy_all_actors(self):
        for act in self.actor_lst:
            try: 
                act.destroy()
                print(f"Actor {act} has been destroyed")
            except:
                print(f'Actor {act} was not destroyed')
                pass
        
        
        
    
'''
ADD a destructor in which 
cv2.destroyAllWindows()
cv2.stop()


'''
          
        
if __name__ == '__main__':
    car =  CarENV()
    print(car)


