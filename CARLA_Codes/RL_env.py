import  gym
import sys
import math 
import random 
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import carla
from gym.spaces import Discrete, Box
# sys.path.append(("C:/CARLA/WindowsNoEditor/PythonAPI/carla"))
sys.path.append(("C:/Users/netbot/CARLA/WindowsNoEditor/PythonAPI/carla"))
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
PARAM = [6.3, 2.875] # [ld: look ahead distance, wheel based] 
REWARD_ARR = [-1, -100, 10, 1000, -50]


class CarENV(gym.Env):
    
    def __init__(self, Vehicle_model = VEHICLE_MODEL, host = HOST, port = PORT,no_render_mode = NO_RENDER_MODE, time_out  = TIMEOUT, IMG_HEIGHT = IMG_HEIGHT , IMG_WIDTH =IMG_WIDTH, show_local_view = SHOW_LOCAL_VIEW, sampling_resolution = SAMPLING_RESOLUTION, model = PARAM, seconds_per_ep = SECONDS_PER_EP, reward_arr = REWARD_ARR, Debugger = True):
        super().__init__()
        try: 
            #print("Please wait as we attempt to connect to the CARLA server.")
            self.client = carla.Client(host, port)        
            self.client.set_timeout(time_out)
            #print("You have successfully connected to the CARLA server.")
        except:
            raise Exception("Sorry you were unable to connect to the CARLA server. Check if you have a CARLA server running.")
        self.world = self.client.get_world()
        self.bp_lip = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points() 
        self.vehicle_model = self.bp_lip.filter(Vehicle_model)[0]
        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = no_render_mode
        self.world.apply_settings(self.settings)
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH =IMG_WIDTH
        self.SHOW_LOCAL_VIEW = show_local_view 
        self.camera_flag = None
        self.camera_observation = None
        self.sampling_resolution = sampling_resolution
        self.Global_Route_Planner = GlobalRoutePlanner(self.world.get_map(), self.sampling_resolution)
        self.radius = 2.5 #Gets Radius of the car
        ''''
        I had implemented 
        self.radius = (self.get_Edge_List()) #Gets Radius of the car
        but car has not been spawned yet: SO we can keep calling this at every reset but for now we can live with the hardcode value
        '''
        self.Debugger = Debugger 
        self.model = {'ld': model[0],
                      'wheel_base': model[1],
                      'raduis': self.radius}
        self.seconds_per_episode = seconds_per_ep
        self.reward_array = reward_arr #Based on MObile Path planning in Dynamic Environments through globally guided reinforcement learning
        self.episodes = 0
        

        self.action_space = Discrete(3) #Left, Up, right
        self.observation_space = Box(
            low=0, high=255, shape=[IMG_HEIGHT, IMG_WIDTH, 3], dtype=np.uint8)
        # np.uint8
        '''
        Need to do:
            1. Complete the Reset method, i.e. 
                a) The me

        
        
        '''

    def path_visualizor(self):
         for waypoint in self.route:
            self.world.debug.draw_string(waypoint[0].transform.location, '^', draw_shadow=False,
                color=carla.Color(r=0, g=0, b=255), life_time=5.0,
                persistent_lines=True)


    def reset(self, seed=None, options=None, start_point = None, end_point = None):
        '''
        Params:
            start_point: The vehicle transform at the start of the route
            end_point: The vehicle transform at the end of the route

        self.info contains list for velocities and locations for each episode
        '''
        
        
        try:
            self.destroy_all_actors()


        except:
            #print("Actors already destroyed")
            asad = 1
        
        self.info = {'velocity': [],
                     'location': []
                    }

        self.N = 0 #used in reward function
        


        self.episodes += 1

        self.collision = {'collision':False}  #check the collision sensor
        self.actor_lst = []
        if start_point == None:        
            self.start_transform = np.random.choice(self.spawn_points)
            # self.start_transform = self.spawn_points[0]
        else:
            self.start_transform = start_point    
        self.vehicle = self.world.try_spawn_actor(self.vehicle_model, self.start_transform)
        if  self.vehicle == None:
            raise Exception("Failed to spawn the vehicle")
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
        if end_point == None:
            self.end_point = random.choice(self.spawn_points)
        else:
            self.end_point = end_point

        self.route = self.Global_Route_Planner.trace_route(self.start_transform.location, self.end_point.location)
        self.waypoint_lst = []
        for waypoint, _ in self.route:
            loc = waypoint.transform.location
            self.waypoint_lst.append((loc.x,loc.y)) 
        
        ''' This is just to visualize the route
        Make this in
        '''
        self.path_visualizor()
        self.control.throttle = 0.5
        
        self.vehicle.apply_control(self.control)

        time.sleep(0.1)
        
        print(f"The environmnet has been reset and the episode is {self.episodes}")
        
        while self.camera_flag is None:
            time.sleep(0.01)
            #print('.',end='')

        self.episode_start_time = time.time()
        info = None
        #print(type(np.array(self.camera_observation))) 
        return self.camera_observation
    
    def get_vehicle_coordinates(self):
        vehicle_location = self.vehicle_transform.location
        x = vehicle_location.x
        y = vehicle_location.y
        z = vehicle_location.z
        return (x, y, z)

    def get_target_coordinates(self,transform):
        target = transform.location
        x = target.x
        y = target.y
        z = target.z
        return (x, y, z)
    
    def get_vehicle_velocity(self):        
        vel_array = self.vehicle.get_velocity() 
        x_dot, y_dot, z_dot = (vel_array.x, vel_array.y, vel_array.z)
        velocity = int(3.6 * math.sqrt(x_dot**2 + y_dot**2 + z_dot**2))  
        return velocity

    def get_target(self, action):
        
        x, y, z = self.get_vehicle_coordinates()
        # #print(action)
        if action == 0:
            # Go left
            transform = carla.Transform(self.vehicle.get_transform().transform(carla.Location(x=-self.sampling_resolution*10, y =self.sampling_resolution*5)))
            target_coor = self.get_target_coordinates(transform) 
            target = (target_coor[0], target_coor[1])
        elif action == 1:
            #  Go forward
            transform = carla.Transform(self.vehicle.get_transform().transform(carla.Location( y =self.sampling_resolution*5)))
            target_coor = self.get_target_coordinates(transform) 
            target = (target_coor[0], target_coor[1])
            # target = (x + self.sampling_resolution*10 , y + self.sampling_resolution*10)
        elif action == 2:
            # Go Right
            transform = carla.Transform(self.vehicle.get_transform().transform(carla.Location(x=-self.sampling_resolution*10, y =self.sampling_resolution*5)))
            target_coor = self.get_target_coordinates(transform) 
            target = (target_coor[0], target_coor[1])
            # target = (x , y + self.sampling_resolution*10)
        return target
    

    def get_steering(self, target, coordinates , yaw):
            tx, ty = target
            L = self.model['wheel_base']
            ld = self.model['ld']
            x, y, z = coordinates
            alpha = math.atan2((ty - y), (tx-x)) - yaw
            steering_angle = np.clip(math.atan2(2*L*math.sin(alpha),ld ),-1,1)
            
            return steering_angle
    
    def get_bounding(self):
        bb = self.vehicle.bounding_box.get_world_vertices(self.vehicle.get_transform())
        z_loc = [z.z for z in bb]
        s = [z == min(z_loc) for z in z_loc]
        enumerated_list = list(enumerate(z_loc))
        min_element = min(z_loc)
        min_indices = [index for (index, element) in enumerated_list if element == min_element]
        vert = list()
        for i in min_indices:
            vert.append((bb[i]))
        return vert

    def get_reward(self, On_global_flag, end_flag):
        # self.Reward_array = [-1, -100, 10, 1000, -50]
        if self.collision['collision'] == True:
            done = True
            reward  = self.reward_array[0] + self.reward_array[1]

        elif not On_global_flag: 
            self.N += 1
            done = False
            reward = self.reward_array[0]
        elif On_global_flag:
            done = False
            
            reward = self.reward_array[0] + (self.N+1) * self.reward_array[2]
            print("This condition is run", On_global_flag, reward)
            self.N = 0
        elif self.episode_start_time + self.seconds_per_episode < time.time(): #This means that the time of the episode has expired
            done = True
            reward = self.reward_array[-1]
        elif end_flag:
            reward =self.reward_array[3]
            done = True
        if self.Debugger:
            print(f"The reward is {reward} and done is{done}, and steps since g = {self.N}")


        # #print(f"the reward is = {reward}")
        return reward, done

    ##### Aux Variables to check if on global route:
    def get_edges(self,min_indices):
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        min_edges = []
        for edge in edges:
            i,j = tuple(edge)
            if i in min_indices and j in min_indices:
                min_edges.append((i,j))

        return min_edges

    def get_Edge_List(self):
        bb = self.vehicle.bounding_box.get_world_vertices(self.vehicle.get_transform())
        z_loc = [z.z for z in bb]
        enumerated_list = list(enumerate(z_loc))
        min_element = min(z_loc)
        min_indices = [index for (index, element) in enumerated_list if element == min_element]
        min_edges = self.get_edges(min_indices)
        Edge_lst = list()
        for i,j in min_edges:
            Edge_lst.append(((bb[i].x,bb[i].y),(bb[j].x,bb[j].y)))
        return Edge_lst
    
    def get_distance(self, p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        dist = np.linalg.norm(p1 - p2)
        return dist



    def car_radius(self,edge_lst):
        diameter = 0
        for p1,p2 in edge_lst:
            distance = self.get_distance(p1, p2)
            if distance >= diameter:
                diameter = distance
        return diameter/2

    def isInside(self,circle_x, circle_y, x, y):
     
        # Compare radius of circle
        # with distance of its center
        # from given point
        if ((x - circle_x) * (x - circle_x) +
            (y - circle_y) * (y - circle_y) <= self.radius * self.radius):
            return True
        else:
            return False


    def check_if_on_waypoint( self):
    
        circle_x, circle_y = self.vehicle_transform.location.x,self.vehicle_transform.location.y

        for (x,y) in self.waypoint_lst:
            global_waypoint = carla.Location(x = x, y = y)
            
            
            flag = self.isInside(circle_x, circle_y, global_waypoint.x, global_waypoint.y)
            if flag:
                self.world.debug.draw_string(global_waypoint, '0', draw_shadow=False,
                    color=carla.Color(r=255, g=0, b=0), life_time=5.0,
                    persistent_lines=True)
                break
        return flag    

    def check_if_end(self):
        circle_x, circle_y = self.vehicle_transform.location.x,self.vehicle_transform.location.y
        end_x = self.end_point.location.x
        end_y = self.end_point.location.y
        flag = self.isInside(circle_x, circle_y, end_x, end_y)

        return flag
    def step(self, action):

        '''
        self.vehicle_transform: is the vehicle transform at a given step
        '''
        On_global_flag = False #Flag to check if on global route
        On_end_point = False # Flag to check if the vehicle has reached end point

        self.vehicle_transform = self.vehicle.get_transform()
        target = self.get_target(action)
        self.velocity = self.get_vehicle_velocity()
        v_kmh = self.velocity


        coordinates = self.get_vehicle_coordinates()
        self.world.debug.draw_string(carla.Location(x = target[0], y = target[1]), '+', draw_shadow=False,
                    color=carla.Color(r=0, g=255, b=0), life_time=5.0,
                    persistent_lines=True)
        
        yaw = self.vehicle_transform.rotation.yaw
        yaw = np.radians(yaw)  ### Converting in radians
        steering_angle = self.get_steering( target, coordinates , yaw)
        
        ##### Now the control bit ####

        self.control.steer = steering_angle
        self.vehicle.apply_control(self.control)

        ##### Might Add a PID controller here to control velocity
        On_global_flag = self.check_if_on_waypoint()
        On_end_point = self.check_if_end()
        reward, done = self.get_reward(On_global_flag, On_end_point)

        self.info['velocity'].append(v_kmh)
        self.info['location'].append(coordinates)
        
        time.sleep(0.1)
        
      

        return self.camera_observation, reward, done,  self.info





    def render(self):
        pass

    def close(self):
        pass


    def collision_detector(self, event):
        self.collision['collision'] = True
        #print("A collision has occured")
    
        

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.IMG_HEIGHT, self.IMG_WIDTH, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_LOCAL_VIEW:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.camera_observation = i3
        # #print(np.shape(np.array(self.camera_observation)))
        if  self.camera_observation is None:
            self.camera_flag = False
            #print(f'This is to show that camera image hasn\'t loaded yet: {i3}')
        else:
            self.camera_flag = True
    

    def destroy_all_actors(self):
        for act in self.actor_lst:
            try: 
                act.destroy()
                #print(f"Actor {act} has been destroyed")
            except:
                #print(f'Actor {act} was not destroyed')
                pass
    
        
        
    
'''
ADD a destructor in which 
cv2.destroyAllWindows()
cv2.stop()


'''
          
        
if __name__ == '__main__':
    car =  CarENV()
    print(car)


