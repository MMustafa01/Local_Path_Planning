import gymnasium as gym
import sys
import math 
import random 
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import carla
from gymnasium.spaces import Discrete, Box
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
SECONDS_PER_EP = 200
SAMPLING_RESOLUTION = 1
PARAM = [6.3, 2.875] # [ld: look ahead distance, wheel based] 
REWARD_ARR = [-10, -1000, 100, 1000, -1200] # [normal, coll, back on global, time up]
DT = 0.1 #Fixed time step

START_END_DIC = {7:[8,9,15,16,19,20], 
                 6:[8,9,15,16,19,20],
                 121: [12,13,14,11],
                 13: [6,7,8,9,11,12,15,16,20],
                 14: [6,7,8,9,11,12,15,16,20]
                 }





'''
Things to do:
1. Improve the start and end dictionary
2. Make action space box, and use that for steering values, also throttle values
3. Try to make the observation space bigger
4. Implement a video saving method


'''

class CarENV(gym.Env):
    
    def __init__(self, Vehicle_model = VEHICLE_MODEL, host = HOST, port = PORT,no_render_mode = NO_RENDER_MODE, time_out  = TIMEOUT, IMG_HEIGHT = IMG_HEIGHT , IMG_WIDTH =IMG_WIDTH, show_local_view = SHOW_LOCAL_VIEW, sampling_resolution = SAMPLING_RESOLUTION, model = PARAM, seconds_per_ep = SECONDS_PER_EP, reward_arr = REWARD_ARR, Debugger = True, start_end_dic = START_END_DIC, dt = DT, action_space_type = 'Discrete'):
        super().__init__()
        self.host = host
        self.port = port
        self.time_out = time_out

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



        #  Fixed time step
        self.dt = dt

        ####### CARLA SERVER SETTING ######################
        '''
        Implementing synchronous mode and no rendering mode
        '''
        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = no_render_mode
        
        self.settings.fixed_delta_seconds = self.dt
        
        self.world.apply_settings(self.settings)
        

        # Recording the time of total steps and reseting step
        self.reset_step = 0
        self.time_step = 0
        self.total_steps = 0
  


        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH =IMG_WIDTH
        self.SHOW_LOCAL_VIEW = show_local_view 
        self.camera_flag = None
        self.camera_observation = None
        self.sampling_resolution = sampling_resolution
        self.Global_Route_Planner = GlobalRoutePlanner(self.world.get_map(), self.sampling_resolution)
        self.radius = 2.25 #Gets Radius of the car
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


        self.total_reward = 0
        self.steps = 0
        self.N = 0 #used in reward function
        
        self.action_space_type = action_space_type
 
        if self.action_space_type == 'Discrete':
            self.action_space = Discrete(5) #Left, Up, right, idle,Follow_pure
        elif self.action_space_type == 'Box':
            self.action_space = Box(low = np.array([-1,0]), high=np.array([2,1]), dtype=np.float32) #[steer, throttle] if steer > 1.0 then PP
        

        
        self.observation_space = Box(
            low=0, high=255, shape=[IMG_HEIGHT, IMG_WIDTH, 3], dtype=np.uint8)

        self.start_end_dic = start_end_dic

        self.action_lst = []

    def path_visualizor(self):
         for waypoint in self.route:
            self.world.debug.draw_string(waypoint[0].transform.location, '^', draw_shadow=False,
                color=carla.Color(r=0, g=0, b=255), life_time=5,
                persistent_lines=True)



    def get_start_end_transform(self):
        '''
        start_end_dic = {7:[8,9,15,16,19,20], 
                 6:[8,9,15,16,19,20],
                 121: [12,13,14,11],
                 13: [6,7,8,9,11,12,15,16,20],
                 14: [6,7,8,9,11,12,15,16,20]
                 }
        '''

        start_transform_idx = random.choice(list(self.start_end_dic.keys()))
        end_transform_idx = random.choice(self.start_end_dic[start_transform_idx])

        start_transform = self.spawn_points[start_transform_idx]
        end_transform = self.spawn_points[end_transform_idx]

        return start_transform, end_transform
    


    def reset(self, seed=None, options=None):
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
       
        
        self.visited_global_waypoint = []

        self.episodes += 1

        self.collision_hist = {'event' :None} #Just to print the collision event
        self.collision = {'collision':False}  #check the collision sensor

        self.lane_change = {'Solid':False, 'Broken':False}
        self.actor_lst = []
        self.remaining_time = 0

        try:
            print(self.action_lst)
        finally:
            self.action_lst = []
    
        self.start_transform, self.end_point =  self.get_start_end_transform()
        self.vehicle = self.world.try_spawn_actor(self.vehicle_model, self.start_transform)
        if  self.vehicle == None:
            print("Failed to spawn the vehicle")
            print("Connecting to the server once again")
            try: 
                #print("Please wait as we attempt to connect to the CARLA server.")
                self.client = carla.Client(self.host, self.port)        
                self.client.set_timeout(self.time_out)
                #print("You have successfully connected to the CARLA server.")
            except:
                raise Exception("Sorry you were unable to connect to the CARLA server. Check if you have a CARLA server running.")
            self.world = self.client.get_world()
            self.bp_lip = self.world.get_blueprint_library()
            self.spawn_points = self.world.get_map().get_spawn_points() 
            # self.vehicle_model = self.bp_lip.filter(Vehicle_model)[0]
            self.vehicle = self.world.try_spawn_actor(self.vehicle_model, self.start_transform)
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
        self.Camera_bp.set_attribute('sensor_tick', f'{self.dt}')
        
        cam_transform = carla.Transform(self.vehicle.get_transform().transform(carla.Location(x=-5,z=10)), carla.Rotation(yaw=90, pitch=-90))
        transform = carla.Transform(carla.Location(x = 2.5, z = 0.7), carla.Rotation(pitch = 0))
        

        self.Camera  =self.world.try_spawn_actor(self.Camera_bp, cam_transform, attach_to= self.vehicle)
        self.actor_lst.append(self.Camera)
        
        ###### Spawn Collision Sensor ######## 
        collision_Sensor_bp = self.bp_lip.find("sensor.other.collision")

        
        self.collision_Sensor = self.world.try_spawn_actor(collision_Sensor_bp, transform,  attach_to =  self.vehicle)
        self.actor_lst.append(self.collision_Sensor)
        
        ###### Implementing a lane invasion sensor ######
        lane_inv_bp = self.bp_lip.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.try_spawn_actor(lane_inv_bp, carla.Transform(), attach_to = self.vehicle)

        
        spectator = self.world.get_spectator() 
        transform = carla.Transform(self.vehicle.get_transform().transform(carla.Location(x=-5,z=20)), carla.Rotation(yaw=90, pitch=-90)) 
        spectator.set_transform(transform)
        

        ''' set sensor recording'''
        self.Camera.listen(lambda data:self.process_img(data))
        self.collision_Sensor.listen(lambda event: self.collision_detector(event))
        self.lane_sensor.listen(lambda event: self.on_invasion(event))


        '''Apparently this makes the spawning quicker'''
        self.control = carla.VehicleControl()
        self.control.throttle = 0.0
        self.control.steer = 0.0
        self.vehicle.apply_control(self.control)
        
        ############ Tracing a Global Route ############
        

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

        # time.sleep(0.1)
        
        print(f"The environmnet has been reset. The total reward was {self.total_reward}. And steps were {self.time_step} and the episode is {self.reset_step} and the total_steps are {self.total_steps}")
        self.total_reward = 0
        self.steps = 0
        self.N = 0 #used in reward function

        
        self.reset_step += 1
        self.time_step = 0
        


        while self.camera_flag is None:
            # time.sleep(0.01)
            pass
            #print('.',end='')

        self.episode_start_time = time.time()
        info = None
        #print(type(np.array(self.camera_observation))) 
        return self.camera_observation, self.info
    
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

    def get_action(self, action):
        """If the PP reaches the final global way_point"""
        Done = False

        x, y, z = self.get_vehicle_coordinates()

        if self.action_space_type == 'Discrete':
            if action == 0:
                # Go left
                self.control.steer = -1
                self.control.throttle = 0.5
            elif action == 1:
                self.control.steer = 0
                self.control.throttle = 0.5
            elif action == 2:
                self.control.steer = 1
                self.control.throttle = 0.5
            elif action == 3:
                ## Follow pure pursuit
                self.control.throttle = 0.5
                _, global_target, info = self.get_global_target()
                yaw = self.vehicle_transform.rotation.yaw
                steering = self.get_steering(global_target, (x,y,z), yaw )
                self.control.steer = steering
                global_target_way = carla.Location(x= global_target[0], y= global_target[1])
                self.world.debug.draw_string(global_target_way, '0', draw_shadow=False,
                            color=carla.Color(r=0, g=255, b=0), life_time=2.5,
                            persistent_lines=True)
                if info:
                    Done = True
            elif action == 4: #idle
                self.control.throttle = 0
                self.control.steer = 0

        elif self.action_space_type == 'Box': #[steer, throttle] if steer > 1.0 then PP
            self.control.throttle = action[1] # steering
            if action[0] <= 1:
                self.control.steer = action[0]
            else:
                ## Follow pure pursuit
                _, global_target, info = self.get_global_target()
                yaw = self.vehicle_transform.rotation.yaw
                steering = self.get_steering(global_target, (x,y,z), yaw )
                self.control.steer = steering
                global_target_way = carla.Location(x= global_target[0], y= global_target[1])
                self.world.debug.draw_string(global_target_way, '0', draw_shadow=False,
                            color=carla.Color(r=0, g=255, b=0), life_time=2.5,
                            persistent_lines=True)
                if info:
                    Done = True
        print(f"the action is [steering, throttle] = {action}")
        self.vehicle.apply_control(self.control)
        if self.Debugger == True:
            print(f'The action chosen was {action}', end = '||')
        return Done
        # return target
    

    def get_steering(self, target, coordinates , yaw):
            tx, ty = target
            L = self.model['wheel_base']
            ld = self.model['ld']
            x, y, z = coordinates
            alpha = math.atan2((ty - y), (tx-x)) - yaw
            steering_angle = np.clip(math.atan2(2*L*math.sin(alpha),ld ),-1,1)
            
            return steering_angle
    
    
    def get_global_target(self):
        ''' 
        Try to implement next target index using this method
        http://paulbourke.net/geometry/pointlineplane/

        Implementation given here
        '''
        waypoint_list = self.waypoint_lst
        x,y,z = self.get_vehicle_coordinates()
        print(f'the last location in the list {waypoint_list[-1]}')
        info = False
        exception = False
        dxl, dyl = [], []
        for i in range(len(waypoint_list)):
            dx = abs(x - waypoint_list[i][0])
            dxl.append(dx) #list of dx1
            dy = abs(y - waypoint_list[i][1])
            dyl.append(dy)

        dist = np.hypot(dxl, dyl)
        try:
            idx = np.argmin(dist) + 4 
        except: idx = 4
        # take closest waypoint, else last wp

        if idx < len(waypoint_list):
            print(f'This line is printed')
            tx = waypoint_list[idx][0]
            ty = waypoint_list[idx][1]

        else:
            print(f'the exception is printed')
            tx = waypoint_list[-1][0]
            ty = waypoint_list[-1][1]
            exception = True
        print(f'The current waypoint is {(tx, ty)}')

        if (tx, ty) == waypoint_list[-1] and exception == False:
            info = True
        return idx, (tx, ty), info



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

    def get_done_event(self, end_flag):
        # done \in {2 -> coll, 3 -> lanechange,1 ->time up, 0 -> not done }
        if self.time_step >self.seconds_per_episode: #This means that the time of the episode has expired
            done = 1
        elif self.collision['collision'] == True:
            done = 2
        elif self.lane_change['Solid'] == True:
            done = 3
        elif end_flag:
            done = 4
        else:
            done = 0

        return done
    def get_reward(self, On_global_flag, end_flag):
        # self.Reward_array = [-1, -100, 10, 1000, -50]
        # REWARD_ARR = [-1, -1000, 100, 1000, -1200] # [normal, coll, back on global, time up]
        done_dic = {1:"Time up", 2: "collision", 3: "off road",4:"end_flag" ,0: "Not done"}
        done_num = self.get_done_event(end_flag=end_flag)
        if done_num:
            done = True
            if done_num == 1: #This means that the time of the episode has expired
                print(f"Done condition: max time steps reached")
                reward = self.reward_array[-1]


            elif done_num == 2 or done_num == 3:
                reward  = self.reward_array[0] + self.reward_array[1]


            elif end_flag:
                reward =self.reward_array[3]
            print(f"Done condition: {done_dic[done_num]}")
        
        else:
            done = False
            if not On_global_flag:
                if self.lane_change['Broken'] == True: 
                    reward = self.reward_array[0] * 10
                elif self.lane_change['Broken'] == False:
                    reward = self.reward_array[0]
                self.N += 1


            elif On_global_flag:
                reward = self.reward_array[0] + (self.N+1) * self.reward_array[2]
                self.N = 0
        self.total_reward += reward
        if self.Debugger:
            print(f"The reward is {reward} and done is{done}, and steps since g = {self.N}")
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
            Edge_lst.append(((bb[i].x,bb[i].y,bb[i].z ),(bb[j].x,bb[j].y,bb[i].z)))
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
                if not (x,y) in self.visited_global_waypoint: #This is done to prevent loops
                    if self.Debugger:
                        print("First time visit")
                    self.visited_global_waypoint.append((x,y))
                    self.world.debug.draw_string(global_waypoint, '0', draw_shadow=False,
                        color=carla.Color(r=255, g=0, b=0), life_time=2.5,
                        persistent_lines=True)
                    break
                else:
                    if self.Debugger:
                        print("Sorry you have visited this waypoint before", end = ' ||')
        return flag    

    def check_if_end(self):
        circle_x, circle_y = self.vehicle_transform.location.x,self.vehicle_transform.location.y
        end_x = self.end_point.location.x
        end_y = self.end_point.location.y
        flag = self.isInside(circle_x, circle_y, end_x, end_y)

        return flag
    
    def step(self, action):

        # if self.time_step%10:
        #     print(action, '||||', end = ' , ')
        '''
        self.vehicle_transform: is the vehicle transform at a given step
        '''
        self.action_lst.append(action)
        if not (self.total_steps + 1)%2048:
            print("Skiping this step")
            
            
        
        self.steps += 1
        On_global_flag = False #Flag to check if on global route
        On_end_point = False # Flag to check if the vehicle has reached end point

        self.vehicle_transform = self.vehicle.get_transform()

        # print(f"The time: \n {self.episode_start_time + self.seconds_per_episode} and {time.time()-self.episode_start_time}")
        ####### The action function performs the action themselves ######
        check_cond =self.get_action(action)
        # print(f"The action is {action}")

        ##### Getting data for later ######
        self.velocity = self.get_vehicle_velocity()
        v_kmh = self.velocity


        coordinates = self.get_vehicle_coordinates()
        
        ##### Might Add a PID controller here to control velocity
        On_global_flag = self.check_if_on_waypoint()
        On_end_point = self.check_if_end()
        if check_cond:
            On_end_point = True
        reward, done = self.get_reward(On_global_flag, On_end_point)

        self.info['velocity'].append(v_kmh)
        self.info['location'].append(coordinates)
        
        self.world.tick()
    

        self.time_step += 1
        self.total_steps += 1 

        # time.sleep(0.1)
        truncated = None
        return self.camera_observation, reward,truncated, done,  self.info

    def get_global_target(self):
        ''' 
        Try to implement next target index using this method
        http://paulbourke.net/geometry/pointlineplane/

        Implementation given here
        '''
        waypoint_list = self.waypoint_lst
        x,y,z = self.get_vehicle_coordinates()
        info = False
        exception = False
        dxl, dyl = [], []
        for i in range(len(waypoint_list)):
            dx = abs(x - waypoint_list[i][0])
            dxl.append(dx) #list of dx1
            dy = abs(y - waypoint_list[i][1])
            dyl.append(dy)

        dist = np.hypot(dxl, dyl)
        try:
            idx = np.argmin(dist) + 4 
        except: idx = 4
        # take closest waypoint, else last wp

        if idx < len(waypoint_list):
            tx = waypoint_list[idx][0]
            ty = waypoint_list[idx][1]

        else:
            tx = waypoint_list[-1][0]
            ty = waypoint_list[-1][1]
            exception = True
        # print(f'The current waypoint is {(tx, ty)}')

        if (tx, ty) == waypoint_list[-1] and exception == False:
            info = True
        return idx, (tx, ty), info



    def render(self):
        pass

    def close(self):
        pass

######### Sensor Handler

    def collision_detector(self, event):
        self.collision['collision'] = True
        self.collision_hist['event'] = event
        if self.Debugger:
            print(f"{self.collision_hist['event']}")
        #print("A collision has occured")
    
        
    def on_invasion(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        for i in range(len(text)):
            if 'Solid' in str(text[i]): # SolidBroken and BrokenSolid are classified as  as solid here
                self.lane_change['Solid'] = True
            elif 'Broken' in str(text[i]) :
                self.lane_change['Broken'] = True
            else:
                print(f'lane type = {text} ')
            if self.Debugger:
                print()
                print('Crossed line %s' % ' and '.join(text))
                print()


    def process_img(self, image):

        ''''
        Get projection matrix
        Get Edge_lst

        
        '''
        i = np.array(image.raw_data)
        i2 = i.reshape((self.IMG_HEIGHT, self.IMG_WIDTH, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_LOCAL_VIEW:
            cv2.imwrite(f'images\image_{str(self.total_steps)}.jpg', i3[:,:,:3])
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


