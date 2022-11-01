#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:51:31 2022

@author: bizzarohd
"""

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from pySerialTransfer import pySerialTransfer as txfer




'''
signal output is now compeltely indpendent of frame rate
I just send an actions list to arduino mega which has 12 PWM outputs 
'''

control_params = {
            "lower_thresh": 0,
            "upper_thresh": 100,
            "bounding_length" : 10,
            "area_filter": 3,
            "field_strength": 1,
            "rolling_frequency": 10.0,
            "gamma": 90
            } 
camera_params = {
            "resize_scale": 50,
            "framerate": 60,
            "exposure": 5000
            } 


class Robot:
    '''
    Robot class to store and ID all new robots
    '''
    def __init__(self):
        self.Position_List = []
        self.Area_List = []
        self.Cropped_frame = []
        self.Avg_Area = 0
        self.Time = []
        self.Frequency = []
        self.Alpha_List = []
    
    def add_area(self,Area):
        self.Area_List.append(Area)
        
    def add_position(self, Position):
        self.Position_List.append(Position)
    
    def add_frame(self,Frame):
        self.Frame_List.append(Frame)
        
    def add_Crop(self,Crop):
        self.Cropped_frame.append(Crop)
        
    def set_Avg_Area(self,Avg_Area):
        self.Avg_Area = Avg_Area
        
    def add_time(self,time):
        self.Time.append(time)
        
    def add_freq(self,f):
        self.Frequency.append(f)
        
    def add_alphas(self, alph):
        self.Alpha_List.append(alph)





class Experiment():
    def __init__(self):
        self.Robot_List = []
        self.num_bots = 0                                                                   
        self.frame_num = 0     


    def mousePoints(self,event,x,y,flags,params):
    
        '''
        Mouse Callback. To run when the left mouse is clicked
        Initilize a new robot instance on each mouse click
        '''
        # Left button mouse click event opencv
        if event == cv2.EVENT_LBUTTONDOWN:
            
            x1 = int(x-control_params["bounding_length"]/2)
            y1 = int(y-control_params["bounding_length"]/2)
            w = control_params["bounding_length"]
            h = control_params["bounding_length"]
            
            robot = Robot()
            robot.add_Crop([x1,y1,w,h])
            
            self.Robot_List.append(robot)
            self.num_bots += 1
            
    def Send(self,arduino,alpha,freq,typ):
        message = arduino.tx_obj([float(alpha),float(freq),float(typ)]) #float(0) => Rolling
        arduino.send(message)
        print("sent")
        
        

    def Tracker(self, arduino, actions): 

        '''
        connect to camera and pperform real time tracking and analysis of MR
        '''                                                           
        #cam = EasyPySpin.VideoCapture(0)
        cam = cv2.VideoCapture("/Users/bizzarohd/Desktop/UpdateOctober/mickyroll1.mp4")        
        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))


        cv2.namedWindow("im")

        while True:
            #!!! Step 1: read the frame and adjust it
            success,frame = cam.read()
            
            resize_scale = camera_params["resize_scale"]
            cam.set(cv2.CAP_PROP_EXPOSURE,camera_params["exposure"])
            frame = cv2.resize(frame, (int(width*resize_scale/100),int(height*resize_scale/100)),interpolation = cv2.INTER_AREA)
           
            self.frame_num += 1
            cv2.setMouseCallback("im", self.mousePoints)
            
            
            if self.num_bots > 0:    #for each defined robot, update stuff
            
                #!!! Step 2: recieve action commands, either from joystick or exterinally
                if actions is not None:
                    #read input arrays
                    timestamp = self.frame_num
                    if timestamp < len(actions[0]):
                        
                        alpha   = actions[0][timestamp]
                        freq    = actions[1][timestamp]
                        typ     = actions[2]
                        print("sent")
                    else:
                        print("-- End of Trajectory --")
                 
                else: 
                    #read joystick...
                    alpha = np.random.randint(-10,10) #control direction
                    freq = np.random.randint(0,20)  # control speed
                    typ = 0
               
                
                
                #!!! Step 3: send those action commands to arduino
                self.Send(arduino,alpha,freq,typ)
                
                
                #!!! Step 4: detect,track, and update robot parameters
                for bot in range(len(self.Robot_List)):
              
                    x1,y1,x2,y2 = self.Robot_List[bot].Cropped_frame[-1]
                    
                    x1 = max(min(x1,width),0)
                    y1 = max(min(y1,height),0)
                    cropped_frame = frame[y1:y1+y2, x1:x1+x2]
                    
                    
                    #carry out mask and thresholding on GPU
                    #gpu_frame = cv2.cuda_GpuMat()
                    #gpu_frame.upload(cropped_frame)

                    #gpu_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                    crop_mask = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                    crop_mask = cv2.GaussianBlur(crop_mask, (21,21), 0)
                    crop_mask = cv2.inRange(crop_mask, (control_params["lower_thresh"]),(control_params["upper_thresh"]))
                    #ret,gpu_frame = cv2.cuda.threshold(gpu_frame, control_params["upper_thresh"],255,cv2.THRESH_BINARY)
                    #gpu_frame = cv2.cuda.bitwise_not(gpu_frame)
                    #crop_mask = gpu_frame.download()
                    
                    #find contours and areas of contours 
                    contours,_ = cv2.findContours(crop_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    area_threshold_lower = self.Robot_List[bot].Avg_Area /control_params["area_filter"]
                    area_list = []
                    w_list = [] #creating these lists to store all conoutrs because I only want the first w,h not the most recent one
                    h_list = []
                    for cnt in contours:
                        #remove small elements by calcualting arrea
                        area = cv2.contourArea(cnt)
                        
                        if area > area_threshold_lower:# and area < 3000:# and area < 2000: #pixels
                            area_list.append(area)
                            
                            x,y,w,h = cv2.boundingRect(cnt)
                            current_pos = [(x+x+w)/2, (y+y+h)/2]
                            w_list.append(w)
                            h_list.append(h)

                            cv2.rectangle(cropped_frame, (x,y), (x+w,y+h),(255,0,0),1)
                            cv2.drawContours(cropped_frame,[cnt], -1,(0,255,255),1)# -1: draw all
                            
                    
                    if area_list:
                        
                        #cacluate and analyze contours areas
                        avg_area = sum(area_list)/len(area_list)
                        self.Robot_List[bot].add_area(avg_area)
                        
                        avg_global_area = sum(self.Robot_List[bot].Area_List)/len(self.Robot_List[bot].Area_List)
                        self.Robot_List[bot].set_Avg_Area(avg_global_area)
                        #update cropped region based off new position and average area
                        
                        x1new = x1+current_pos[0]-max(w_list)
                        y1new = y1+current_pos[1]-max(h_list)
                        x2new = 2*max(w_list)
                        y2new = 2*max(h_list)
                        new_crop =  [int(x1new), int(y1new), int(x2new), int(y2new)]
                        
                        #update robots params
                        self.Robot_List[bot].add_Crop(new_crop)
                        self.Robot_List[bot].add_position([current_pos[0]+x1, current_pos[1]+y1])     
                        self.Robot_List[bot].add_time(time.time())
                        self.Robot_List[bot].add_freq(freq)
                        self.Robot_List[bot].add_alphas(alpha)
  
    
                color = plt.cm.rainbow(np.linspace(0, 1, self.num_bots))*255                       
                for bot,c in zip(range(self.num_bots),color):
                    #display dragon tails                
                    pts = np.array(self.Robot_List[bot].Position_List, np.int32)
                    cv2.polylines(frame, [pts], False, c, 1)
                    
            
            
            #!!! Step 5: display the frames
            cv2.imshow("im", frame)
            k = cv2.waitKey(1000)
            if k == ord("q"):
                break
        
        #close coils
        message = arduino.tx_obj([float(0),float(0),float(4)]) #float(4) => Close
        arduino.send(message)
        
        #close camera
        cam.release()
        cv2.destroyAllWindows()



def run_exp(actions):
    '''
    press left mouse button on robot to detect
    press q to exit the window
    
    Parameters:
        actions : if None ==  learn via joystick
                  else: pass actions list [alpha,freq,type].right now it just indexs through the list by frame number
        aim : 1 = LEARN, 2 = RL
    Returns: (X,Y,alpha,time,freq) if bots were detected
             None if else

    '''
    #connect to arduino
    arduino  = txfer.SerialTransfer('/dev/cu.usbserial-210')
    arduino.open()
    Exp = Experiment() #create an experiement
    Exp.Tracker(arduino,actions) #run the tracker
    
    if len(Exp.Robot_List) > 0:
        MyRobot = Exp.Robot_List[-1]  # only use last robot in list of clicked on robots
    
        X = np.array(MyRobot.Position_List)[:,0]
        Y = np.array(MyRobot.Position_List)[:,1]
        alpha = np.array(MyRobot.Alpha_List)
        time= np.array(MyRobot.Time)
        freq= np.array(MyRobot.Frequency)
        print("-- robies detected --")
        return X,Y,alpha,time,freq
    else:
        print("-- no robies --")
        return None

    arduino.close()

    return X,Y,alpha,time,freq

if __name__ == '__main__':
    #actions
    a = np.arange(0,360,1) #alpha
    b = [] 
    for i in range(0,360):
        b.append(5)
    c = 0

    actions = None#[a,b,c]

        
    X,Y,alpha,time,freq = run_exp(actions) 

