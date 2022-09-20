
	
#TO DO: 
    #fix vel

    #add images to saved things
import matplotlib.pyplot as plt
import cv2 
import numpy as np
import time 
import EasyPySpin
import pandas as pd
import csv
import socket
from RobotClass import Robot
from Coils import *
import struct
import pickle
from tqdm import tqdm


class Tracker:
    def __init__(self):
        self.alpha = 1000
        self.lower_thresh = 0
        self.upper_thresh = 150
        self.bounding_length = 50 
        self.resize_scale = 50
        
        self.Draw_Trajectory =False
        self.Robot_List = []
        self.Raw_Frames = []
        self.bot_loc = None
        self.target = None                                                                 
        self.node = None
        self.num_bots = 0                                                                   
        self.frame_num = 0       
        self.THREADSTATUS = True
        
        #variables for socket connection
        if SEND == 'y':
            try:
                self.HOST = "169.254.132.142"
                self.PORT = 5560
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.HOST,self.PORT))
            except Exception:
                print('error connecting to device...')

        if LISTEN == 'y':
            try:
                self.HOST = "192.168.1.180"
                self.PORT = 5560
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                self.sock.bind((self.HOST,self.PORT))
                self.sock.listen(5)
                self.socket_server, self.sock_name = self.sock.accept()
            except Exception:
                print("error setting up socket...")

            
           
     

            

    def mousePoints(self,event,x,y,flags,params):
       
        '''
        Mouse Callback. To run when the left mouse is clicked
        Initilize a new robot instance on each mouse click
        '''
        # Left button mouse click event opencv
        if event == cv2.EVENT_LBUTTONDOWN:
            CoilOn = False
            self.bot_loc = [x,y]
            
            x1 = int(x-self.bounding_length/2)
            y1 = int(y-self.bounding_length/2)
            w = self.bounding_length
            h = self.bounding_length
            
            initial_pos = [(x1+(x1+w))/2, (y1+(y1+h))/2]

            robot = Robot()
            robot.add_position(initial_pos)
            robot.add_Crop([x1,y1,w,h])
            self.Robot_List.append(robot)
            
            #add starting point of trajectory
            self.node = 0
            self.Robot_List[-1].add_trajectory(self.bot_loc)
            self.num_bots += 1
        
            
        elif event == cv2.EVENT_RBUTTONDOWN: 
            self.target = [x,y]
            #create trajectory
            self.Robot_List[-1].add_trajectory(self.target)
            self.Draw_Trajectory = True #Target Position

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.Draw_Trajectory == True:
                self.target = [x,y]
                self.Robot_List[-1].add_trajectory(self.target)
                
        elif event == cv2.EVENT_RBUTTONUP:        
            self.Draw_Trajectory = False

        elif event == cv2.EVENT_MBUTTONDOWN:
            #CLEAR EVERYTHING AND RESTART ANALYSIS
            del self.Robot_List[:]
            self.num_bots = 0
            self.node = 0
            self.alpha = 1000

    def main_thread(self): 

        '''
        connect to camera and pperform real time tracking and analysis of MB
        '''
        #global self.BFIELD                                                                
        cam = EasyPySpin.VideoCapture(0)
        #cam = cv2.VideoCapture('/home/max/Desktop/microrobots/LOOKHERE/mickyroll1.mp4')        
        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cam.get(cv2.CAP_PROP_FPS)
        print(width,height,fps)
        
                                                                        #how fast to update frames
        area_threshold = 0                                                            #minimum area of coutours to neglext
        scale_dia = 4                                                                  #96 pixels = 600 um mulitple averge diameter by this value. depends on bot and frame size
        elapsed_time = 0
        #950 um = 1464 height in pixels
        
        #%%
        start = time.time()
        prev_frame_time = 0
        new_frame_time = 0
        fps_list = []
        size = (int(width*self.resize_scale/100),int(height*self.resize_scale/100))
        if len(FILENAME) >1:
            result = cv2.VideoWriter(FILENAME+'.avi', cv2.VideoWriter_fourcc(*'MJPG'),fps,size)
        
        obj = 10 #x
        pix2metric = 950/1464 #pix/micron
        
        while True:
            success,frame = cam.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame is None:
                print("Game Over")
                break
            frame = cv2.resize(frame, (int(width*self.resize_scale/100),int(height*self.resize_scale/100)),interpolation = cv2.INTER_AREA)
            
            self.frame_num += 1
            cv2.setMouseCallback("im", self.mousePoints)


        #DETECT ROBOT    
        #%%    
            '''
            -for each robot defined Crop frame around it based on initial left mouse click position
            -apply mask and find contours 
            -from contours draw a bounding box around the contours
            -find the centroid of the bounding box and use this as the robots current position
            '''
            if self.num_bots > 0:    #for each defined robot update stuff
                for bot in range(len(self.Robot_List)):
                

                    #crop the frame based on initial ROI dimensions
                    x1,y1,x2,y2 = self.Robot_List[bot].Cropped_frame[-1]
                    if x1 < 0:
                        x1 = 0
                    if y1 < 0:
                        y1 = 0
                    cropped_frame = frame[y1:y1+y2, x1:x1+x2]
                    
                    
                    #apply mask
                    crop_mask = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                    crop_mask = cv2.GaussianBlur(crop_mask, (21,21), 0)
                    crop_mask = cv2.inRange(crop_mask, self.lower_thresh,self.upper_thresh)
                    
                    
                    #find contours and areas of contours 
                    contours,_ = cv2.findContours(crop_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    area_list = []
                    for cnt in contours:
                        #remove small elements by calcualting arrea
                        area = cv2.contourArea(cnt)
                        
                        if area > area_threshold:# and area < 3000:# and area < 2000: #pixels
                            area_list.append(area)
                            
                            x,y,w,h = cv2.boundingRect(cnt)
                            current_pos = [(x+x+w)/2, (y+y+h)/2]
                        
                            #cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),1)
                            #cv2.rectangle(cropped_frame, (x,y), (x+w,y+h),(255,0,0),1)
                            #frame= cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                            cv2.drawContours(cropped_frame,[cnt], -1,(0,255,255),1)# -1: draw all
                            
                    
                    #interesting condition for when a robot splits
                    if len(area_list) > 1:
                        print("bot split")
        #TRACK AND UPDATE ROBOT           
        #%%           
                    '''
                    -if contours were found from the previous step calculate the area of the contours and append it to the Robot class.
                    -update global average running list of areas 
                    -based on the current position calculate above, adjust cropped area dimensions for next frame
                    -update Robot class with new cropped dimension, position, velocity, and area
                    ''' 
                    #compute filters
                    
                    if area_list:
                        
                        #cacluate and analyze contours areas
                        avg_area = sum(area_list)/len(area_list)
                        self.Robot_List[bot].add_area(avg_area)
                        avg_global_area = sum(self.Robot_List[bot].Area_List)/len(self.Robot_List[bot].Area_List)
                        self.Robot_List[bot].set_Avg_Area(avg_global_area)
                        #update cropped region based off new position and average area
                        avg_dia = np.sqrt(avg_global_area/np.pi)*scale_dia#scaled 1.5 times for wiggle room
                        x1new = x1+current_pos[0]-avg_dia/2
                        y1new = y1+current_pos[1]-avg_dia/2
                        new_crop =  [int(x1new), int(y1new), int(avg_dia), int(avg_dia)]
                        
                        #calculate velocity based on last position and fps
                        if len(self.Robot_List[bot].Position_List) > 5:
                            velx = (current_pos[0]+x1 - self.Robot_List[bot].Position_List[-1][0])  * fps * (pix2metric)
                            vely = (current_pos[1]+y1 - self.Robot_List[bot].Position_List[-1][1])  * fps * (pix2metric)
                            vmag = np.sqrt(velx**2+vely**2)
                            vel = [velx,vely,vmag]
                            self.Robot_List[bot].add_velocity(vel)
            
                        
                        #update robots params
                        self.Robot_List[bot].add_Crop(new_crop)
                        self.Robot_List[bot].add_position([current_pos[0]+x1, current_pos[1]+y1]) 
                        self.Robot_List[bot].add_frame(self.frame_num)
                       
                        #display
                        cv2.circle(cropped_frame,(int(current_pos[0]),int(current_pos[1])),2,(1,255,1),-1)
                  
                    
                    #if when i click there is no detected area: delete the most recent robot instance
                    else:
                        del self.Robot_List[-1]
                        self.num_bots -=1 
                        #i also want to recent everything and clear the screen
                        

        #CONTROL LOOP          
        #%%

                '''
                -define target position
                -display target position
                -if target position is defined, look at most recently clicked bot and display its trajectory
                
                        basically I want to move the robot to each node in the trajectory array
                        if the error is less than a certain amount move on to the next note
                '''
                if self.num_bots > 0: #necesary condition for some reason
                    if len(self.Robot_List[-1].Trajectory) > 1:
                        if self.node == len(self.Robot_List[-1].Trajectory):
                            print("arrived")
                            self.alpha = 1000 #can be any number not 0 - 2pi. indicates stop outpting current
                             #define first node in traj array
                            targetx = self.Robot_List[-1].Trajectory[self.node][0]
                            targety = self.Robot_List[-1].Trajectory[self.node][1]
                            
                            #calcualte bots position
                            robotx = self.Robot_List[-1].Position_List[-1][0]   #choose the last bot that was pressed for now
                            roboty = self.Robot_List[-1].Position_List[-1][1]
                            
                            error = np.sqrt((targetx - robotx)**2 + (targety - roboty)**2)

                            
                        else:
                            #non-linear closed loop                
                            #display trajectory
                            pts = np.array(self.Robot_List[-1].Trajectory , np.int32)
                            cv2.polylines(frame, [pts], False, (1,1,255), 1)
                            
                            #define first node in traj array
                            targetx = self.Robot_List[-1].Trajectory[self.node][0]
                            targety = self.Robot_List[-1].Trajectory[self.node][1]
                            
                            #calcualte bots position
                            robotx = self.Robot_List[-1].Position_List[-1][0]   #choose the last bot that was pressed for now
                            roboty = self.Robot_List[-1].Position_List[-1][1]
                            
                            #calcualte error
                            cv2.arrowedLine(frame, (int(robotx), int(roboty)),(int(targetx),int(targety)),[0,0,0],3)
                            error = np.sqrt((targetx - robotx)**2 + (targety - roboty)**2)
                            
                            if error < 10:                
                                new_targetx = self.Robot_List[-1].Trajectory[self.node][0]
                                new_targety = self.Robot_List[-1].Trajectory[self.node][1]
                                direction_vec = [new_targetx - robotx, new_targety - roboty]
                                self.alpha = np.arctan2(direction_vec[1],direction_vec[0]) 

                                self.Robot_List[-1].add_track(self.frame_num, error, [robotx,roboty], [new_targetx,new_targety],self.alpha)
                                
                                self.node +=1
                                
                            else:
                                direction_vec = [targetx - robotx, targety - roboty]
                                self.alpha = np.arctan2(direction_vec[1],direction_vec[0]) 
                                self.Robot_List[-1].add_track(self.frame_num, error, [robotx,roboty], [targetx,targety],self.alpha)
                                
                            #output coil strength from above
            
                            
                            #output direction to socket
                            
                        if SEND == 'y':
                            #outputs a string, need to ensure its same length each time.
                            #on rasppi, I need to fix edge cases
                            message = round(self.alpha, 3)
                            message = struct.pack('!d', message)
                            self.sock.send(message)

                        if LISTEN == 'y':
                            self.socket_server.recv(8)
                            alpha = struct.unpack('!d', message)
                            print(alpha)
                            self.Robot_List[-1].add_track(self.frame_num, error, [robotx,roboty], [targetx,targety],alpha)
 
                    
        #DISPLAY 
        #%%
                '''
                -display dragon tails and other HUD graphics
                '''
                
                color = plt.cm.rainbow(np.linspace(0, 1, self.num_bots))*255                       
                for bot,c in zip(range(self.num_bots),color):
                    #display dragon tails                
                    pts = np.array(self.Robot_List[bot].Position_List, np.int32)
                    cv2.polylines(frame, [pts], False, c, 1)
                    
                    if len(self.Robot_List[bot].Velocity_List) > 10:
                        vmag = np.array(self.Robot_List[bot].Velocity_List[-10:])[:,2]
                        vmag_avg = sum(vmag)/len(vmag)
                        cv2.putText(frame,str(bot+1)+' : '+str(int(vmag_avg)), (0,150+bot*20), cv2.FONT_HERSHEY_COMPLEX,.5,c,1)
                
                
                #crop_mask = cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2BGR)
                #Verti = np.concatenate((cropped_frame, crop_mask), axis=0)  
                #cv2.imshow("im"+str(bot), Verti)
                
                #compute FPS and scale bar
            new_frame_time = time.time()
            elapsed_time = (new_frame_time-prev_frame_time)
            fps = 1/elapsed_time
            fps_list.append(fps)
            prev_frame_time = new_frame_time
            if len(fps_list) > 10:
                avg_fps = sum(fps_list[-10:-1])/len(fps_list[-10:-1])
            else:
                avg_fps = sum(fps_list)/len(fps_list)
            cv2.putText(frame,str(int(avg_fps)), (75,50), cv2.FONT_HERSHEY_COMPLEX,.9,(0,0,255),1)
            #scale bar
            cv2.line(frame, (75,80), (int(100*(1/pix2metric)),80),(0,0,255),3)
            #add videos a seperate list to save space and write the video afterwords
        
            cv2.imshow("im", frame)
            if len(FILENAME) > 1:
                pass
                #result.write(frame)
                #self.Raw_Frames.append(frame)
               
            #Exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        
        
        if SEND == 'y':
            self.sock.close() #close socket connection
        if len(FILENAME) > 1:
            result.release()   
        cam.release()
        cv2.destroyAllWindows()
     
    
       

            
    
    def convert2pickle(self, filename,rolling_frequency):
        Pickles = []
        print(" --- writing robots ---")
        for i in tqdm(self.Robot_List):
            Pickles.append(i.as_dict())
        
        print(" --- writing frames ---")
        cap = cv2.VideoCapture(filename+".avi")
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw = []
        for i in tqdm(range(length)):
            success,frame = cap.read()
            raw.append(frame)
        Pickles.append(raw)
        Pickles.append(rolling_frequency)
        print(" -- writing pickle --")
        with open(filename+".pickle", "wb") as handle:
            pickle.dump(Pickles, handle, protocol = pickle.HIGHEST_PROTOCOL)
        print(" -- DONE -- ")    


if __name__ == "__main__":
    SEND = input("SEND?")   # y if outputing signals to handheld device
    LISTEN = input("LISTEN?") #y if listening to alpha value
    FILENAME =input("FILENAME: ")   #"experiment1_closedloop1"  # not none if want to save session
    
    Tracking = Tracker()
    Tracking.main_thread()
    if len(FILENAME) >1:
        rolling_frequency = input("Enter the rolling frequency (HZ): ")
        Tracking.convert2pickle(FILENAME,rolling_frequency)
    
 
 
   


