#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import cv2

from torchvision import transforms
from posenet.posenet_resnet50_ import PoseNet
from yolo_qr import QRdecoder
from PIL import Image
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Machine is',device)
model_path = 'best_model/best_model_130.pt'

class Pathplanner():
    def __init__(self):
        
        # Map info
        self.resolution = 0.05000000074505806
        self.width = 608
        self.height = 384
        self.origin_x = -10
        self.origin_y = -10
        self.map = np.load('./data/2gong_after.npy')

        # Target point
        self.destination = None
        self.goal_x = None
        self.goal_y = None
        self.map_goal_xy = None

        # Current point
        self.my_x = None
        self.my_y = None

        # Path planning
        self.Kp_att = 1
        self.Kp_rel = 30
        self.obstacle_bound = 50
        self.path = None

        # junction location on map
        self.map_junction = [323, 180]

        # QR decoder
        self.qrdecoder = QRdecoder()
        
    def convert_pose_to_map(self, x, y):
        x_map = int( (x-self.origin_x) / self.resolution)
        y_map = int( (y-self.origin_y) / self.resolution)
        return [x_map, y_map]
    
    def convert_map_to_pose(self, x, y):
        x_pose = x*self.resolution + self.origin_x
        y_pose = y*self.resolution + self.origin_y
        return [x_pose, y_pose]
    
    def set_goal(self):
        self.destination = input("Where are you go? :")
        
        if self.destination == "toilet":
            self.goal_x = 6.5
            self.goal_y = -3
            print("toilet : ({0}, {1})".format(self.goal_x, self.goal_y))
        # else:
        #     print("none")
            
    def visualizer(self):
        pyplot.imshow(self.map)
        pyplot.show()
        pyplot.pause(0.001)
    
    def draw_current_pose2map(self, pose):
        self.map[pose[1], pose[0]] = 255
        

    def planning(self):
        
        print("target point : ", self.goal_x, self.goal_y)
        self.visualizer()
        self.map_goal_xy = self.convert_pose_to_map(self.goal_x, self.goal_y)
        self.map[self.map_goal_xy[1], self.map_goal_xy[0]] = 50

        self.my_x, self.my_y = 2.1, -0.9


        obs_idx = np.array(np.where(self.map == 100)).T
        
        self.path = self.Artificial_potential_field(self.my_x, self.my_y, obs_idx)
      
        for i in range(len(self.path[0])):
            x = self.path[0][i]
            y = self.path[1][i]
            self.map[y, x] = 100
        
        self.visualizer()

        

    def att_force(self, x, y):
        e_x, e_y = self.map_goal_xy[0]-x, self.map_goal_xy[1]-y
        distance = np.linalg.norm([e_x, e_y])

        att_x = self.Kp_att * e_x / distance
        att_y = self.Kp_att * e_y / distance

        return att_x, att_y
    

    def rep_force(self, x, y, obs):

        rep_x, rep_y = 0, 0
        #print(max(obs[:,0]), max(obs[:,1]))
        for obs_xy in np.ndindex(obs.shape[0]):
            #print("obs , ",obs[obs_xy][0], obs[obs_xy][1], x, y)
            obs_dis_x, obs_dis_y = obs[obs_xy][1]-x, obs[obs_xy][0]-y
            obs_dis = np.linalg.norm([obs_dis_x, obs_dis_y])

            if obs_dis < self.obstacle_bound:
                rep_x = rep_x - self.Kp_rel * (1/obs_dis - 1/self.obstacle_bound)*(1/(obs_dis*obs_dis))*obs_dis_x/obs_dis
                rep_y = rep_y - self.Kp_rel * (1/obs_dis - 1/self.obstacle_bound)*(1/(obs_dis*obs_dis))*obs_dis_y/obs_dis
            else:
                rep_x = rep_x
                rep_y = rep_y
        return rep_x, rep_y
    

    def Artificial_potential_field(self, start_x, start_y, obs):
        max_error = None
        [x, y] = self.convert_pose_to_map(start_x, start_y)
        trace_x = []
        trace_y = []

        trace_x.append(int(x))
        trace_y.append(int(y))

        while(1):
            att_x, att_y = self.att_force(x, y)
            
            rep_x, rep_y = self.rep_force(x, y, obs)

            #print(rep_x, rep_y)
            pot_x = att_x + rep_x
            pot_y = att_y + rep_y

            x = x + pot_x
            y = y + pot_y
            #print("cur x",x,"cur y",y, "target:", self.map_goal_xy[0], self.map_goal_xy[1])
            trace_x.append(int(x))
            trace_y.append(int(y))

            error = np.linalg.norm([self.map_goal_xy[0]-x, self.map_goal_xy[1]-y])
            if max_error is None:
                max_error = error
            progress_status = ( np.abs(max_error - error) / max_error ) * 100
            #print("progress status : {:.2f}%".format(progress_status)) 
            sys.stdout.write("\r progress status : [%-20s] %d%%" % ('='*int(progress_status//5), progress_status))
            sys.stdout.flush()
            if error < 1:
                print("\nPath planning end")
                return [trace_x, trace_y]
                break

    def gstreamer_pipeline(self,
        capture_width=1280,
        capture_height=720,
        display_width=1280,
        display_height=720,
        framerate=60,
        flip_method=0,
    ):
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
        )


    # QR code detection and initialize position
    def initialize(self):
        cap = cv2.VideoCapture(self.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        if cap.isOpened():
            while True:
                ret, frame = cap.read()
                #print("r")
                if ret is False:
                    continue
                else:
                    cv2.imshow('fre', frame)
                    cv2.waitKey(1)

                    # frame을 수정해서 qr을 인식하도록 함
                    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
                    ln = self.qrdecoder.net.getLayerNames()
                    ln = [ln[i-1] for i in self.qrdecoder.net.getUnconnectedOutLayers()]
                    
                    self.qrdecoder.net.setInput(blob)
                    outs = self.qrdecoder.net.forward(ln)
                    self.qrdecoder.postprocess(frame,outs)
                    if self.qrdecoder.x is not None and self.qrdecoder.y is not None:
                        self.my_x, self.my_y = self.qrdecoder.x, self.qrdecoder.y
                        print("initialized your position : ", self.my_x, self.my_y)
                        break

    def show_camera(self):
        # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
        print(self.gstreamer_pipeline(flip_method=0))
        cap = cv2.VideoCapture(self.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        if cap.isOpened():
            window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
            # Window
            while cv2.getWindowProperty("CSI Camera", 0) >= 0:
                ret_val, img = cap.read()
                cv2.imshow("CSI Camera", img)
                # This also acts as
                keyCode = cv2.waitKey(30) & 0xFF
                # Stop the program on the ESC key
                if keyCode == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            print("Unable to open camera")

    def navigate(self, map_pose):
        dist_to_junction = np.linalg.norm(np.array(self.map_junction) - np.array(map_pose))
        print(dist_to_junction)
        if dist_to_junction < 10:
            dx, dy = map_pose[0] - self.map_goal_xy[1], map_pose[1] - self.map_goal_xy[0]
            angle = np.arctan2(dy, dx)
            print(dx, dy, angle)
            if angle > np.pi /4:
                print("turn left")
            elif angle < np.pi / 4:
                print("turn right")
        else:
            print("Go straight")

                



    def localize(self):
        print("localization started")
        cap = cv2.VideoCapture(self.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        
        mean=[0.49, 0.486, 0.482] 
        std=[0.197, 0.189, 0.187]   
        #tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean, std)])
        tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean, std)])
        # Test cap
        #cap = cv2.VideoCapture(0)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

        # model init
        model = PoseNet()
        model.to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # localization loop
        while cap.isOpened():
            ret_val, cv_frame = cap.read()
            cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
            # cv2.imshow('cv', cv_frame)
            # cv2.waitKey(1)
            frame = Image.fromarray(cv_frame)
            frame = tf(frame).to(device)
            
            # posenet inference

            pose = model(frame.unsqueeze(0))
            pose = pose.cpu().detach().numpy()
            self.my_x, self.my_y = pose[0][0][0], pose[0][0][1]

            print("current location : {:.2f}, {:.2f}".format(self.my_x, self.my_y))


            pose_map = self.convert_pose_to_map(self.my_x, self.my_y)
            print("pose in map : {0}, {1}".format(pose_map[0], pose_map[1]))
            self.navigate(pose_map)

            self.draw_current_pose2map(pose_map)
            self.visualizer()


if __name__ == '__main__':

    planner = Pathplanner()
    pyplot.ion()
    planner.set_goal()
    #planner.initialize()
    planner.planning()
    planner.localize()




