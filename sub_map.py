#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import cv2

from posenet.posenet_resnet50_ import PoseNet
from yolo_qr import QRdecoder

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
        self.Kp_att = 0.5
        self.Kp_rel = 30
        self.obstacle_bound = 50

        # QR decoder
        self.qrdecoder = QRdecoder()
        

    def set_goal(self):
        self.destination = input("Where are you go? :")
        self.goal_x = 6.5
        self.goal_y = -3
        # if self.destination == "toilet":
        #     self.goal_x = 6.5
        #     self.goal_y = -3
        # else:
        #     print("none")
            
        
    def planning(self):

        print("target point : ", self.goal_x, self.goal_y)

        self.map_goal_xy = [int((self.goal_x - self.origin_x) / self.resolution), int((self.goal_y - self.origin_y) / self.resolution)] 
        self.map[self.map_goal_xy[1], self.map_goal_xy[0]] = 200

        self.my_x, self.my_y = 2.1, -0.9


        obs_idx = np.array(np.where(self.map == 100)).T
        
        path = self.Artificial_potential_field(self.my_x, self.my_y, obs_idx)
      
        for i in range(len(path[0])):
            x = path[0][i]
            y = path[1][i]
            self.map[y, x] = 255  

        pyplot.imshow(self.map)
        pyplot.show()

    def att_force(self, x, y):
        e_x, e_y = self.map_goal_xy[0]-x, self.map_goal_xy[1]-y
        distance = np.linalg.norm([e_x, e_y])

        att_x = self.Kp_att * e_x / distance
        att_y = self.Kp_att * e_y / distance

        return att_x, att_y
    

    def rep_force(self, x, y, obs):

        rep_x, rep_y = 0, 0
        print(max(obs[:,0]), max(obs[:,1]))
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

        x, y = (start_x - self.origin_x)/self.resolution, (start_y-self.origin_y)/self.resolution

        trace_x = []
        trace_y = []

        trace_x.append(int(x))
        trace_y.append(int(y))

        while(1):
            att_x, att_y = self.att_force(x, y)
            
            rep_x, rep_y = self.rep_force(x, y, obs)

            print(rep_x, rep_y)
            pot_x = att_x + rep_x
            pot_y = att_y + rep_y

            x = x + pot_x
            y = y + pot_y
            print("cur x",x,"cur y",y, "target:", self.map_goal_xy[0], self.map_goal_xy[1])
            trace_x.append(int(x))
            trace_y.append(int(y))

            error = np.linalg.norm([self.map_goal_xy[0]-x, self.map_goal_xy[1]-y])
            print("err :", error)
            if error < 1:
                print("Path planning end")
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
                print("r")
                if ret is False:
                    continue
                else:
                    cv2.imshow('fre', frame)
                    cv2.waitKey(10)

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
    
    def localize(self):
        print("localization started")
        cap = cv2.VideoCapture(self.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

        # model init
        model = PoseNet()
        model.to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # localization loop
        while cap.isOpened():
            ret_val, img = cap.read()
            
            # posenet inference

            pose = model(img)


if __name__ == '__main__':

    planner = Pathplanner()
    # planner.set_goal()
    planner.initialize()
    # planner.planning()
    planner.localize()




