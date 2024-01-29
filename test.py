# General
import time
from queue import Queue, Full
import threading
import cv2
import torch
import heapq
import Jetson.GPIO as GPIO
import numpy as np
import matplotlib.pyplot as plt
import queue

qr_map = np.load('qr_map.npy')

input_queue = Queue(maxsize=1)
output_queue = Queue(maxsize=1)
display_queue = Queue(maxsize=1)


GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
BUZZER_PIN = 35

BUTTON1_PIN = 33
BUTTON2_PIN = 31
BUTTON3_PIN = 29
BUTTON4_PIN = 23
GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(BUTTON1_PIN, GPIO.IN)
GPIO.setup(BUTTON2_PIN, GPIO.IN)
GPIO.setup(BUTTON3_PIN, GPIO.IN)
GPIO.setup(BUTTON4_PIN, GPIO.IN)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'----------{device}----------')

qcd = cv2.QRCodeDetector() 

neighbors = {
    (13, 9): {(20, 9): 1}, # ICML
    (20, 9): {(13, 9): 1, (20, 17): 1}, # ICML hallway
    (20, 17): {(20, 9): 1, (20, 20): 1},  # hallway_01
    (20, 20): {(20, 17): 1, (20, 23): 1}, # hallway_02
    (20, 23): {(20, 20): 1, (20, 27): 1}, # hallway_03
    (20, 27): {(20, 23): 1, (20, 33): 1, (1,  27): 1, (34, 27): 1},  # cross intersection
    (1, 27): {(20, 27): 1}, # eng_building_3 exit
    (34, 27): {(20, 27): 1}, # eng_building_1 exit
    (20, 33): {(20, 27): 1, (20, 37): 1}, # toilet_woman
    (20, 37): {(20, 33): 1, (30, 36): 1}, # toilet_man
    (30, 36): {(20, 37): 1}, # stairway
}



# class NoDuplicatesQueue(queue.Queue):
#     def put(self, item, block=True, timeout=None):
#         if item not in self.queue:
#             super().put(item, block, timeout)
    



def destination_select():    
    print('Waiting ... Please select destination')
    while True:
        if GPIO.input(BUTTON1_PIN) == 0:
            end = (1, 27) # eng_building_3_exit
            print('Destination is ENG_building 3')
            break
        elif GPIO.input(BUTTON2_PIN) == 0:
            end = (34, 27) # eng_building_1_exit
            print('Destination is ENG_building 1')
            break
        elif GPIO.input(BUTTON3_PIN) == 0:
            end = (20, 37) # toilt_man
            print('Destination is Man toilet')
            break
    qr_map[end[0]-1, end[1]-1] = 50
    return end

#---Path planning func---##########################################################################
def a_star_algorithm(start, end, neighbors, heuristic):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {start: None}
    g_score = {node: float('inf') for node in neighbors}
    g_score[start] = 0
    f_score = {node: float('inf') for node in neighbors}
    f_score[start] = heuristic(start, end)

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == end:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for neighbor in neighbors[current]:
            tentative_g_score = g_score[current] + neighbors[current][neighbor]
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

def heuristic(node, end):
    # 이 함수는 휴리스틱 함수를 정의합니다. 실제 사용 시에는 문제에 맞는 휴리스틱을 구현해야 합니다.
    # 여기서는 간단한 예시로 맨해튼 거리를 사용합니다.
    return abs(node[0] - end[0]) + abs(node[1] - end[1])

def update_path(current_position, path, end, neighbors, heuristic):
    if current_position in path:
        path = path[path.index(current_position)+1 : ]
        dir = (current_position[0] - path[0][0], current_position[1] - path[0][1]) # Direction to next node
    
    else: # if your currnet position is not in path, find new path
        path = a_star_algorithm(current_position, end, neighbors, heuristic)
        path = path[path.index(current_position)+1 : ]
        dir = (current_position[0] - path[0][0], current_position[1] - path[0][1]) # Direction to next node

    return path, dir

#---Path planning func---##########################################################################


def gstreamer_pipeline(
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



###################################################################################################
def capture_thread(input_queue, display_queue):
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    while True:
        ret, image = cap.read()
        if ret:
            try:
                input_queue.put(image, block=False)
                display_queue.put(image, block=False)
            except Full:
                pass
        if cv2.waitKey(1)==27:
            cap.release()
            break
###################################################################################################
def localization_thread(input_queue, output_queue):
    is_first = True
    end = destination_select()
    pose = []
    while True:

        # Get current position from QR_Code image
        frame = input_queue.get()
        retval, decoded_info, _0, _1 = qcd.detectAndDecodeMulti(frame) # decoded_info is location
        # print(retval)
        
        if len(decoded_info) == 1:
            if len(decoded_info[0])!=0:
                current_position = tuple(map(int, decoded_info[0].split(',')))
               
                print('Current_Location :',current_position)
                qr_map[current_position[0]-1, current_position[1]-1] = 150
                # Planning path
                if is_first & retval:
                    prev_position = current_position
                    path = a_star_algorithm(current_position, end, neighbors, heuristic)
                    is_first = False
                    path, dir = update_path(current_position, path, end, neighbors, heuristic)

                elif current_position == end:
                    print('-----------Arrived at destination!-----------')
                    dir = (1000,1000)
                    output_queue.put(dir)
                    
                    is_first = True
                    end = destination_select() # new destination
                    continue

                else: 
                    qr_map[prev_position[0]-1, prev_position[1]-1] = 0
                    qr_map[current_position[0]-1, current_position[1]-1] = 150
                    path, dir = update_path(current_position, path, end, neighbors, heuristic)

                    
                    prev_position = current_position

                output_queue.put(dir)

def turn_right():
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    time.sleep(0.2)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

def turn_left():
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    time.sleep(0.2)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    time.sleep(0.2)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

def go_straight():
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(2)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

def arrival():
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    time.sleep(0.1)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    time.sleep(0.1)

        # direction = 100
        # output_queue.put(direction)
###################################################################################################
def output_thread(output_queue):
    prev_dir = (0,0)
    # GPIO (buzzer)
    while True:
        # dir : (x,y) current-next
        dir = output_queue.get()
        #print("dir : ", dir)
        if prev_dir == dir:
            continue
        else:
            
            if dir[1] == 0:
                if dir[0] == -7: # right / 2 times 
                    print('Go straight')
                    go_straight()
                    
                elif dir[0] < 0:
                    print("Turn right and go straight")
                    turn_right()
                elif dir[0] > 0:
                    print("Turn left and go straight")
                    turn_left()


            else:
                if dir[1] == -8:
                    print("Turn left and go straight")
                    turn_left()
                elif dir[1] == -10:
                    print("Go staright")
                    go_straight()
                elif dir[1] == 1000:
                    print("------------Arrived at destination!-------")
                    arrival()

                else:
                    print("Go straight")
                    go_straight()
                    
                
            
   
                



        prev_dir = dir 
                
      
###################################################################################################
def display_thread(display_queue): # main thread
    if False:
        while True:
            qr_img = display_queue.get()
            cv2.imshow('QR', qr_img)
            if cv2.waitKey(1) & 0xFF == 27: # if you want to quit, Press 'ESC'
                break
    if True:
        while True:
            plt.imshow(qr_map)
            plt.pause(1)
###################################################################################################
capture_t = threading.Thread(target=capture_thread, args=(input_queue, display_queue))
localization_t = threading.Thread(target=localization_thread, args=(input_queue, output_queue))
output_t = threading.Thread(target=output_thread, args=(output_queue,))

capture_t.start()
localization_t.start()
output_t.start()

display_thread(display_queue)