import cv2
import numpy as np

class QRdecoder():
    def __init__(self):
        self.qcd = cv2.QRCodeDetector()
        self.classes = open('./data/qrcode.names').read().strip().split('\n')
        self.net = cv2.dnn.readNetFromDarknet('./data/qrcode-yolov3-tiny.cfg', './data/qrcode-yolov3-tiny.weights')
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.threshold = 0.6
        self.x = None
        self.y = None

    def postprocess(self, frame, outs):
        frameHeight, frameWidth = frame.shape[:2]
        classIds = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.threshold:
                    x, y, width, height = detection[:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
                    left = int(x - width / 2)
                    top = int(y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, int(width), int(height)])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.threshold, self.threshold - 0.1)
        for i in indices:
            i = i
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cropped_image = frame[top:top + height, left:left + width]
            cv2.imshow('cropped', cropped_image)
            #cv.imwrite('cropped.jpg', cropped_image)
            try:
                retval, decoded_info, points, straight_qrcode = self.qcd.detectAndDecodeMulti(cropped_image)
                #print("retval:{0}, dec:{1}".format(retval, decoded_info))
                if retval==True:
                    xy = str(decoded_info)
                    replaces = ["(", "'", " "]

                    for rep in replaces:
                        xy = xy.replace(rep, '')
                    #print(xy)
                    split = xy.split(',')
                    x = int(split[0])
                    y = int(split[1])
                    print("x : {0}, y : {1}".format(x, y))
                    self.x, self.y = x, y
                    break
            except:
                print("Can't decode QR")