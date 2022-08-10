import cv2
import numpy as np
from app_10_utils import proccess_frame,draw_prediction

confidence=0.3
nms=0.4
with open(r'C:\Users\Waseem K\Desktop\Intership\Day-3\Assig-10 cars detection\coco.names') as f:
    classes = f.read().rstrip('\n').split('\n')

net=cv2.dnn.readNet(r'C:\Users\Waseem K\Desktop\Intership\Day-3\Assig-10 cars detection\yolov3.weights',r'C:\Users\Waseem K\Desktop\Intership\Day-3\Assig-10 cars detection\yolov3.cfg','darknet')
layernames=net.getUnconnectedOutLayersNames()
cam=cv2.VideoCapture(r'C:\Users\Waseem K\Desktop\Intership\Day-3\Assig-10 cars detection\images.jpg')
while True:
    status,image=cam.read()
    if status:
        blob=cv2.dnn.blobFromImage(image,1/255.0,(416,416),swapRB=True,crop=False)
        net.setInput(blob)
        outs=net.forward(layernames)
        proccess_frame(image,outs,classes,confidence,nms)
        cv2.imshow("result",image)
        cv2.waitKey(1)