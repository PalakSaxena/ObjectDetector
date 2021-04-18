
"""
    Task 1 : OBJECT DETECTOR completed by PALAK SAXENA
"""

import cv2

''' If the model is confident up to the given confidence threshold value only then it will detect
    otherwise it will ignore that object.
'''

Thresh = 0.45      # Confidence Threshold value to detect the object

# VideoCapture() will start the camera. 0 value is used for the Inbuilt WebCam.
cap = cv2.VideoCapture(0)
# Parameters to set Window Size
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

# Empty list to store class names from coco file
classNames = []

classFile = 'coco.names'

# Reading class names from the coco file
with open(classFile, 'rt') as f:
    classNames = f.read().strip('\n').split('\n')

# Files having configurations and trained weights of the object classes
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

''' OpenCV’s dnn (Deep Neural Network) module is used to load a pre-trained object detection network.
    This will enable us to pass input images through the network and obtain the output bounding box
    (x, y)-coordinates of each object in the image.
'''
net = cv2.dnn_DetectionModel(weightsPath, configPath)
# Set input size for frame with parameters width and height
net.setInputSize(320, 320)
# Set scale factor value for frame
net.setInputScale(1.0/127.5)
# Set mean value for frame
net.setInputMean((127.5, 127.5, 127.5))
# Set flag swapRB for frame. swapRB	Flag which indicates that swap first and last channels.
net.setInputSwapRB(True)

while True:
    success, img = cap.read()

    # This will provide us the ClassIds(for coco names), confidence(possibility value), bounding boxes for our objects
    classIds, confs, bbox = net.detect(img, confThreshold=Thresh)
    print(classIds, bbox)

    if len(classIds) != 0:
        # Instead of using three for loops we are using zip function
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            # This will put Class Names as text
            cv2.putText(img, classNames[classId - 1].upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            # This will put Confidence(Possibility value) as text
            cv2.putText(img, str(round(confidence * 100, 2)) + '%', (box[0] + 250, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('MyFrame', img)
    # This wil break the loop whenever esc is pressed
    if cv2.waitKey(10) & 0xFF == 27:
        break
