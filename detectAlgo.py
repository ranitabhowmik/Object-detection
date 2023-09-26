import numpy as np
import argparse
import time
import cv2
import os
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
               help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
               help="minimum probability to filter weak detections, IoU threshold")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
               help="threshold when applying non maxima suppression")
args = vars(ap.parse_args())
labelsPath = 'yolo-coco\\coco.names'
LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0,255, size = (len(LABELS), 3),
                           dtype="uint8")
weightsPath = 'yolo-coco\\yolov3.weights'
configPath = 'yolo-coco\\yolov3.cfg'

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416,416),
                             swapRB=True, crop=False)
net.setInput(blob)
layerOutputs = net.forward(ln)

boxes = []
confidences = []
classIDs = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > args["confidence"]:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width/2))
            y = int(centerY - (height/2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                         args["threshold"])
if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, color, 2)
# Resize the output window to match the image dimensions
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", W, H)

cv2.imshow("image", image)
cv2.waitKey(0)
#end