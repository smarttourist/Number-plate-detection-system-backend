
import argparse
import cv2
import numpy as np
from keras.models import Sequential, load_model
from segmentImage import segmentImage
width=608
height=608
confThreshold = 0.5
nmsThreshold = 0.4

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
args = parser.parse_args()
outputFile = args.image[:-4] + '_yolo_out_py.jpg'
outputPlate = args.image[:-4] + '_plate.jpg'
classesFile = "classes.names"

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

predConf="darknet-yolov3.cfg"
weights='lapi.weights'

model = load_model('cnn_classifier.h5')


predictor = cv2.dnn.readNetFromDarknet(predConf, weights)
predictor.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
predictor.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

cap = cv2.VideoCapture(args.image)

def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    #    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
    plate = img[top:bottom, left:right]
    #cv2.imshow("CROPPED",plate)
    predResults=segmentImage(plate)
    cv2.imwrite(outputPlate, plate.astype(np.uint8))
    print(predResults)
    return predResults
    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    '''
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(img, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv2.FILLED)
    #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
    cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
    '''
def process(img,res):
    frameHeight=img.shape[0]
    frameWidth=img.shape[1]
    classIds = []
    confidences = []
    boxes = []
    for out in res:
        #print("out.shape : ", out.shape)
        for detection in out:
            #if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            #if scores[classId]>confThreshold:
            confidence = scores[classId]
            #print(detection)
            #if detection[4]>confThreshold:
                #print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
                #print(detection)
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))

                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:

        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left-4, top-4, left + width+4, top + height+4)


def OutputNames(predictor):
    # Get the names of all the layers in the network
    layersNames = predictor.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in predictor.getUnconnectedOutLayers()]


ret, img= cap.read()
while(True):
    if (ret):

        blob = cv2.dnn.blobFromImage(img, 1/255, (width, height), [0,0,0], 1, crop=False)
        predictor.setInput(blob)

        res=predictor.forward(OutputNames(predictor))
        process(img,res)
        cv2.imshow("IMG", img)
        if (args.image):
            cv2.imwrite(outputFile, img.astype(np.uint8))

        k = cv2.waitKey(0)
        if (k == 27):
            cv2.destroyAllWindows()
            break

