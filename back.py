from datetime import datetime
import uuid
import cv2
import numpy as np

from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from flask import Flask,request,abort,jsonify

width=608
height=608
confThreshold = 0.5
nmsThreshold = 0.4

predConf="darknet-yolov3.cfg"
weights='lapi.weights'
model = load_model('cnn_classifier.h5')
img=[]
kernal=np.ones((1,1),np.uint8)

predictor = cv2.dnn.readNetFromDarknet(predConf, weights)
predictor.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
predictor.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
classesFile = "classes.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

##############################
app=Flask(__name__)
users =[]
chat=[]
messages=dict()
UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1280 * 1280
####1
def recognise(cropped):
    d = np.reshape(cropped, (1, 28, 28, 1))
    out = model.predict(d)
    # Get max pre arg
    p = []
    precision = 0
    for i in range(len(out)):
        z = np.zeros(36)
        z[np.argmax(out[i])] = 1.
        precision = max(out[i])
        p.append(z)
    prediction = np.array(p)

    # Inverse one hot encoding
    alphabets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    classes = []
    for a in alphabets:
        classes.append([a])
    ohe = OneHotEncoder(handle_unknown='ignore', categorical_features=None)
    ohe.fit(classes)
    pred = ohe.inverse_transform(prediction)

    if precision > 0.6:
        print('Prediction : ' + str(pred[0][0]) + ' , high : ' + str(precision))
        return pred[0][0]
    else:
        print(('Prediction : ' + str(pred[0][0]) + ' , Low : ' + str(precision)))
###2
def segmentImage(platel):
    resStr=''

    plate=cv2.cvtColor(platel, cv2.COLOR_BGR2GRAY)
    plate = cv2.bilateralFilter(plate, 5, 15, 15)  #21515
    height=plate.shape[0]
    width=plate.shape[1]
    area = height * width
    #print(height,width)
    scale1 = 0.01
    scale2 = 0.1
    area_condition1 = area * scale1
    area_condition2 = area * scale2




    #cv2.imshow(" Plate", plate)
    ret3, th3 = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(hierarchy)
    except Exception as e:
        print(e)

    contours = sorted(contours, key=cv2.boundingRect, reverse=False)
    results = []

    i=0
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)


        if (w * h > area_condition1 and w * h < area_condition2 and w / h > 0.3 and h / w > 0.3):
            #cv2.drawContours(plate, [cnt], 0, (0, 255, 0), 1)

            #cv2.rectangle(platel, (x, y), (x + w, y + h), (255, 0, 0), 1)
            c = th3[y:y + h, x:x + w]
            c = np.array(c)
            c = cv2.bitwise_not(c)
            #c = square(c)
            c = cv2.resize(c, (28, 28), interpolation=cv2.INTER_AREA)
            #cv2.imshow("c", c)
            #cv2.imshow(str(i), c)

            result= recognise(c)


            if (result != None):
                results.append(result)

            i+=1
    i=0
    for i in results:
        resStr += str(i)
    #print(resStr)
    return resStr


def OutputNames(predictor):
    # Get the names of all the layers in the network
    layersNames = predictor.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in predictor.getUnconnectedOutLayers()]
###3
def process(img,res):

    frameHeight=img.shape[0]
    frameWidth=img.shape[1]

    classIds = []
    confidences = []
    boxes = []
    proRes=[]
    for out in res:

        for detection in out:

            #if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            #if scores[classId]>confThreshold:
            confidence = scores[classId]
            #if detection[4]>confThreshold:
                #print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
                #print(detection)
            #print(scores)
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                #print([left, top, width, height])
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    #print(indices)
    for i in indices:

        i = i[0]
        box = boxes[i]
        left = box[0]-4
        top = box[1]-4
        width = box[2]
        height = box[3]
        #classIds[i], confidences[i],

        right=left + width+4
        bottom= top + height+4
        plate = img[top:bottom, left:right]
        predResults = segmentImage(plate)
        proRes.append(predResults)
    return proRes

@app.route('/')
def hello():
    return 'hello\n'
'''
    cap = cv2.VideoCapture('image.jpg')
    ret, img = cap.read()
    if (ret):
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (width, height), [0, 0, 0], 1, crop=False) #presprocess
        predictor.setInput(blob)

        res = predictor.forward(OutputNames(predictor))
        #print(res)
        final= process(img, res)
        #print(final)
    return str(final)
'''






@app.route("/send",methods=['POST'])
def send():
    default_name = '0'
    f = request.files['upload'].read()
    #f.save(f.filename)
    npimg = np.frombuffer(f, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    #cap = cv2.VideoCapture('image.jpg')
    #ret, img = cap.read()
    ret=1
    '''
    cv2.imshow('im',img)
    #cv2.waitKey()
    k = cv2.waitKey(0)
    if (k == 27):
        cv2.destroyAllWindows()
    '''

    if (ret):
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (width, height), [0, 0, 0], 1, crop=False) #presprocess
        predictor.setInput(blob)

        res = predictor.forward(OutputNames(predictor))
        #print(res)
        final= process(img, res)
        #print(final)
    #return str(final)
    id=str(uuid.uuid4())
    '''
    messages[id]={
        'timestamp':datetime.now(),
        'id':id,
    }
    chat.append(id)
    '''
    return jsonify(id=id,number=final)

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/sh')
def sh():
    shutdown_server()
    return 'Server shutting down...'

if __name__=='__main__':
    app.run(debug=False,host='0.0.0.0',threaded=False)

