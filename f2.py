from flask import Flask,request,abort,jsonify
from datetime import datetime
import uuid
import cv2
import numpy as np
from PIL import Image
import base64
import os , io , sys
app=Flask(__name__)
users =[]
chat=[]
messages=dict()

UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1280 * 1280

@app.route('/')
def hello():
    return 'hello\n'


@app.route("/send",methods=['POST'])
def send():
    default_name = '0'
    f = request.files['upload']
    f.save(f.filename)
   # print(file)
    '''
    npimg = np.fromstring(f, np)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)


    ######### Do preprocessing here ################
    # img[img > 150] = 0
    ## any random stuff do here
    ################################################
    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)

    #img_base64 = base64.b64encode(rawBytes.read())
    #return jsonify({'status': str(img_base64)})
    '''
    id=str(uuid.uuid4())
    messages[id]={
        'timestamp':datetime.now(),
        'id':id,
    }
    chat.append(id)

    return jsonify({'id':id})



if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')