import os 
from PIL import Image
from flask import Flask,request
import json
from flask_cors import CORS
import base64
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

allClasses = ['Bird', 'Flower', 'Hand', 'House', 'Mug', 'Pencil', 'Spoon', 'Sun', 'Tree', 'Umbrella']
ort_session = ort.InferenceSession('./model.onnx')
def process(path):
    image = Image.fromarray(plt.imread(path)[:, :, 3])
    image = image.resize((64, 64))
    image = (np.array(image)>0).astype(np.float32)
    image = Image.fromarray(image)
    image = np.array(image)[None, :, :]
    return image[None]

def test(path):
    image = process(path)
    output = ort_session.run(None,{'data': image})[0].argmax()
    print (allClasses[output],output)
    return allClasses[output]

app = Flask(__name__)
cors = CORS(app)
datasetPath = 'data'

@app.route('/api/classname')
def className():
    return "Hello World"
 
@app.route('/api/upload_canvas', methods=['POST'])
def upload_canvas():
    data = json.loads(request.data.decode('UTF-8'))
    image_data = data['image'].split(',')[1].encode('UTF-8')
    filename = data['filename']
    classname = data['className']
    os.makedirs(f'{datasetPath}/{classname}/image',exist_ok=True)
    with open (f'{datasetPath}/{classname}/image/{filename}',"wb") as fh:
        fh.write(base64.decodebytes(image_data))
    return "Got the Image"

@app.route('/api/get_classname', methods=['POST','GET'])
def get_classname():
    data = json.loads(request.data.decode('UTF-8'))
    image_data = data['image'].split(',')[1].encode('UTF-8')
    filename = data['filename']
    os.makedirs(f'{datasetPath}/testimage',exist_ok=True)
    with open (f'{datasetPath}/testimage/{filename}',"wb") as fh:
        fh.write(base64.decodebytes(image_data))
    return test(f'{datasetPath}/testimage/{filename}')
