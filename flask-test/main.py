from werkzeug.urls import quote 
from flask import Flask, render_template, request
import os
import cv2
import torch
from ultralytics import YOLO
from datetime import datetime 
from torchvision import transforms as T

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'.jpg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def run_inference(image_path):
    try:
        # Load the YOLO model
        model = YOLO("model/best (1).pt")

        res = model.predict(source=image_path,conf=0.25)
        box1 = res[0].boxes.xywh[0]
        bounding_box1 = box1.cpu().numpy()
        box2 = res[0].boxes.xywh[1]
        bounding_box2 = box2.cpu().numpy()
        if res[0].boxes.data[0][0].numpy()<res[0].boxes.data[1][0].numpy():
            digit1=res[0].boxes.data[0][5].numpy()
            digit2=res[0].boxes.data[1][5].numpy()
        else:
            digit2=res[0].boxes.data[0][5].numpy()
            digit1=res[0].boxes.data[1][5].numpy()
        number=(digit1*10)+digit2
        num=number.tostring()
        x0 = bounding_box1[0] - bounding_box1[2] / 2
        x1 = bounding_box1[0] + bounding_box1[2] / 2
        y0 = bounding_box1[1] - bounding_box1[3] / 2
        y1 = bounding_box1[1] + bounding_box1[3] / 2
        x2 = bounding_box2[0] - bounding_box2[2] / 2
        x3 = bounding_box2[0] + bounding_box2[2] / 2
        y2 = bounding_box2[1] - bounding_box2[3] / 2
        y3 = bounding_box2[1] + bounding_box2[3] / 2

        start_point1 = (int(x0), int(y0))
        end_point1 = (int(x1), int(y1))
        start_point2 = (int(x2), int(y2))
        end_point2 = (int(x3), int(y3))
        img=cv2.imread(image_path)
        cv2.rectangle(img, start_point1, end_point1, color=(0,255,0), thickness=2)
        cv2.rectangle(img, start_point2, end_point2, color=(0,255,0), thickness=2)
    
        return number
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        results = run_inference(filename)
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return render_template('index.html', results=results, current_timestamp=current_timestamp)

    return render_template('index.html', results=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
