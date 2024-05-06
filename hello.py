from flask import Flask, render_template,request,redirect,url_for,session,Response
from flask_pymongo import PyMongo # type: ignore
from flask import jsonify
from pymongo import MongoClient # type: ignore
import subprocess
import cv2
import torch 
import math
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt, strip_optimizer
from torchvision import transforms 
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import mediapipe as mp # type: ignore
from keras.models import load_model
from gtts import gTTS # type: ignore
from io import BytesIO
import pyttsx3 # type: ignore
import threading
app= Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = 'newproject'
count=0
malecount=0
femalecount=0
client = MongoClient('mongodb://localhost:27017/')
db = client['user_database']
users_collection = db['users']

@torch.no_grad()
def findAngle(image, kpts, p1, p2, p3, draw=True):
    coord = []
    no_kpt = len(kpts) // 3
    for i in range(no_kpt):
        cx, cy = kpts[3 * i], kpts[3 * i + 1]
        conf = kpts[3 * i + 2]
        coord.append([i, cx, cy, conf])

    x1, y1 = coord[p1][1:3]
    x2, y2 = coord[p2][1:3]
    x3, y3 = coord[p3][1:3]

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360

    if draw:
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)
        cv2.line(image, (int(x3), int(y3)), (int(x2), int(y2)), (255, 255, 255), 3)
        cv2.circle(image, (int(x1), int(y1)), 10, (255, 255, 255), cv2.FILLED)
        cv2.circle(image, (int(x1), int(y1)), 20, (235, 235, 235), 5)
        cv2.circle(image, (int(x2), int(y2)), 10, (255, 255, 255), cv2.FILLED)
        cv2.circle(image, (int(x2), int(y2)), 20, (235, 235, 235), 5)
        cv2.circle(image, (int(x3), int(y3)), 10, (255, 255, 255), cv2.FILLED)
        cv2.circle(image, (int(x3), int(y3)), 20, (235, 235, 235), 5)

    return int(angle)

def gen_frames10():
    global malecount
    malecount=0
    variable=0
    flag=0
    device = select_device('0')
    model = attempt_load('yolov7-w6-pose.pt', map_location=device)
    _ = model.eval()

    cap = cv2.VideoCapture(0)  # use 0 for webcam
    bcount = 0
    direction = 0
    fontpath = "sfpro.ttf"
    font = ImageFont.truetype(fontpath, 24)
    while cap.isOpened:
        ret, frame = cap.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (1280, 768), interpolation=cv2.INTER_LINEAR)
            image = letterbox(image, (1280), stride=64, auto=True)[0]
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
            image = image.to(device)
            image = image.float()

            with torch.no_grad():
                output, _ = model(image)

            output = non_max_suppression_kpt(output, 0.5, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
            output = output_to_keypoint(output)
            img = image[0].permute(1, 2, 0) * 255
            img = img.cpu().numpy().astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            for idx in range(output.shape[0]):
                kpts = output[idx, 7:].T
                angle = findAngle(img, kpts, 12, 14, 16, draw=True)
                percentage = np.interp(angle, (240, 330), (0, 100))
                bar = np.interp(angle, (250, 330), (668, 100))
                variable=variable+1
                if(percentage<60 and variable % 21==0 and flag==0):
                    speak("you are doing it wrong")
                if percentage >= 60.0:
                    if direction == 0:
                        bcount += 0.5
                        direction = 1
                        flag=1
                                    
                elif percentage == 0.0:
                    if direction == 1:
                        speak("you are doing great")
                        bcount += 0.5
                        direction = 0
                        flag=0
                malecount=int(bcount)
                cv2.line(img, (20, 105), (20, 768 - 100), (232, 222, 117), 20)
                cv2.line(img, (20, int(bar)), (20, 768 - 100), (111, 122, 122), 20)

                if int(percentage) < 10:
                    cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                elif int(percentage) >= 10 and int(percentage) < 100:
                    cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                else:
                    cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)

                cv2.circle(img, (1170, 55), 30, (85, 45, 255), -1)
                cv2.putText(img, f"{int(bcount)}", (1160, 65), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                im = Image.fromarray(img)
                draw = ImageDraw.Draw(im)
                draw.text((1170-(1170-40), int(bar)-15), f"{int(percentage)}%", font=font, fill=(255, 255, 255))
                img = np.array(im)
            ret,buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break
        cv2.waitKey(100)
    cap.release()

def gen_frames11():
    global malecount
    malecount=1
    variable=0
    flag=0
    device = select_device('0')
    model = attempt_load('yolov7-w6-pose.pt', map_location=device)
    _ = model.eval()

    cap = cv2.VideoCapture(0)  # use 0 for webcam
    bcount = 0
    direction = 0
    fontpath = "sfpro.ttf"
    font = ImageFont.truetype(fontpath, 24)

    while cap.isOpened:
            
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (1280, 768), interpolation=cv2.INTER_LINEAR)
                image = letterbox(image, (1280), stride=64, auto=True)[0]
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
                image = image.to(device)
                image = image.float()

                with torch.no_grad():
                    output, _ = model(image)

                output = non_max_suppression_kpt(output, 0.5, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                
                for idx in range(output.shape[0]):
                        kpts = output[idx, 7:].T
                        angle = findAngle(img, kpts, 5, 11, 13, draw=True)
                        
                        percentage = np.interp(angle, (240, 330), (0, 100))
                        
                        bar = np.interp(angle, (250, 330), (668, 100))
                        

                        variable=variable+1


                        if(percentage<70 and variable % 21==0 and flag==0):
                            speak("you are doing it wrong")
                        if percentage >= 70.0:
                            if direction == 0:
                                bcount += 0.5
                                direction = 1
                                flag=1
                        if percentage == 0.0:
                            if direction == 1:
                                speak("you are doing great")
                                bcount += 0.5
                                direction = 0
                                flag=0
                        malecount=int(bcount+1)
                        cv2.line(img, (20, 105), (20, 768 - 100), (232, 222, 117), 20)
                        cv2.line(img, (20, int(bar)), (20, 768 - 100), (111, 122, 122), 20)

                        if int(percentage) < 10:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                        elif int(percentage) >= 10 and int(percentage) < 100:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                        else:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)

                        cv2.circle(img, (1170, 55), 30, (85, 45, 255), -1)
                        cv2.putText(img, f"{int(bcount)}", (1160, 65), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                        im = Image.fromarray(img)
                        draw = ImageDraw.Draw(im)
                        draw.text((1170-(1170-40), int(bar)-15), f"{int(percentage)}%", font=font, fill=(255, 255, 255))
                        img = np.array(im)
                ret,buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                    break
            cv2.waitKey(100)
    cap.release()

def gen_frames12():
    global malecount
    malecount=2
    variable=0
    flag=0
    device = select_device('0')
    model = attempt_load('yolov7-w6-pose.pt', map_location=device)
    _ = model.eval()

    cap = cv2.VideoCapture(0)  # use 0 for webcam
    bcount = 0
    direction = 0
    fontpath = "sfpro.ttf"
    font = ImageFont.truetype(fontpath, 24)
    while cap.isOpened:
           
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (1280, 768), interpolation=cv2.INTER_LINEAR)
                image = letterbox(image, (1280), stride=64, auto=True)[0]
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
                image = image.to(device)
                image = image.float()

                with torch.no_grad():
                    output, _ = model(image)

                output = non_max_suppression_kpt(output, 0.5, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                
                for idx in range(output.shape[0]):
                        kpts = output[idx, 7:].T
                        angle = findAngle(img, kpts, 5, 7, 9, draw=True)
                        
                        percentage = np.interp(angle, (240, 330), (0, 100))
                        
                        bar = np.interp(angle, (250, 330), (668, 100))
                        

                        variable=variable+1
                        if(percentage<60 and variable % 21==0 and flag==0):
                            speak("you are doing it wrong")
                        if percentage >= 40.0:
                            if direction == 0:
                                bcount += 0.5
                                direction = 1
                                flag=1
                        if percentage == 0.0:
                            if direction == 1:
                                speak("you are doing great")
                                bcount += 0.5
                                direction = 0
                                flag=0
                        malecount=int(bcount+2)
                        cv2.line(img, (20, 105), (20, 768 - 100), (232, 222, 117), 20)
                        cv2.line(img, (20, int(bar)), (20, 768 - 100), (111, 122, 122), 20)

                        if int(percentage) < 10:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                        elif int(percentage) >= 10 and int(percentage) < 100:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                        else:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)

                        cv2.circle(img, (1170, 55), 30, (85, 45, 255), -1)
                        cv2.putText(img, f"{int(bcount)}", (1160, 65), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                        im = Image.fromarray(img)
                        draw = ImageDraw.Draw(im)
                        draw.text((1170-(1170-40), int(bar)-15), f"{int(percentage)}%", font=font, fill=(255, 255, 255))
                        img = np.array(im)
                ret,buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                    break
            cv2.waitKey(100)
    cap.release()

def gen_frames13():
    global malecount
    malecount=3
    variable=0
    flag=0
    device = select_device('0')
    model = attempt_load('yolov7-w6-pose.pt', map_location=device)
    _ = model.eval()

    cap = cv2.VideoCapture(0)  # use 0 for webcam
    bcount = 0
    direction = 0
    fontpath = "sfpro.ttf"
    font = ImageFont.truetype(fontpath, 24)
    while cap.isOpened:
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (1280, 768), interpolation=cv2.INTER_LINEAR)
                image = letterbox(image, (1280), stride=64, auto=True)[0]
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
                image = image.to(device)
                image = image.float()

                with torch.no_grad():
                    output, _ = model(image)

                output = non_max_suppression_kpt(output, 0.5, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                
                for idx in range(output.shape[0]):
                        kpts = output[idx, 7:].T
                        angle = findAngle(img, kpts, 5, 11, 13, draw=True)
                        
                        percentage = np.interp(angle, (240, 330), (0, 100))
                        
                        bar = np.interp(angle, (250, 330), (668, 100))
                        

                        variable=variable+1


                        if(percentage<20 and variable % 21==0 and flag==0):
                            speak("you are doing it wrong")
                        if percentage >= 20.0:
                            if direction == 0:
                                bcount += 0.5
                                direction = 1
                                flag=1
                        if percentage == 0.0:
                            if direction == 1:
                                speak("you are doing great")
                                bcount += 0.5
                                direction = 0
                                flag=0
                        malecount=int(bcount+3)
                        cv2.line(img, (20, 105), (20, 768 - 100), (232, 222, 117), 20)
                        cv2.line(img, (20, int(bar)), (20, 768 - 100), (111, 122, 122), 20)

                        if int(percentage) < 10:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                        elif int(percentage) >= 10 and int(percentage) < 100:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                        else:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)

                        cv2.circle(img, (1170, 55), 30, (85, 45, 255), -1)
                        cv2.putText(img, f"{int(bcount)}", (1160, 65), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                        im = Image.fromarray(img)
                        draw = ImageDraw.Draw(im)
                        draw.text((1170-(1170-40), int(bar)-15), f"{int(percentage)}%", font=font, fill=(255, 255, 255))
                        img = np.array(im)
                ret,buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                    break
            cv2.waitKey(100)
    cap.release()

def gen_frames14():
    global malecount
    malecount=4
    variable=0
    flag=0
    device = select_device('0')
    model = attempt_load('yolov7-w6-pose.pt', map_location=device)
    _ = model.eval()

    cap = cv2.VideoCapture(0)  # use 0 for webcam
    bcount = 0
    direction = 0
    fontpath = "sfpro.ttf"
    font = ImageFont.truetype(fontpath, 24)

    while cap.isOpened:
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (1280, 768), interpolation=cv2.INTER_LINEAR)
                image = letterbox(image, (1280), stride=64, auto=True)[0]
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
                image = image.to(device)
                image = image.float()

                with torch.no_grad():
                    output, _ = model(image)

                output = non_max_suppression_kpt(output, 0.5, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                
                for idx in range(output.shape[0]):
                        kpts = output[idx, 7:].T
                        angle = findAngle(img, kpts, 5, 7, 9, draw=True)
                       
                        percentage = np.interp(angle, (240, 330), (0, 100))
                       
                        bar = np.interp(angle, (250, 330), (668, 100))
                        
                        # check for the bicep curls
                        variable=variable+1
                        if(percentage<60 and variable % 21==0 and flag==0):
                            speak("you are doing it wrong")
                        if percentage >= 80.0:
                            if direction == 0:
                                bcount += 0.5
                                direction = 1
                                flag=1
                        if percentage == 0.0:
                            if direction == 1:
                                speak("you are doing great")
                                bcount += 0.5
                                direction = 0
                                flag=0
                        malecount=int(bcount+4)
                        cv2.line(img, (20, 105), (20, 768 - 100), (232, 222, 117), 20)
                        cv2.line(img, (20, int(bar)), (20, 768 - 100), (111, 122, 122), 20)

                        if int(percentage) < 10:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                        elif int(percentage) >= 10 and int(percentage) < 100:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                        else:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)

                        cv2.circle(img, (1170, 55), 30, (85, 45, 255), -1)
                        cv2.putText(img, f"{int(bcount)}", (1160, 65), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                        im = Image.fromarray(img)
                        draw = ImageDraw.Draw(im)
                        draw.text((1170-(1170-40), int(bar)-15), f"{int(percentage)}%", font=font, fill=(255, 255, 255))
                        img = np.array(im)
                ret,buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                    break
            cv2.waitKey(100)
    cap.release()

def gen_frames15():
    global femalecount
    femalecount=0
    variable=0
    flag=0
    device = select_device('0')
    model = attempt_load('yolov7-w6-pose.pt', map_location=device)
    _ = model.eval()

    cap = cv2.VideoCapture(0)  # use 0 for webcam
    bcount = 0
    direction = 0
    fontpath = "sfpro.ttf"
    font = ImageFont.truetype(fontpath, 24)
    while cap.isOpened:
        ret, frame = cap.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (1280, 768), interpolation=cv2.INTER_LINEAR)
            image = letterbox(image, (1280), stride=64, auto=True)[0]
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
            image = image.to(device)
            image = image.float()

            with torch.no_grad():
                output, _ = model(image)

            output = non_max_suppression_kpt(output, 0.5, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
            output = output_to_keypoint(output)
            img = image[0].permute(1, 2, 0) * 255
            img = img.cpu().numpy().astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            for idx in range(output.shape[0]):
                kpts = output[idx, 7:].T
                angle = findAngle(img, kpts, 12, 14, 16, draw=True)
                percentage = np.interp(angle, (240, 330), (0, 100))
                bar = np.interp(angle, (250, 330), (668, 100))
                variable=variable+1
                if(percentage<60 and variable % 21==0 and flag==0):
                    speak("you are doing it wrong")
                if percentage >= 60.0:
                    if direction == 0:
                        bcount += 0.5
                        direction = 1
                        flag=1
                                    
                elif percentage == 0.0:
                    if direction == 1:
                        speak("you are doing great")
                        bcount += 0.5
                        direction = 0
                        flag=0
                femalecount=int(bcount)
                cv2.line(img, (20, 105), (20, 768 - 100), (232, 222, 117), 20)
                cv2.line(img, (20, int(bar)), (20, 768 - 100), (111, 122, 122), 20)

                if int(percentage) < 10:
                    cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                elif int(percentage) >= 10 and int(percentage) < 100:
                    cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                else:
                    cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)

                cv2.circle(img, (1170, 55), 30, (85, 45, 255), -1)
                cv2.putText(img, f"{int(bcount)}", (1160, 65), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                im = Image.fromarray(img)
                draw = ImageDraw.Draw(im)
                draw.text((1170-(1170-40), int(bar)-15), f"{int(percentage)}%", font=font, fill=(255, 255, 255))
                img = np.array(im)
            ret,buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break
        cv2.waitKey(100)
    cap.release()

def gen_frames16():
    global femalecount
    femalecount=1
    variable=0
    flag=0
    device = select_device('0')
    model = attempt_load('yolov7-w6-pose.pt', map_location=device)
    _ = model.eval()

    cap = cv2.VideoCapture(0)  # use 0 for webcam
    bcount = 0
    direction = 0
    fontpath = "sfpro.ttf"
    font = ImageFont.truetype(fontpath, 24)

    while cap.isOpened:
            
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (1280, 768), interpolation=cv2.INTER_LINEAR)
                image = letterbox(image, (1280), stride=64, auto=True)[0]
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
                image = image.to(device)
                image = image.float()

                with torch.no_grad():
                    output, _ = model(image)

                output = non_max_suppression_kpt(output, 0.5, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                
                for idx in range(output.shape[0]):
                        kpts = output[idx, 7:].T
                        angle = findAngle(img, kpts, 5, 11, 13, draw=True)
                        
                        percentage = np.interp(angle, (240, 330), (0, 100))
                        
                        bar = np.interp(angle, (250, 330), (668, 100))
                        

                        variable=variable+1


                        if(percentage<70 and variable % 21==0 and flag==0):
                            speak("you are doing it wrong")
                        if percentage >= 70.0:
                            if direction == 0:
                                bcount += 0.5
                                direction = 1
                                flag=1
                        if percentage == 0.0:
                            if direction == 1:
                                speak("you are doing great")
                                bcount += 0.5
                                direction = 0
                                flag=0
                        femalecount=int(bcount+1)
                        cv2.line(img, (20, 105), (20, 768 - 100), (232, 222, 117), 20)
                        cv2.line(img, (20, int(bar)), (20, 768 - 100), (111, 122, 122), 20)

                        if int(percentage) < 10:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                        elif int(percentage) >= 10 and int(percentage) < 100:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                        else:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)

                        cv2.circle(img, (1170, 55), 30, (85, 45, 255), -1)
                        cv2.putText(img, f"{int(bcount)}", (1160, 65), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                        im = Image.fromarray(img)
                        draw = ImageDraw.Draw(im)
                        draw.text((1170-(1170-40), int(bar)-15), f"{int(percentage)}%", font=font, fill=(255, 255, 255))
                        img = np.array(im)
                ret,buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                    break
            cv2.waitKey(100)
    cap.release()

def gen_frames17():
    global femalecount
    femalecount=2
    variable=0
    flag=0
    device = select_device('0')
    model = attempt_load('yolov7-w6-pose.pt', map_location=device)
    _ = model.eval()

    cap = cv2.VideoCapture(0)  # use 0 for webcam
    bcount = 0
    direction = 0
    fontpath = "sfpro.ttf"
    font = ImageFont.truetype(fontpath, 24)
    while cap.isOpened:
           
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (1280, 768), interpolation=cv2.INTER_LINEAR)
                image = letterbox(image, (1280), stride=64, auto=True)[0]
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
                image = image.to(device)
                image = image.float()

                with torch.no_grad():
                    output, _ = model(image)

                output = non_max_suppression_kpt(output, 0.5, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                
                for idx in range(output.shape[0]):
                        kpts = output[idx, 7:].T
                        angle = findAngle(img, kpts, 5, 7, 9, draw=True)
                        
                        percentage = np.interp(angle, (240, 330), (0, 100))
                        
                        bar = np.interp(angle, (250, 330), (668, 100))
                        

                        variable=variable+1


                        if(percentage<60 and variable % 21==0 and flag==0):
                            speak("you are doing it wrong")
                        if percentage >= 60.0:
                            if direction == 0:
                                bcount += 0.5
                                direction = 1
                                flag=1
                        if percentage == 0.0:
                            if direction == 1:
                                speak("you are doing great")
                                bcount += 0.5
                                direction = 0
                                flag=0
                        femalecount=int(bcount+2)
                        cv2.line(img, (20, 105), (20, 768 - 100), (232, 222, 117), 20)
                        cv2.line(img, (20, int(bar)), (20, 768 - 100), (111, 122, 122), 20)

                        if int(percentage) < 10:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                        elif int(percentage) >= 10 and int(percentage) < 100:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                        else:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)

                        cv2.circle(img, (1170, 55), 30, (85, 45, 255), -1)
                        cv2.putText(img, f"{int(bcount)}", (1160, 65), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                        im = Image.fromarray(img)
                        draw = ImageDraw.Draw(im)
                        draw.text((1170-(1170-40), int(bar)-15), f"{int(percentage)}%", font=font, fill=(255, 255, 255))
                        img = np.array(im)
                ret,buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                    break
            cv2.waitKey(100)
    cap.release()

def gen_frames18():
    global femalecount
    femalecount=3
    variable=0
    flag=0
    device = select_device('0')
    model = attempt_load('yolov7-w6-pose.pt', map_location=device)
    _ = model.eval()

    cap = cv2.VideoCapture(0)  # use 0 for webcam
    bcount = 0
    direction = 0
    fontpath = "sfpro.ttf"
    font = ImageFont.truetype(fontpath, 24)
    while cap.isOpened:
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (1280, 768), interpolation=cv2.INTER_LINEAR)
                image = letterbox(image, (1280), stride=64, auto=True)[0]
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
                image = image.to(device)
                image = image.float()

                with torch.no_grad():
                    output, _ = model(image)

                output = non_max_suppression_kpt(output, 0.5, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                
                for idx in range(output.shape[0]):
                        kpts = output[idx, 7:].T
                        angle = findAngle(img, kpts, 5, 11, 13, draw=True)
                        
                        percentage = np.interp(angle, (240, 330), (0, 100))
                        
                        bar = np.interp(angle, (250, 330), (668, 100))
                        

                        variable=variable+1


                        if(percentage<20 and variable % 21==0 and flag==0):
                            speak("you are doing it wrong")
                        if percentage >= 20.0:
                            if direction == 0:
                                bcount += 0.5
                                direction = 1
                                flag=1
                        if percentage == 0.0:
                            if direction == 1:
                                speak("you are doing great")
                                bcount += 0.5
                                direction = 0
                                flag=0
                        femalecount=int(bcount+3)
                        cv2.line(img, (20, 105), (20, 768 - 100), (232, 222, 117), 20)
                        cv2.line(img, (20, int(bar)), (20, 768 - 100), (111, 122, 122), 20)

                        if int(percentage) < 10:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                        elif int(percentage) >= 10 and int(percentage) < 100:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                        else:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)

                        cv2.circle(img, (1170, 55), 30, (85, 45, 255), -1)
                        cv2.putText(img, f"{int(bcount)}", (1160, 65), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                        im = Image.fromarray(img)
                        draw = ImageDraw.Draw(im)
                        draw.text((1170-(1170-40), int(bar)-15), f"{int(percentage)}%", font=font, fill=(255, 255, 255))
                        img = np.array(im)
                ret,buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                    break
            cv2.waitKey(100)
    cap.release()

def gen_frames19():
    global femalecount
    femalecount=4
    variable=0
    flag=0
    device = select_device('0')
    model = attempt_load('yolov7-w6-pose.pt', map_location=device)
    _ = model.eval()

    cap = cv2.VideoCapture(0)  # use 0 for webcam
    bcount = 0
    direction = 0
    fontpath = "sfpro.ttf"
    font = ImageFont.truetype(fontpath, 24)

    while cap.isOpened:
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (1280, 768), interpolation=cv2.INTER_LINEAR)
                image = letterbox(image, (1280), stride=64, auto=True)[0]
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
                image = image.to(device)
                image = image.float()

                with torch.no_grad():
                    output, _ = model(image)

                output = non_max_suppression_kpt(output, 0.5, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                
                for idx in range(output.shape[0]):
                        kpts = output[idx, 7:].T
                        angle = findAngle(img, kpts, 5, 7, 9, draw=True)
                       
                        percentage = np.interp(angle, (240, 330), (0, 100))
                       
                        bar = np.interp(angle, (250, 330), (668, 100))
                        
                        # check for the bicep curls
                        variable=variable+1
                        if(percentage<60 and variable % 21==0 and flag==0):
                            speak("you are doing it wrong")
                        if percentage >= 70.0:
                            if direction == 0:
                                bcount += 0.5
                                direction = 1
                                flag=1
                        if percentage == 0.0:
                            if direction == 1:
                                speak("you are doing great")
                                bcount += 0.5
                                direction = 0
                                flag=0
                        femalecount=int(bcount+4)
                        cv2.line(img, (20, 105), (20, 768 - 100), (232, 222, 117), 20)
                        cv2.line(img, (20, int(bar)), (20, 768 - 100), (111, 122, 122), 20)

                        if int(percentage) < 10:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                        elif int(percentage) >= 10 and int(percentage) < 100:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)
                        else:
                            cv2.line(img, (40, int(bar)), (70, int(bar)), (111, 122, 122), 20)

                        cv2.circle(img, (1170, 55), 30, (85, 45, 255), -1)
                        cv2.putText(img, f"{int(bcount)}", (1160, 65), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                        im = Image.fromarray(img)
                        draw = ImageDraw.Draw(im)
                        draw.text((1170-(1170-40), int(bar)-15), f"{int(percentage)}%", font=font, fill=(255, 255, 255))
                        img = np.array(im)
                ret,buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                    break
            cv2.waitKey(100)
    cap.release()


def gen_frames():
    # Replace this with your actual frame generation logic
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()
@app.route("/", methods=['GET','POST'])
def home():
    if request.method == "POST":
        email = request.form['email']
        # password = request.form['password']
        user = users_collection.find_one({'email': email,})

        if user:
            session['email'] = email
            # session['password'] = password
            return redirect(url_for('workout'))
        else:
            return render_template('login.html')
    return render_template("home.html")

@app.route("/check_user", methods = ['POST'])
def check_user():
    if request.method == "POST":
        email = request.form['email']
        # password = request.form['password']
        user = users_collection.find_one({'email': email,})

        if user:
            session['email'] = email
            # session['password'] = password
            return render_template('workouts.html')
        else:
            return render_template('login.html')
    
@app.route("/login",methods= ['GET','POST'])
def login():
    return render_template("login.html")

@app.route("/workout",methods= ['GET','POST'])
def workouts():
    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")
    age = request.form.get("age")
    gender = request.form.get("gender")
    weight = request.form.get("weight")
    height = request.form.get("height")

    session['email'] = email
    users_collection.insert_one({
        'name': name,
        'email': email,
        'password': password,
        'age': age,
        'gender': gender,
        'weight': weight,
        'height': height
    })
    return render_template("workouts.html")

@app.route("/signup",methods= ['GET','POST'])
def signup():
    return render_template("register.html")

@app.route("/nextpage",methods = ['GET','POST'])
def nextpage():
        email = session.get('email')
        # Retrieve user's data from the database based on email
        user = users_collection.find_one({'email': email})
        if user:
            # Redirect the user to the appropriate workout page based on gender
            gender = user.get('gender','unknown')
            if gender == "M":
                return redirect(url_for('male_workout'))
            elif gender== 'F':
                return redirect(url_for('female_workout'))

@app.route("/workout/male",methods = ['GET','POST'])
def male_workout():
    return render_template("maleworkout.html")

@app.route("/crunches",methods = ['GET','POST'])
def crunches():
    return render_template("crunches.html")

@app.route("/pushups",methods = ['GET','POST'])
def pushups():
    return render_template("pushups.html")

@app.route("/legraise",methods = ['GET','POST'])
def legraise():
    return render_template("legraise.html")

@app.route("/bicepcurls",methods = ['GET','POST'])
def bicepcurls():
    return render_template("bicepcurls.html")

@app.route("/workout/female",methods = ['GET','POST'])
def female_workout():
    return render_template("femaleworkout.html")

@app.route("/female/crunches",methods = ['GET','POST'])
def female_crunches():
    return render_template("femalecrunches.html")

@app.route("/female/legraise",methods = ['GET','POST'])
def female_legraise():
    return render_template("femalelegraise.html")

@app.route("/female/bicepcurls",methods = ['GET','POST'])
def female_bicepcurls():
    return render_template("femalebicepcurls.html")

@app.route("/female/pushups",methods = ['GET','POST'])
def female_pushups():
    return render_template("femalepushups.html")

@app.route("/cricket",methods = ['GET', 'POST'])
def cricket():
    email = session.get('email')
        # Retrieve user's data from the database based on email
    user = users_collection.find_one({'email': email})
    if user:
            # Redirect the user to the appropriate workout page based on gender
        age = int(user.get('age','unknown'))
        if age<19:
            return redirect(url_for('cricket1_workout'))
        else:
            return redirect(url_for('cricket3_workout'))

@app.route("/cricket1",methods = ['GET','POST'])
def cricket1_workout():
    return render_template("stance.html")
@app.route('/get_count')
def get_count():
    global count
    return str(count)

@app.route('/get_malecount')
def get_malecount():
    global malecount
    return str(malecount)

@app.route('/get_femalecount')
def get_femalecount():
    global femalecount
    return str(femalecount)


@app.route("/coverdrive",methods = ['GET','POST'])
def coverdrive_workout():
    return render_template("coverdrive.html")

@app.route("/thankyou",methods = ['GET','POST'])
def thankyou_page():
    return render_template("lastpage.html")

@app.route("/cricket3",methods = ['GET','POST'])
def cricket3_workout():
    return render_template("latecut.html")

@app.route("/backfootdefence",methods = ['GET','POST'])
def cricket2_workout():
    return render_template("backfootdefence.html")

@app.route("/yoga",methods = ['GET','POST'])
def yoga():
        email = session.get('email')
        # Retrieve user's data from the database based on email
        user = users_collection.find_one({'email': email})
        if user:
            # Redirect the user to the appropriate workout page based on gender
            weight = int(user.get('weight','unknown'))
            height = int(user.get('height','unknown'))
            height= height/100
            bmi = weight/(height**2)
            if bmi<18.5:
                return redirect(url_for('underweight_workout'))
            elif bmi>18.5 and bmi<=24.9:
                return redirect(url_for('healthyweight_workout'))
            else:
                return redirect(url_for('overweight_workout'))

@app.route("/workout/overweight",methods = ['GET','POST'])
def overweight_workout():
    return render_template("goddess.html")
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    engine.say(text)
    def run_engine():
        engine.runAndWait()
    thread = threading.Thread(target=run_engine)
    thread.start()

def inFrame(lst):
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
        return True 
    return False

model  = load_model("yoga_mediapipe/yoga-main/goddess/model1.h5")
label = np.load("yoga_mediapipe/yoga-main/goddess/labels1.npy")
model1 = load_model("yoga_mediapipe/yoga-main/treepose/model1.h5")
label1 = np.load("yoga_mediapipe/yoga-main/treepose/labels1.npy")
model3 = load_model("cricket_mediapipe/stance/model1.h5")
label3 = np.load("cricket_mediapipe/stance/labels1.npy")
model4 = load_model("cricket_mediapipe/backfootdefence/model1.h5")
label4 = np.load("cricket_mediapipe/backfootdefence/labels1.npy")
model5 = load_model("cricket_mediapipe/latecut/model1.h5")
label5 = np.load("cricket_mediapipe/latecut/labels1.npy")
model6 = load_model("cricket_mediapipe/pullshot/model1.h5")
label6 = np.load("cricket_mediapipe/pullshot/labels1.npy")
model7 = load_model("cricket_mediapipe/coverdrive/model1.h5")
label7 = np.load("cricket_mediapipe/coverdrive/labels1.npy")
model8 = load_model("yoga_mediapipe/yoga-main/warrior/model1.h5")
label8 = np.load("yoga_mediapipe/yoga-main/warrior/labels1.npy")
model9 = load_model("yoga_mediapipe/yoga-main/triangle/model1.h5")
label9 = np.load("yoga_mediapipe/yoga-main/triangle/labels1.npy")
holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
desired_width = 740
desired_height = 580
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

def gen_frames1():
    global count
    count=0
    while True:
        lst = []
        success, frm = cap.read()
        if not success:
            break

        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        frm = cv2.blur(frm, (4, 4))

        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst = np.array(lst).reshape(1, -1)

            p = model1.predict(lst)
            pred = label1[np.argmax(p)]

            if p[0][np.argmax(p)] > 0.75:
                if pred == "treepose":
                    count=count+1
                    speak("Good, you are fantastic!")
                    landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 0), circle_radius=3, thickness=3)
                    connection_drawing_spec = drawing.DrawingSpec(color=(0, 255, 0), thickness=6)
                else:
                    speak("you are doing it wrong")
                    landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
                    connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)
            else:
                landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
                connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)
        else: 
            speak("Make sure full body is visible")
            landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
            connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)

        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                                   connection_drawing_spec=connection_drawing_spec,
                                   landmark_drawing_spec=landmark_drawing_spec)

        ret, buffer = cv2.imencode('.jpg', frm)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Wait for a while to let the speaking finish before proceeding to the next frame
        cv2.waitKey(100)

    # Release the camera capture
    cap.release()

def gen_frames():
    global count
    count=0
    while True:
        lst = []
        success, frm = cap.read()
        if not success:
            break

        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        frm = cv2.blur(frm, (4, 4))

        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst = np.array(lst).reshape(1, -1)

            p = model.predict(lst)
            pred = label[np.argmax(p)]

            if p[0][np.argmax(p)] > 0.75:
                if pred == "goddess":
                    count=count+1
                    speak("Good, you are fantastic!")
                    landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 0), circle_radius=3, thickness=3)
                    connection_drawing_spec = drawing.DrawingSpec(color=(0, 255, 0), thickness=6)
                else:
                    speak("you are doing it wrong")
                    landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
                    connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)
            else:
                cv2.putText(frm, "Asana is either wrong or not trained", (100, 180), cv2.FONT_ITALIC, 1.8, (0, 0, 255), 3)
                landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
                connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)
        else: 
            
            speak("Make sure full body is visible")
            landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
            connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)

        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                                   connection_drawing_spec=connection_drawing_spec,
                                   landmark_drawing_spec=landmark_drawing_spec)

        ret, buffer = cv2.imencode('.jpg', frm)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Wait for a while to let the speaking finish before proceeding to the next frame
        cv2.waitKey(100)

    # Release the camera capture
    cap.release()
def gen_frames2():
    global count
    count=0
    while True:
        lst = []
        success, frm = cap.read()
        if not success:
            break

        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        frm = cv2.blur(frm, (4, 4))

        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst = np.array(lst).reshape(1, -1)

            p = model3.predict(lst)
            pred = label3[np.argmax(p)]

            if p[0][np.argmax(p)] > 0.75:
                if pred == "stance":
                    count=count+1
                    speak("Good, you are fantastic!")
                    landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 0), circle_radius=3, thickness=3)
                    connection_drawing_spec = drawing.DrawingSpec(color=(0, 255, 0), thickness=6)
                else:
                    speak("you are doing it wrong")
                    landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
                    connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)
            else:
                landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
                connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)
        else: 
            
            speak("Make sure full body is visible")
            
            landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
            connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)

        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                                   connection_drawing_spec=connection_drawing_spec,
                                   landmark_drawing_spec=landmark_drawing_spec)

        ret, buffer = cv2.imencode('.jpg', frm)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Wait for a while to let the speaking finish before proceeding to the next frame
        cv2.waitKey(100)

    # Release the camera capture
    cap.release()
def gen_frames4():
    global count
    count=0
    while True:
        lst = []
        success, frm = cap.read()
        if not success:
            break

        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        frm = cv2.blur(frm, (4, 4))

        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst = np.array(lst).reshape(1, -1)

            p = model4.predict(lst)
            pred = label4[np.argmax(p)]

            if p[0][np.argmax(p)] > 0.75:
                if pred == "backfootdefence":
                    count=count+1
                    speak("Good, you are fantastic!")
                    landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 0), circle_radius=3, thickness=3)
                    connection_drawing_spec = drawing.DrawingSpec(color=(0, 255, 0), thickness=6)
                else:
                    speak("you are doing it wrong")
                    landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
                    connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)
            else:
                landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
                connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)
        else: 
            
            speak("Make sure full body is visible")
            landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
            connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)

        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                                   connection_drawing_spec=connection_drawing_spec,
                                   landmark_drawing_spec=landmark_drawing_spec)

        ret, buffer = cv2.imencode('.jpg', frm)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Wait for a while to let the speaking finish before proceeding to the next frame
        cv2.waitKey(100)

    # Release the camera capture
    cap.release()
def gen_frames5():
    global count
    count=0
    while True:
        lst = []
        success, frm = cap.read()
        if not success:
            break

        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        frm = cv2.blur(frm, (4, 4))

        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst = np.array(lst).reshape(1, -1)

            p = model5.predict(lst)
            pred = label5[np.argmax(p)]

            if p[0][np.argmax(p)] > 0.75:
                if pred == "latecut":
                    count=count+1
                    speak("Good, you are fantastic!")
                    landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 0), circle_radius=3, thickness=3)
                    connection_drawing_spec = drawing.DrawingSpec(color=(0, 255, 0), thickness=6)
                else:
                    speak("you are doing it wrong")
                    landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
                    connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)
            else:
                landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
                connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)
        else: 
       
            speak("Make sure full body is visible")
            landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
            connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)

        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                                   connection_drawing_spec=connection_drawing_spec,
                                   landmark_drawing_spec=landmark_drawing_spec)

        ret, buffer = cv2.imencode('.jpg', frm)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Wait for a while to let the speaking finish before proceeding to the next frame
        cv2.waitKey(100)

    # Release the camera capture
    cap.release()

def gen_frames6():
    global count
    count=0
    while True:
        lst = []
        success, frm = cap.read()
        if not success:
            break

        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        frm = cv2.blur(frm, (4, 4))

        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst = np.array(lst).reshape(1, -1)

            p = model6.predict(lst)
            pred = label6[np.argmax(p)]

            if p[0][np.argmax(p)] > 0.75:
                if pred == "pullshot":
                    count=count+1
                    speak("Good, you are fantastic!")
                    landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 0), circle_radius=3, thickness=3)
                    connection_drawing_spec = drawing.DrawingSpec(color=(0, 255, 0), thickness=6)
                else:
                    speak("you are doing it wrong")
                    landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
                    connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)
            else:
                landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
                connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)
        else: 
            speak("Make sure full body is visible")        
            landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
            connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)

        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                                   connection_drawing_spec=connection_drawing_spec,
                                   landmark_drawing_spec=landmark_drawing_spec)

        ret, buffer = cv2.imencode('.jpg', frm)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Wait for a while to let the speaking finish before proceeding to the next frame
        cv2.waitKey(100)

    # Release the camera capture
    cap.release()

def gen_frames7():
    global count
    count=0
    while True:
        lst = []
        success, frm = cap.read()
        if not success:
            break

        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        frm = cv2.blur(frm, (4, 4))

        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst = np.array(lst).reshape(1, -1)

            p = model7.predict(lst)
            pred = label7[np.argmax(p)]

            if p[0][np.argmax(p)] > 0.75:
                if pred == "coverdrive":
                    count=count+1
                    speak("Good, you are fantastic!")
                    landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 0), circle_radius=3, thickness=3)
                    connection_drawing_spec = drawing.DrawingSpec(color=(0, 255, 0), thickness=6)
                else:
                    speak("you are doing it wrong")
                    landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
                    connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)
            else:
                landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
                connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)
        else: 
            speak("Make sure full body is visible")
            landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
            connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)

        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                                   connection_drawing_spec=connection_drawing_spec,
                                   landmark_drawing_spec=landmark_drawing_spec)

        ret, buffer = cv2.imencode('.jpg', frm)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Wait for a while to let the speaking finish before proceeding to the next frame
        cv2.waitKey(100)

    # Release the camera capture
    cap.release()

def gen_frames8():
    global count
    count=0
    while True:
        lst = []
        success, frm = cap.read()
        if not success:
            break

        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        frm = cv2.blur(frm, (4, 4))

        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst = np.array(lst).reshape(1, -1)

            p = model8.predict(lst)
            pred = label8[np.argmax(p)]

            if p[0][np.argmax(p)] > 0.75:
                if pred == "warrior":
                    count=count+1
                    speak("Good, you are fantastic!")
                    landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 0), circle_radius=3, thickness=3)
                    connection_drawing_spec = drawing.DrawingSpec(color=(0, 255, 0), thickness=6)
                else:
                    speak("you are doing it wrong")
                    landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
                    connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)
            else:
                landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
                connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)
        else: 
            speak("Make sure full body is visible")
            landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
            connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)

        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                                   connection_drawing_spec=connection_drawing_spec,
                                   landmark_drawing_spec=landmark_drawing_spec)

        ret, buffer = cv2.imencode('.jpg', frm)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Wait for a while to let the speaking finish before proceeding to the next frame
        cv2.waitKey(100)

    # Release the camera capture
    cap.release()

def gen_frames9():
    global count
    count=0
    while True:
        lst = []
        success, frm = cap.read()
        if not success:
            break

        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        frm = cv2.blur(frm, (4, 4))

        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst = np.array(lst).reshape(1, -1)

            p = model9.predict(lst)
            pred = label9[np.argmax(p)]

            if p[0][np.argmax(p)] > 0.75:
                if pred == "traingle":
                    count=count+1
                    speak("Good, you are fantastic!")
                    landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 0), circle_radius=3, thickness=3)
                    connection_drawing_spec = drawing.DrawingSpec(color=(0, 255, 0), thickness=6)
                else:
                    speak("you are doing it wrong")
                    landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
                    connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)
            else:
                landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
                connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)
        else: 
            
            speak("Make sure full body is visible")
            landmark_drawing_spec = drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3)
            connection_drawing_spec = drawing.DrawingSpec(color=(255, 255, 255), thickness=6)

        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                                   connection_drawing_spec=connection_drawing_spec,
                                   landmark_drawing_spec=landmark_drawing_spec)

        ret, buffer = cv2.imencode('.jpg', frm)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Wait for a while to let the speaking finish before proceeding to the next frame
        cv2.waitKey(100)

    # Release the camera capture
    cap.release()


@app.route('/treepose',methods = ['GET','POST'])
def tree_pose():
    return render_template("treepose.html")


@app.route('/triangle',methods = ['GET','POST'])
def triangle_pose():
    return render_template("triangle.html")

@app.route('/pullshot',methods = ['GET','POST'])
def pullshot_pose():
    return render_template("pullshot.html")

@app.route("/workout/healthyweight",methods = ['GET','POST'])
def healthyweight_workout():
    return render_template("healthyweightworkout.html")

@app.route("/workout/underweight",methods = ['GET','POST'])
def underweight_workout():
    return render_template("warrior.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed1')
def video_feed1():
    return Response(gen_frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3')
def video_feed3():
    return Response(gen_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed4')
def video_feed4():
    return Response(gen_frames4(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed5')
def video_feed5():
    return Response(gen_frames5(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed6')
def video_feed6():
    return Response(gen_frames6(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed7')
def video_feed7():
    return Response(gen_frames7(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed8')
def video_feed8():
    return Response(gen_frames8(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed9')
def video_feed9():
    return Response(gen_frames9(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed10')
def video_feed10():
    return Response(gen_frames10(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed11')
def video_feed11():
    return Response(gen_frames11(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed12')
def video_feed12():
    return Response(gen_frames12(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed13')
def video_feed13():
    return Response(gen_frames13(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed14')
def video_feed14():
    return Response(gen_frames14(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed15')
def video_feed15():
    return Response(gen_frames15(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed16')
def video_feed16():
    return Response(gen_frames16(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed17')
def video_feed17():
    return Response(gen_frames17(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed18')
def video_feed18():
    return Response(gen_frames18(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed19')
def video_feed19():
    return Response(gen_frames19(), mimetype='multipart/x-mixed-replace; boundary=frame')
app.run(debug=True)