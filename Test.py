import torch
import cv2
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
from torchvision import transforms as v2
from model import Classifier  
import numpy as np
import random
import time
random.seed(42)
# Belirli bir uyarıyı filtreleme
warnings.filterwarnings(action='ignore')
start=time.time()

transform = v2.Compose([
    v2.ToTensor()
    #v2.Resize((224, 224)),  # Resmi istediğiniz boyuta yeniden boyutlandırın
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizasyon uygulayın
])


def getModel():
    model = Classifier(1)

    return model
    



model_path = r'weights\classification_model.pt' # Bu model Dino networku 
model = getModel()
model.load_state_dict(torch.load(model_path))
model.eval()


def new_pre(image_path):
    image = cv2.imread(image_path)
    img=cv2.resize(image,(280,280),interpolation=cv2.INTER_LINEAR)
    img=img/255
    image=torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).float()

    return image,img

# Predict the class of the image
def predict(model, image_path):
    image,img = new_pre(image_path)
    with torch.no_grad():
        model.eval()
        output = model(image)
    return output,img

        
# Test with an image
import glob
import os
sigmoid = nn.Sigmoid()
test_image_path = r'Dataset_Quart_smoothed_hikmet\Val\*'
accuracy=0
total_data=0
tp=0
fn=0
tn=0
fp=0
for image_path in glob.glob(os.path.join(test_image_path, '*')):
    
    total_data+=1
    #print(image_path)
    #label_str=(image_path.split("\\")[2])
    label_str=(image_path.split("\\")[2])
    print(label_str)

    if(label_str=="NOK"):
        label=0
    else:
        label=1

    result,img= predict(model, image_path)
    font_color = (255, 255, 255)  # Beyaz renk
    font_thickness = 1
    position = (60, 100)  # Yazının başlangıç konumu
    font_scale = 0.4  # Yazı boyutu
    font = cv2.FONT_HERSHEY_SIMPLEX  # Yazı tipi
    
    acc = result
    acc = sigmoid(acc)
    if(acc>=0.60):
        if(label==1):
            cv2.putText(img, str("Predict OK"), position, font, font_scale, font_color, font_thickness)
            cv2.putText(img, str("Label OK"), (60,122), font, font_scale, font_color, font_thickness)
            cv2.rectangle(img, (0, 0), (144-1, 144-1), (0,255,0), 2)  # -1 dolgu anlamına gelir
            accuracy+=1
            # plt.imshow(img)
            # plt.show()
            print("TRUE POSİTİVE--> OK Doguruluk {}".format(acc[0]))
            tp+=1
        if(label==0):
            cv2.putText(img, str("Predict OK"), position, font, font_scale, font_color, font_thickness)
            cv2.putText(img, str("Label NOK"), (60,122), font, font_scale, font_color, font_thickness)
            cv2.rectangle(img, (0, 0), (144-1, 144-1), (255,0,0), 2)  # -1 dolgu anlamına gelir
            # plt.imshow(img)
            # plt.show()
            print("FALSE POSİTİVE--> OK Doguruluk {}".format(acc[0]))
            fp+=1
    else:
        if(label==0):
            cv2.putText(img, str("Predict NOK"), position, font, font_scale, font_color, font_thickness)
            cv2.putText(img, str("Label NOK"), (60,122), font, font_scale, font_color, font_thickness)
            accuracy+=1
            cv2.rectangle(img, (0, 0), (144-1, 144-1), (0,255,0), 2)  # -1 dolgu anlamına gelir
            # plt.imshow(img)
            # plt.show()
            print("TRUE NEGATİVE--> NOK Doguruluk {}".format(acc[0]))
            tn+=1
        if(label==1):
            cv2.putText(img, str("Predict NOK"), position, font, font_scale, font_color, font_thickness)
            cv2.putText(img, str("Label OK"), (60,122), font, font_scale, font_color, font_thickness)
            cv2.rectangle(img, (0, 0), (144-1, 144-1), (255,0,0), 2)  # -1 dolgu anlamına gelir
            # plt.imshow(img)
            # plt.show()
            print("FALSE NEGATİVE--> NOK Doguruluk {}".format(acc[0]))
            fn+=1

print("--- %s seconds ---" % (time.time() - start))

try:
    precision = tp/(tp+fp)
except:
    precision=1
try:
    recall = tp/(tp+fn)
except:
    recall=1

f1=(2*precision*recall)/(precision+recall)
print("Precision",precision)
print("Recall",recall)
print("F1",f1)
print("TP",tp)
print("FP",fp)
print("TN",tn)
print("FN",fn)
print("Accuracy",accuracy/total_data)