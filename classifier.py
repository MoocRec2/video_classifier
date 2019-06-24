import cv2
import os
import os.path
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('tf')

classifier_path = './models/vgg16_model.h5'
classifier = load_model(classifier_path)
classifier.summary()
model = VGG16(weights='imagenet', include_top=False )

video_directory = './videos'
img_directory = './images'
num = 0
count = 0
head = 0
code = 0
slide = 0
model = VGG16(weights='imagenet', include_top=False )
classifier_path = './models/vgg16_model.h5'

def deleteImages():
    for root, dirs, files in os.walk(img_directory):
        for file in files:
            os.remove(os.path.join(root,file))


def predict(file):
    global model
    x = image.load_img(file, target_size=(150,150))
    x = image.img_to_array(x)
    x = x/255
    x = np.expand_dims(x, axis=0)
    features = model.predict(x)
    result = classifier.predict_classes(features)
    if result[0] == 0:
        prediction = 'code'
    elif result[0] == 1:
        prediction = 'head'
    elif result[0] == 2:
        prediction = 'slide'
    return prediction

def videoStyles(file):
    global count,head,code,slide,num
    cap = cv2.VideoCapture(video_directory+file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if(count%120 ==0):
            num += 1
            name = './images/img_'+str(num)+'.jpg'
            print ('Creating...'+name)
            cv2.imwrite(name,frame)
        count += 1

    cap.release()

    list_img = os.listdir(img_directory)
    img_count = len(list_img)
    for img in list_img:
        temp = predict(img_directory+img)
        if temp == 'head':
            head +=1
        elif temp == 'code':
            code +=1
        elif temp == 'slide':
            slide +=1

    deleteImages()

    head_p = round(((head / img_count) * 100), 2)
    code_p = round(((code / img_count) * 100), 2)
    slide_p = round(((slide / img_count) * 100), 2)
    return head_p, code_p, slide_p

    head_c,code_c,slide_c = videoStyles('./test.mp4')
    print('Talkin Head: '+str(head_c))
    print('Code: '+str(code_c))
    print('Slide: '+str(slide_c))
