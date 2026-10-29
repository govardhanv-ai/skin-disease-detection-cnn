from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.core.files.storage import FileSystemStorage
import cv2
import tensorflow as tf
import os
from dotenv import load_dotenv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
load_dotenv()

global cnn_algorithm, X_train, X_test, y_train, y_test
class_labels = ['Actinic Keratosis','Basal Cell Carcinoma','Dermatofibroma','Melanoma','Nevus','Pigmented Benign Keratosis',
                'Seborrheic Keratosis','Squamous Cell Carcinoma','Vascular Lesion']

def get_db_connection():
    try:
        return pymysql.connect(
            host='127.0.0.1',
            port=3306,
            user='root',
            password='root',
            database='skindisease',
            charset='utf8'
        )
    except Exception:
        return pymysql.connect(
            host='127.0.0.1',
            port=3306,
            user='root',
            password='',
            database='skindisease',
            charset='utf8'
        )


def build_cnn_model(num_classes=9):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_or_build_model():
    model = build_cnn_model()
    if os.path.exists('model/model_weights.h5'):
        model.load_weights('model/model_weights.h5')
    elif os.path.exists('model/cnn_model_weights.h5'):
        model.load_weights('model/cnn_model_weights.h5')
    return model

def DiseasePrediction(request):
    return render(request, 'DiseasePrediction.html', {})

def DiseasePredictionAction(request):
    if request.method == 'POST' and request.FILES.get('t1'):
        global cnn_algorithm
        try:
            classifier = load_or_build_model()
            myfile = request.FILES['t1']
            fs = FileSystemStorage()
            save_dir = os.path.join('SkinDiseaseApp', 'static', 'samples')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'test.png')
            if os.path.exists(save_path):
                os.remove(save_path)
            fs.save('SkinDiseaseApp/static/samples/test.png', myfile)
            
            image = cv2.imread(save_path)
            if image is not None:
                img = cv2.resize(image, (32, 32))
                im2arr = np.array(img).reshape(1, 32, 32, 3).astype('float32') / 255.0
                preds = classifier.predict(im2arr)
                predict = np.argmax(preds)
                detected_text = "Disease Detected & Classified as : " + class_labels[predict]
                
                # Create annotated image for display
                img_display = cv2.resize(image, (700, 400))
                cv2.putText(img_display, detected_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                result_path = os.path.join(save_dir, 'result.png')
                cv2.imwrite(result_path, img_display)
                
                return render(request, 'DiseasePrediction.html', {
                    'data': detected_text,
                    'result_img': '/static/samples/result.png'
                })
        except Exception as e:
            return render(request, 'DiseasePrediction.html', {'data': f'Error during prediction: {str(e)}'})
    return render(request, 'DiseasePrediction.html', {})

def CNNtestPrediction(name, classifier, X_test, y_test):
    predict = classifier.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    p = precision_score(y_test_labels, predict, average='macro', zero_division=0) * 100
    r = recall_score(y_test_labels, predict, average='macro', zero_division=0) * 100
    f = f1_score(y_test_labels, predict, average='macro', zero_division=0) * 100
    a = accuracy_score(y_test_labels, predict) * 100
    output = '<table border=1 align=center><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th>'
    output += '<th><font size="" color="black">Precision</th><th><font size="" color="black">Recall</th><th><font size="" color="black">FScore</th></tr>'
    output += '<tr><td><font size="" color="black">'+name+'</td>'
    output += '<td><font size="" color="black">'+f"{a:.2f}"+'</td>'
    output += '<td><font size="" color="black">'+f"{p:.2f}"+'</td>'
    output += '<td><font size="" color="black">'+f"{r:.2f}"+'</td>'
    output += '<td><font size="" color="black">'+f"{f:.2f}"+'</td></tr></table>'
    conf_matrix = confusion_matrix(y_test_labels, predict) 
    return conf_matrix, output 

def runCNN(request):
    if request.method == 'GET':
        global cnn_algorithm, X_train, X_test, y_train, y_test
        try:
            X = np.load('model/X.txt.npy')
            Y = np.load('model/Y.txt.npy')
            X = X.astype('float32') / 255.0
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            Y = Y[indices]
            Y = to_categorical(Y)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
            
            classifier = load_or_build_model()
            cnn_algorithm = classifier    
            conf_matrix, output = CNNtestPrediction("CNN Skin Disease Classification", classifier, X_test, y_test)
            context = {'data': output}
            
            plt.figure(figsize=(6, 6)) 
            ax = sns.heatmap(conf_matrix, xticklabels=class_labels, yticklabels=class_labels, annot=True, cmap="viridis", fmt="g")
            ax.set_ylim([0, 9])
            plt.title("CNN Skin Disease Classification Confusion matrix") 
            plt.ylabel('True class') 
            plt.xlabel('Predicted class') 
            plt.tight_layout()
            save_dir = os.path.join('SkinDiseaseApp', 'static', 'samples')
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'cm.png'))
            plt.close()
            
            return render(request, 'ViewOutput.html', context)
        except Exception as e:
            return render(request, 'ViewOutput.html', {'data': f'Error running CNN evaluation: {str(e)}'})

def index(request):
    return render(request, 'index.html', {})

def Login(request):
    return render(request, 'Login.html', {})

def Register(request):
    return render(request, 'Register.html', {})   

def Signup(request):
    if request.method == 'POST':
        username = (request.POST.get('username') or '').strip()
        password = (request.POST.get('password') or '').strip()
        contact = (request.POST.get('contact') or '').strip()
        email = (request.POST.get('email') or '').strip()
        address = (request.POST.get('address') or '').strip()
        if not username or not password:
            return render(request, 'Register.html', {'data': 'Username and Password are required.'})
        try:
            con = get_db_connection()
            with con:
                cur = con.cursor()
                cur.execute("SELECT username FROM register WHERE LOWER(username) = LOWER(%s)", (username,))
                if cur.fetchone():
                    return render(request, 'Register.html', {'data': f'Username "{username}" already exists'})
                
                cur.execute(
                    "INSERT INTO register(username, password, contact, email, address) VALUES(%s, %s, %s, %s, %s)",
                    (username, password, contact, email, address)
                )
                con.commit()
                return render(request, 'Register.html', {'data': 'Signup Process Completed. You can now login.'})
        except Exception as e:
            return render(request, 'Register.html', {'data': f'Database error: {str(e)}'})
    return render(request, 'Register.html', {})

def UserLogin(request):
    if request.method == 'POST':
        username = (request.POST.get('username') or '').strip()
        password = (request.POST.get('password') or '').strip()
        if not username or not password:
            return render(request, 'Login.html', {'data': 'Please enter both username and password.'})
        try:
            con = get_db_connection()
            with con:
                cur = con.cursor()
                cur.execute("SELECT username, password FROM register WHERE LOWER(username) = LOWER(%s)", (username,))
                row = cur.fetchone()
                if row and row[1] == password:
                    return render(request, 'UserScreen.html', {'data': 'Welcome ' + row[0]})
                else:
                    return render(request, 'Login.html', {'data': 'Invalid login details. Please check username and password.'})
        except Exception as e:
            return render(request, 'Login.html', {'data': f'Database error: {str(e)}'})
    return render(request, 'Login.html', {})