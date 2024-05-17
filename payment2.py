from flask import Flask, render_template, request
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import csv
import shutil

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pay', methods=['POST'])
def pay():
    money = request.form['money']
    name = request.form['name']
    pay = True
    predict(pay)
    return render_template('pay.html', money=money, name=name, message=message, message2=message2)

@app.route('/add_money', methods=['POST'])
def add_money():
    money = request.form['money']
    name = request.form['name']
    pay = False
    predict(pay)
    return render_template('add_money.html', money=money, name=name, message=message, message2=message2)

@app.route('/capture', methods=['POST'])
def capture():
    money = request.form['money']
    name = request.form['name']
    capture()
    return render_template('capture.html', money=money, name=name)

def clear():
    txt.delete(0, 'end')
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')
    res = ""
    message.configure(text= res)

pay = True

def pay():
    pay = True
    predict(pay)

def add_money():
    pay = False
    predict(pay)

def capture():
    money=(txt.get())
    name=(txt2.get())
    Id = len(os.listdir('dataSet'))//60
    if money.isdigit() and name.isalpha():
        detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        cam = cv2.VideoCapture(0)

        Id= str(len(os.listdir('dataSet'))//60)
        sampleNum=0

        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                #incrementing sample number
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("dataSet\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows()
        row = [Id, name, money]
        with open('Details\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        res = "Image Captured. Proceed To Train."#+",".join(str(f) for f in Id)
        message.configure(text= res)
    else:
        message.configure(text = "Please Enter Name and Account Balance")

def trainer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    path='dataSet'

    if len(os.listdir(path)):

        def getImagesAndLabels(path):
            #get the path of all the files in the folder
            imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
            #create empty face list
            faceSamples=[]
            #create empty ID list
            Ids=[]

            names_list = []
            #now looping through all the image paths and loading the Ids and the images
            for imagePath in imagePaths:
                #loading the image and converting it to gray scale
                pilImage=Image.open(imagePath).convert('L')
                #Now we are converting the PIL image into numpy array
                imageNp=np.array(pilImage,'uint8')
                #getting the Id from the image
                Id=int(os.path.split(imagePath)[-1].split(".")[1])

                # extract the face from the training image sample
                faces=detector.detectMultiScale(imageNp)
                #If a face is there then append that in the list as well as Id of it
                for (x,y,w,h) in faces:
                    faceSamples.append(imageNp[y:y+h,x:x+w])
                    Ids.append(Id)
            return faceSamples,Ids


        faces,Ids  = getImagesAndLabels('dataSet')
        recognizer.train(faces, np.array(Ids))
        recognizer.save('Trainer/trainer.yml')
        res = "Image Trained"#+",".join(str(f) for f in Id)
        message.configure(text= res)

    else:
        message.configure(text = "No images to train. Please take images first.")


def predict(pay):
    transfer=(txt.get())
    name=(txt2.get())
    if transfer.isdigit():
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('Trainer/trainer.yml')

        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath)

        df=pd.read_csv("Details\StudentDetails.csv")
        col_names =  ['Id','Name', 'Money']
        df.set_index('Id')

        # df.loc[0, 'Money'] = df.loc[0, 'Money'] - 700
        # df.to_csv("StudentDetails\StudentDetails.csv", index = False)

        cam = cv2.VideoCapture(0)
        person = None
        count = 0
        match = False
        ans ="Unknown"

        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:
            ret, im =cam.read()
            gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            faces=faceCascade.detectMultiScale(gray, 1.2,5)
            for(x,y,w,h) in faces:
                cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
                Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
                if(conf<50):
                    aa=df.loc[df['Id'] == Id]['Name'].values
                    ans = aa[0]
                else:
                    ans ="Unknown"
                cv2.putText(im,str(ans),(x,y+h), font, 1,(255,255,255),2)
            cv2.imshow('im',im)
            if cv2.waitKey(10) & 0xFF==ord('q'):
                break
            if ans != "Unknown":
                if name == ans:
                    if person == None:
                        person = ans
                    elif person == ans:
                        count += 1
                    elif person != ans:
                        count = 0
                    if count == 30:
                        match = True
                        print("YES")
                        break
            else:
                count = 0

        if match == True:
            if pay == True:
                df.loc[Id, 'Money'] =  df.loc[Id, 'Money'] - int(transfer)
                df.to_csv("Details\StudentDetails.csv", index = False)
                res = "Amount Payed!"#+",".join(str(f) for f in Id)
                message.configure(text= res)
            else:
                df.loc[Id, 'Money'] =  df.loc[Id, 'Money'] + int(transfer)
                df.to_csv("Details\StudentDetails.csv", index = False)
                res = "Amount Added!"#+",".join(str(f) for f in Id)
                message.configure(text= res)

        cam.release()
        cv2.destroyAllWindows()

def get_balance():
    message2.configure(text = "")
    name=(txt2.get())
    # f = []
    if name.isalpha():
        df=pd.read_csv("Details\StudentDetails.csv")
        try:
            row = df.loc[df['Name'] == name]['Money'].reset_index(drop = True)
            cur = row[0]
            message2.configure(text= cur)
        except KeyError:
            message.configure(text = "User Not Found.")
    else:
        message.configure(text = "Please enter User Name")

def clear_data():
    df=pd.read_csv("Details\StudentDetails.csv")
    s = df
    for i in range(len(df)):
        s = s.drop([i], axis = 0)
    s.to_csv("Details\StudentDetails.csv", index = False)
    shutil.rmtree('dataSet')
    os.makedirs('dataSet')
    message.configure(text = "Data Cleared")

if __name__ == '__main__':
    app.run(debug=True)