from matplotlib.pyplot import text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from simplepreprocessor import SimplePreprocessor
from simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import cv2
import openai
import io
import requests
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, 
    help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1, 
    help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1, 
    help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

#print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
#print(imagePaths)

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
#print(labels)
data = data.reshape((data.shape[0], 3072))

#print("[INFO] features matrix: {:.1f}MB".format( 
   # data.nbytes / (1024 * 1000.0)))

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, 
    test_size=0.25, random_state=42)

#print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], 
    n_jobs=args["jobs"])
model.fit(trainX, trainY)


label_data = ["Basketball Theme Cake", "Delicious Plain Cake", "Piniata Cake", "Pull Me Up Cake", "Superhero Theme Cake", "Unicorn Theme Cake"]

""" print('wow : ', label_data[model.predict(np.array([testX[5]]))[0]])
print(classification_report(testY, model.predict(testX), 
    target_names=le.classes_)) """


app = Flask(__name__)
CORS(app)

def getDescription(label):
    openai.api_key = "YOUR-OPENAI-API-KEY"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt= label,
        temperature=0.3,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )   
    print(response)

    return {'textDescription' : response.choices[0].text, 'label' : label}

def predictClass(url):
    return download_image('input_images/input/', url, "input.jpg")



def download_image(download_path, url, file_name):
    try:

        image_content = requests.get(url).content
    
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file)
        file_path = download_path + file_name 
     
        with open(file_path, "wb") as f:
            image.save(f, "JPEG")

        #-----
        data = []
        print(file_path)
        img = cv2.imread('input_images/input/input.jpg') 
        print(img)
       
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

        data.append(img)

        data2 = np.array(data)
        data2 = data2.reshape((data2.shape[0], 3072))

        #print('Ohooo : ', data2[0])
        return getDescription(label_data[model.predict(np.array([data2[0]]))[0]])

    except Exception as e:
        print("FAILED - ",e)
        
        return {"prediction" : 'FAILED'}



@app.route('/getImgUrl', methods = ["POST"])
def getImgUrl():
    url = request.json['imgUrl']
    return predictClass(url)

if __name__ == "__main__":
    app.run(debug=True)
