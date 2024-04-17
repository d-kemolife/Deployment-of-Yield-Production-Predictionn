# -*- coding: utf-8 -*-

import numpy as np 
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model1 = pickle.load(open('RandomForest.pkl', 'rb'))
model2 = pickle.load(open('RandomForest.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    Crop = int(request.form['Crop'])
    Season = int(request.form['Season'])
    State = int(request.form['State'])
    Area = int(request.form['Area'])
    Annual_Rainfall = int(request.form['Annual_Rainfall'])
    Fertilizer = int(request.form['Fertilizer'])
    Pesticide = int(request.form['Pesticide'])
    #scr = int(request.form['score'])

    final_features = np.array([[Crop, Season, State, Area, Annual_Rainfall, Fertilizer, Pesticide]])
    prediction1 = model1.predict(final_features)
    
    # print(final_features)

    output = prediction1[0]
    
    if output==0:
        result= 'You do not have Good Production'
    else:
        result= 'You have Good Production'
        
    prediction2 = model2.predict(final_features)
    
    output2 = prediction2[0]
    
    if output==0:
        result2 = 0
    else:
        result2 = output2

    return render_template('results.html', Score_num=result2, prediction_text=result)
    

if __name__ == "__main__":
    app.run(debug=True)
