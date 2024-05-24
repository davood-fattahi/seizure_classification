# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:30:07 2023

@author: Davood
"""
import numpy as np
import pickle
import math
from flask import Flask, render_template, request


# import the classifier model

fileName = 'ClassifierModel.pkl'

with open('./' + fileName, 'rb') as f:
    StClf = pickle.load(f)


def calculate_value(param1, param2, param3, param4, param5, param6, param7, param8, param9):
    # Calculate the numerical value based on the 9 parameters
    X_test=np.array([param1,param2,param3,param4,param5,param6,param7,param8,param9],ndmin=2)
    value=StClf.predict(X_test);
    proba=StClf.predict_proba(X_test);
    return value, proba
   
    
def c2int(s):
    try:
        return int(s)
    except ValueError:
        return math.nan

       

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('GUI_v7.html')

@app.route('/calculate', methods=['POST'])



def calculate():
    # Get the form data
    param1 = c2int(request.form['param1'])
    param2 = c2int(request.form['param2'])
    param3 = c2int(request.form['param3'])
    param4 = c2int(request.form['param4'])
    param5 = c2int(request.form['param5'])
    param6 = c2int(request.form['param6'])
    param7 = c2int(request.form['param7'])
    param8 = c2int(request.form['param8'])
    param9 = c2int(request.form['param9'])

    # Perform the calculation
    if (param9 == 1 )| (param9 == 2):
        param9e = 1
    else:
        param9e = 0
    value, proba = calculate_value(param1, param2, param3, param4, param5, param6, param7, param8, param9e)
    result_label = str(value[0])
    proba = str(round(np.max(proba),2))
    # Return the result to the web page
    return render_template('GUI_v7.html', result=result_label, proba=proba, param1=param1, param2=param2, param3=param3, param4=param4, param5=param5, param6=param6, param7=param7, param8=param8, param9=param9)

if __name__ == '__main__':
    app.run()
