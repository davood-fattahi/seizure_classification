# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:06:01 2022

@author: Davood
"""


# import scipy.io
# import os
# import tsfresh
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import preprocessing, model_selection, impute
import numpy as np
# import pickle


def prepare_data(file_path=[], case_indicators=[], label_names=[], label_values=[], feature_names=[], any_description=[]):
    if (not(bool(file_path))):
        file_path = '10 and older AI, Final Nov 3.sav'

    if (not(bool(feature_names))):
        feature_names = [	"Age_At_Onset", "Sex", "SE", "Febrile_Convulsion", "Family_Hx_Of_Epilepsy",
                          "Major_Head_Injury", "Medical_Comorbidity",	"Number_of_Seizure_Types", "Exam"]

    if (not(bool(label_names))):
        label_names = ["FinalDx"]
        
    if (not(bool(label_values))):
        label_values = ['Focal', 'FS', 'IGE']
        
    if (not(bool(case_indicators))):
        case_indicators = ["Row",	"Number"]

    df = pd.read_spss(file_path, convert_categoricals=False)
    data = df.loc[:, case_indicators + label_names + feature_names]
    
    data=data[data.FinalDx.isin(label_values)]
    
    if (bool(any_description)):
        for i in any_description.keys():
            feature_names.append(i)
            # print(i)
            a=any_description.get(i)
            b=df.isin(a).any(axis='columns')
            data[i]=b.astype(int)
            
    enc = preprocessing.OrdinalEncoder(encoded_missing_value=np.nan)
    for i in feature_names:
        if data[i].dtypes.name == 'category':
            data[i] = enc.fit_transform(data.get([i]))

    # data = data.replace({'Yes': 1, 'No': -1, 'Abnormal':-1, 'Normal' : 1, 'Male' : 1, 'Female' : -1})
    # data = data.replace({'Yes': 'Yes', np.nan:'Missing'})
 
    # enc = preprocessing.OrdinalEncoder(encoded_missing_value=np.nan)
    # ddata = enc.fit_transform(data)
    
    # imp = impute.SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=(-1))   
    # dddata = imp.fit_transform(ddata)


    return data, case_indicators, label_names, feature_names, enc
