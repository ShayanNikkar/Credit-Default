#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 23:16:39 2025

@author: shayannikkar
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data(name , datafolder = "CSV Data"):
    
    '''
    Arg : name of the data frame and the folder are the inputs
    
    Return : A dictionary including training and test datasets as list of [X , y] 
    '''
    
    
    new_directory = "/Users/shayannikkar/Desktop/Qmul/Projects/Credit Default"
    os.chdir(new_directory)
    try:
        data = pd.read_csv(os.path.join(datafolder , name) , index_col="id")
        print("Data Successfully Loaded")
        print("**************************************************************")
    except Exception as e:
        raise RuntimeError(f"Error Loading File :{e}")
        
        
    df = data.copy()    
    df["return_dummy"] = np.where(df["loan_status"]=="Charged Off" , 1 , 0) 
    feature_cols = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 
                'dti', 'open_acc', 'revol_bal', 'revol_util', 'total_acc' ,
                'return_dummy']
    
    df = df.loc[: , df.isna().sum()< 1000]
    df = df[feature_cols]
    df.dropna(inplace=True)
    
    average_loan_amount = df["loan_amnt"].mean()
    
    X = df.loc[: , df.columns != "return_dummy"]
    y = df["return_dummy"]
    X_train , X_test , y_train , y_test = train_test_split(X , y ,
                                                           test_size=0.3 , random_state=42,
                                                           stratify= y)
    
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    data_dictionary = {"train" : (X_train , y_train),  
                       "test" : (X_test , y_test),
                       "avrg_loan_amnt" : average_loan_amount}
    
    
    return data_dictionary



