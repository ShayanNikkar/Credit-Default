#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 18:33:23 2025

@author: shayannikkar
"""

import pandas as pd
import numpy as np
from Dataloader import load_data
from Model import logistic_reg, decision_tree, random_forest ,k_neighbour, SVM , nn , x_gboost
from impact import Business_impact
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def main():
    
    data =load_data("Data.csv")
    
    models = {"LogisticRegression" : logistic_reg,
              "DecisionTree" : decision_tree,
              "RandomForest" : random_forest,
              "K-NearestNeighbor" : k_neighbour,
              #"SupportVectorMachine" : SVM,
              "NeuralNetwork" : nn,
              "XGBoost" : x_gboost}
    

    business_result = {}
    predictions = {}

    for name , model in models.items():
        y_proba = model(data)
        prof_margin = 0.05
        default_cost = 0.4
        prof={}
        for threshold in np.arange(0.1,1, 0.1):
            y_pred = (y_proba > threshold).astype(int)
            _,_, profit = Business_impact(data["test"][1], y_pred,
                                                   data["avrg_loan_amnt"],
                                                   prof_margin , default_cost)
            prof[threshold]=profit
            
            
            
        best_threshold = max(prof , key= prof.get)
        print("For {} Model Best Threshold is:{}".format(name , round(best_threshold,2)))
        y_pred = (y_proba > best_threshold).astype(int)
        rate , cost, profit = Business_impact(data["test"][1], y_pred,
                                               data["avrg_loan_amnt"],
                                               prof_margin , default_cost)
        predictions[name] = np.where(y_pred==1 , "Default", "No-Default")
        business_result[name]=[rate , cost , profit]
        conf_matrix = confusion_matrix(data["test"][1], y_pred)
    
    
        plt.figure(figsize=(8,6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['No Default', 'Default'], 
               yticklabels=['No Default', 'Default'])
        plt.title('Confusion Matrix - {}'.format(name))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        print("**************************************************************")
        
        
        
    business_result = pd.DataFrame(business_result , index=["Approval_rate", "Model_Total_Cost",
                                                            "Net_Profit"])

    predictions = pd.DataFrame(predictions)
    predictions.insert(0 ,
                           "Actual" ,
                           np.where(data["test"][1]>0 , "Default" , "No-Default")
                           )
    
    
    with pd.ExcelWriter("Credit_Default_Model_Predictor.xlsx") as writer:
        business_result.to_excel(writer, sheet_name='BusinessResult')
        predictions.to_excel(writer, sheet_name='Default-Predictions', index=False)
        print("Results saved to Credit_Default_Model_Predictor.csv")
    
if __name__ == "__main__":
    main()
