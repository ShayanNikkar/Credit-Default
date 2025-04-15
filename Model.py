#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 16:12:42 2025

@author: shayannikkar
"""

import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tensorflow as tf
import Dataloader
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb


def logistic_reg (data):
    X_train , y_train = data["train"]
    X_test , y_test = data["test"]
    model = LogisticRegression(fit_intercept= False )
    param_grid = { "C" : np.logspace(-3, 3 ,7)}
    lr = GridSearchCV(model, param_grid , scoring="accuracy" , cv=5)
    lr.fit(X_train , y_train)
    print("Fitting Logistic Regression Completed!")
    print("The best hyper parameter is {}".format(lr.best_params_))
    print("Accuracy of Logistic Regression is {}%".format(round(lr.score(X_test,y_test)*100,2)))
    print("**************************************************************")
    y_proba = lr.predict_proba(X_test)[:,1]
    return y_proba


def decision_tree(data):
    X_train , y_train = data["train"]
    X_test , y_test = data["test"]
    model = DecisionTreeClassifier(class_weight="balanced")
    param_grid = {"max_depth" : [5,10,15],
                  "min_samples_split" : [20,30,40],
                  "min_samples_leaf" : [10,20,30]}
    d_tree = GridSearchCV(model, param_grid = param_grid , scoring="accuracy" , cv=5)
    d_tree.fit(X_train , y_train)
    print("Fitting Decision Tree Classifier Completed!")
    print("Best parameters are:{}".format(d_tree.best_params_))
    print("Accuracy of Decision Tree is {}%".format(round(d_tree.score(X_test,y_test)*100,2)))
    print("**************************************************************")
    y_proba = d_tree.predict_proba(X_test)[:,1]
    return y_proba
    

def random_forest(data):
    X_train , y_train = data["train"]
    X_test , y_test = data["test"]
    r_forest = RandomForestClassifier(random_state=42 , max_depth=5 , class_weight="balanced")
    r_forest.fit(X_train , y_train)
    print("Fitting Random Forest Classifier Completed!")
    print("Accuracy of Random Forest is {}%".format(round(r_forest.score(X_test,y_test)*100,2)))
    print("**************************************************************")
    y_proba = r_forest.predict_proba(X_test)[:,1]
    return y_proba
    

def k_neighbour(data):
    X_train , y_train = data["train"]
    X_test , y_test = data["test"]
    model = KNeighborsClassifier(weights="distance")
    param_grid = {"n_neighbors" : [x for x in range(1,100,10)]}
    k_n = GridSearchCV(model, param_grid = param_grid , cv=5 , scoring="accuracy")
    k_n.fit(X_train , y_train)
    print("Fitting K-Neighbors Classifier Completed!")
    print("Best Number of Neighbor is:{}".format(k_n.best_params_))
    print("Accuracy of K-Neighbors is {}%".format(round(k_n.score(X_test,y_test)*100,2)))
    print("**************************************************************")
    y_proba = k_n.predict_proba(X_test)[:,1]
    return y_proba
    
def SVM(data):   
    X_train , y_train = data["train"]
    X_test , y_test = data["test"]
    model = SVC(class_weight="balanced" , probability=True)
    param_grid = {"C" : np.logspace(-1, 1 , 3)}
    svm = GridSearchCV(model, param_grid = param_grid , cv=2 , scoring="accuracy")
    svm.fit(X_train , y_train)
    print("Fitting Support Vector Machine Classifier Completed!")
    print("Best Regularization Parameter is:{}".format(svm.best_params_))
    print("Accuracy of SVM is {}%".format(round(svm.score(X_test,y_test)*100,2)))
    y_proba = svm.predict_proba(X_test)[:,1]
    print("**************************************************************")
    return y_proba


def nn(data):
    X_train , y_train = data["train"]
    X_test , y_test = data["test"]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128 , activation="relu" , input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
        ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1,
              callbacks=[early_stop] , class_weight=class_weights_dict,verbose=0)
    y_proba = model.predict(X_test)
    y_pred = (y_proba > 0.5).astype(int)
    print("Fitting Neural Network Classifier Completed!")
    print("Accuracy score is {}%".format(round(accuracy_score(y_test, y_pred)*100,2)))
    print("**************************************************************")
    return y_proba.ravel()

def x_gboost(data):
    X_train , y_train = data["train"]
    X_test , y_test = data["test"]
    model = xgb.XGBClassifier()
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train) ,  y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    sample_weights = np.array([class_weights_dict[int(label)] for label in y_train])
    model.fit(X_train ,y_train , sample_weight = sample_weights)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    print("Fitting XGBOOST Classifier Completed!")
    print("Accuracy score is {}%".format(round(accuracy_score(y_test, y_pred)*100,2)))
    print("**************************************************************")
    return y_proba
    
    
