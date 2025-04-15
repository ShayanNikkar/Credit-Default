#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 17:57:03 2025

@author: shayannikkar
"""

import pandas as pd 
import numpy as np
from sklearn.metrics import confusion_matrix



def Business_impact(y_test , y_pred ,avrg_loan_amnt, profit_margin_good_loan=0.05 , def_cost=0.4 ):
    tn , fp, fn , tp = confusion_matrix(y_test, y_pred).ravel()
    number_of_approaved_loans = tn + fn
    number_of_rejected_loans = tp + fp
    
    approval_rate  = number_of_approaved_loans / (number_of_approaved_loans+number_of_rejected_loans)
    
    default_cost = def_cost * fn * avrg_loan_amnt
    opportunity_cost = profit_margin_good_loan * fp * avrg_loan_amnt
    total_cost = default_cost + opportunity_cost
    profit = tn * avrg_loan_amnt * profit_margin_good_loan
    
    net_profit = profit - default_cost
    
    return round(approval_rate,4) , round(total_cost,1) , round(net_profit,1)
    
    
    
