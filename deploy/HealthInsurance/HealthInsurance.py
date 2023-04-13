import joblib
import pandas as pd
import numpy as np


class HealthInsurance(object):
    
    def __init__ (self):
        pass   
        
        
    def get_prediction( self, model, original_data, test_data):
        # model prediction
        pred = model.predict_proba( test_data )
        
        # join prediction into original data
        original_data['score'] = pred[:, 1].tolist()
        
        return original_data.to_json( orient='records', date_format='iso' )