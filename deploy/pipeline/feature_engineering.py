from sklearn.base             import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = self.feature_engineering(X)
        
        return X
    
    def feature_engineering(self, df ):
        # Copy Dataframe
        data = df.copy()
        
        # Vehicle Age
        vehicle_age_map = {"1-2 Year": 1.5,
                           "< 1 Year": 1.0,
                           "> 2 Years": 2.0 }

        data.vehicle_age = data.vehicle_age.map(vehicle_age_map)
        
        # Power vintage ( Square )
        data['vintage'] = data.vintage.pow(2)        
        
        # Create New Features
        data['annual_vintage'] = data.annual_premium * data.vintage
        data['policy_region'] = data.policy_sales_channel * (data.region_code + 1)
        data['age_vintage'] = data.age * data.vintage
        data['annual_age'] = data.annual_premium * data.age
        data['annual_vehicle'] = data.annual_premium * data.vehicle_age
        data['vintage_vehicle'] = data.vintage * data.vehicle_age
        
        return data