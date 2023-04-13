import os
import joblib
import pandas as pd
import pipeline
from flask import Flask, request, Response
from health_insurance.HealthInsurance import HealthInsurance

# loading model
path = ''
model = joblib.load('model/health_insurance_pipe.joblib')

# initialize API
app = Flask( __name__ )

@app.route( '/predict', methods=['POST'] )
def health_insurance_predict():
    # Recebe um json
    test_json = request.get_json()
    
    if test_json: # there is data
        if isinstance( test_json, dict ): # unique example
            test_raw = pd.DataFrame( test_json, index=[0] )
            
        else: # multiple example
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
            
        # Instantiate HealthInsurance class
        pipeline = HealthInsurance()
        
        # copy the original data
        df_raw = test_raw.copy()
        
        # data for prediction 
        df_test = df_raw.drop(columns=['id'])
        
        # prediction
        df_response = pipeline.get_prediction( model, df_raw, df_test )
        
        return df_response
    
    else:
        return Response( '{}', status=200, mimetype='application/json' )
    
if __name__ == '__main__':
    port = os.environ.get( 'PORT', 5000 )
    app.run( host='0.0.0.0', port=port )