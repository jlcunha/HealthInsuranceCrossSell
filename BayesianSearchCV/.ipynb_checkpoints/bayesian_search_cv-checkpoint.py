import pandas as pd
import numpy as np
from skopt import gp_minimize
from scipy.stats import mode


class BayesianSearchCV:
    def __init__(self, model_selection, pipe_processing, params_space, model_params, metric, n_calls=10):
        """
        BayesSearchCV class constructor.

        Args:
        - model_selection: cross-validation method
        - pipe_processing: preprocessing pipeline
        - params_space: hyperparameters space
        - model_params: model parameters
        - n_calls: number of calls for gp_minimize
        - metric: Model evaluation metric.

        Returns:
        - None
        """
        self.model_selection = model_selection
        self.pipe_processing = pipe_processing
        self.params_space = params_space
        self.model_params = model_params
        self.n_calls = n_calls
        self.metric = metric
        
    def fit(self, X, y):
        """
        Function to fit the model.

        Args:
        - X: input features
        - y: target variable

        Returns:
        - info: dictionary containing model performance for each fold
        """
        # Create empty objects to save parameters and scores
        all_params = pd.DataFrame()
        train_score, test_score = [], []
        
        # Join Input features with target variable
        data = X.assign(target=y)
        
        # "Created a counter for Cross Validation Folds."
        fold_no = 1

        # Cross Validation
        for train_index,test_index in self.model_selection.split(data, data.target):
            print(f'\nFold: {fold_no}')
            train, test = data.iloc[train_index], data.iloc[test_index]

            params, evaluation_train, evaluation_test = self.training( train, test )
            
            train_score.append(evaluation_train)
            test_score.append(evaluation_test)
            
            score = pd.DataFrame.from_dict(params, orient='index').T
            all_params = pd.concat([all_params, score], ignore_index=True)
            
            fold_no += 1
            
        all_params = all_params.fillna('None')
        
        all_params['train_score'] = train_score
        all_params['test_score'] = test_score
        
        self.show_all_ = all_params
        
        all_params = all_params.loc[:, (all_params.nunique()>1).values]
        self.best_parameters_ = self.summary_dataframe(all_params.drop(columns=['train_score', 'test_score']))
        
        self.cv_results_ = all_params
        
        return None

    def training( self, train, test ): 
        """
        Function to train the model for cross-validation.

        Args:
        - train: training data
        - test: testing data

        Returns:
        - metrics: model performance metric
        - params: model parameters
        """
        self.data = test
        
        x_train = train.drop(columns = 'target')
        self.x_train = self.pipe_processing.fit_transform(x_train)
        self.y_train = train.target
        
        x_test = test.drop(columns = 'target')
        self.x_test = self.pipe_processing.transform(x_test) 
        self.y_test = test.target

        # Tunning model
        model = self.model_tunning( )
        params = model.get_params()
        
        # training model
        model.fit(self.x_train, self.y_train)
        y_train_pred = model.predict(self.x_train)
        evaluation_train = self.metric( self.y_train, y_train_pred)
        
        y_test_pred = model.predict(self.x_test)
        evaluation_test = self.metric( self.y_test, y_test_pred)
        
        return params, evaluation_train, evaluation_test

    def model_tunning( self ):
        """
        Uses Bayesian optimization to search for the best hyperparameters for a given model.

        Returns:
        - mdl: the best model found after hyperparameter optimization
        """
        
        resultados_gp = gp_minimize(self.optimization_function, 
                                    self.params_space, 
                                    random_state=1, 
                                    verbose=1, 
                                    n_calls=self.n_calls, 
                                    n_random_starts=(int(self.n_calls/3)))   
        
        best_params = resultados_gp.x
        
        mdl = self.model_params(best_params)

        return mdl

    def optimization_function(self, params):
        """
        Objective function used for Bayesian hyperparameter optimization.

        Args:
        - params: hyperparameters to be optimized

        Returns:
        - metrics: negative of the evaluation metric used to determine model performance
        """

        mdl = self.model_params(params)

        mdl.fit(self.x_train, self.y_train)

        y_pred = mdl.predict(self.x_test)

        metrics = self.metric( self.y_test, y_pred)
        
        return -metrics    
    
    def summary_dataframe(self, df):
        """
        Function to summarize the data in a DataFrame.

        Args:
        - df: DataFrame to be summarized

        Returns:
        - summary_df: DataFrame with the summary of the data
        """
        summary_df = pd.DataFrame(columns=['column', 'mean_or_mode'])
    
        for column in df.columns:
            # Check the data type of the column
            if df[column].dtype in ['int64', 'float64']:
                # Calculate the mean for numeric columns
                mean = df[column].mean()
                summary_df = summary_df.append({'column': column, 'mean_or_mode': mean}, ignore_index=True)

            elif df[column].dtype == 'object':
                # Calculate the mode for object columns
                mode_value = mode(df[column])[0][0]
                summary_df = summary_df.append({'column': column, 'mean_or_mode': mode_value}, ignore_index=True)

        return summary_df