import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_params, log_artifacts
import warnings
warnings.filterwarnings('ignore')



if __name__ == '__main__':
    print('Starting the experiment')
    mlflow.set_experiment(experiment_name = 'mlflow_Assignment_006')
    
    mlflow.autolog() # record automatically

    # loading the dataset 
    df = pd.read_csv('titanic.csv')
    
    # filling missing values in the column age
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # dropping redundant columns
    df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

    # dummy encode categorical variables
    df_numeric_features = df.select_dtypes(include=[np.number]) # Filter the numerical features in the dataset
    df_categorical_features = df.select_dtypes(include=[np.object]) # Filter the categorical features in the dataset.
    # create data frame with only categorical variables that have been encoded
    for col in df_categorical_features.columns.values:
        dummy_encoded_variables = pd.get_dummies(df_categorical_features[col], prefix=col, drop_first=True)
        df_categorical_features = pd.concat([df_categorical_features, dummy_encoded_variables],axis=1)
        df_categorical_features.drop([col], axis=1, inplace=True)
        
    # concatenate the numerical and dummy encoded categorical variables
    df_dummy = pd.concat([df_numeric_features, df_categorical_features], axis=1)
    
    # split data into train subset and test subset
    X = df_dummy.drop('Survived',axis=1)
    y = df_dummy['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1)
    
    # log multiple parameters
    mlflow.log_params({"Train_shape": X_train.shape, "Test_shape": X_test.shape})
    
    # Model building
    model = RandomForestClassifier(criterion="entropy",max_depth=10, min_samples_leaf=3)
    model.fit(X_train, y_train)
    # prediction of test set
    pred = model.predict(X_test)
    
    mlflow.log_params({"criterion":'entropy',"max_depth":10,"min_sample_leaf":3})
    
    # metrics
    accuracy = accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1score = f1_score(y_test,pred)
    
    # logging metrics 
    log_metric('accuracy', accuracy)
    log_metric('precision',precision)
    log_metric('recall',recall)
    log_metric('f1score',f1score)
    
    # logging model
    mlflow.sklearn.log_model(model, "Model")
    # printing the run_id
    print(mlflow.active_run().info.run_uuid)
    print('Ending the experiment')

