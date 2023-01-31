from airflow.models import DAG

from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

from datetime import datetime

import pandas as pd
import numpy as np
import os.path
import pickle


# ML
from sklearn.model_selection import train_test_split, KFold, cross_val_score # to split the data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score #To evaluate our model

from sklearn.model_selection import GridSearchCV

# Algorithmns models to be compared
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

# Utils 

    # Save and Load Files and model methods

def save_files(df_list):
    '''
    accepts dataframe list as input
    saves each dataframe in the tmp folder as csv
    the file name corresponds to the dataframe "name" attribute
    '''
    [ df.to_csv('/Users/user/airflow/data/output/german-credit-data-with-risk/files/' + df.name + '.csv' , sep=',', index=False) for df in df_list ]

def load_files(names_list, dir_io):
    '''
    accepts a list of names (str) as input
    load each csv file from the tmp folder with the input names
    returns a list of loaded dataframes
    '''
    df_list = []
    [ df_list.append(pd.read_csv("/Users/user/airflow/data/" + dir_io +"/german-credit-data-with-risk/files/" + name + ".csv")) for name in names_list if  os.path.isfile("/Users/user/airflow/data/" + dir_io +"/german-credit-data-with-risk/files/" + name + ".csv") ]
    
    return df_list

def save_model(model, filename_model):
    # save the model to disk
    filename = '/Users/user/airflow/data/output/german-credit-data-with-risk/models/' + filename_model +'.sav'
    pickle.dump(model, open(filename, 'wb'))
    print(filename_model +'.sav saved')

def write_parquet_file(df, filename_parquet):
    filename = '/Users/user/airflow/data/output/german-credit-data-with-risk/parquets/' + filename_parquet +'.parquet'
    df.to_parquet(filename)
    print(filename_parquet +'.parquet saved')

def write_model(model, df, filename):
    save_model(model, filename)
    write_parquet_file(df, filename)


# Previsualization and preparing data
def first_look(df_credit,  **context):
    #Searching for Missings,type of data and also known the shape of data
    print(df_credit.info())
    #Looking unique values
    print(df_credit.nunique())
    #Looking the data
    print(df_credit.head())
    # XCom Push
    context['task_instance'].xcom_push(key='df_credit', value=df_credit)

def new_cat_age_variable(df_credit, **context):
    #Let's look the Credit Amount column
    #df_credit = context['task_instance'].xcom_pull(key='df_credit')
    interval = (18, 25, 35, 60, 120)
    cats = ['Student', 'Young', 'Adult', 'Senior']
    df_credit["Age_cat"] = pd.cut(df_credit.Age, interval, labels=cats)
    return df_credit

def fill_columns(df_credit):
    df_credit['Saving accounts'] = df_credit['Saving accounts'].fillna('no_inf')
    df_credit['Checking account'] = df_credit['Checking account'].fillna('no_inf')
    return df_credit

def dummy_variable(df_credit, column_name, prefix, drop_first = True):
    '''Transforming the data into Dummy variables¶'''
    df_credit = df_credit.merge(
        pd.get_dummies(
            df_credit[column_name], drop_first=drop_first, prefix=prefix),
            left_index=True, right_index=True)
    return df_credit

def new_variables(df_credit, dummy_cols):
    for column_name, prefix , drop_first in dummy_cols:
        df_credit = dummy_variable(df_credit, column_name, prefix, drop_first)
    return df_credit

def del_variables(df_credit, col_list):
    for col in col_list:
        del df_credit[col]
    return df_credit

def preprocess(df_credit, dummy_cols, col_list ):
    df_credit = new_cat_age_variable(df_credit)
    df_credit = fill_columns(df_credit)
    df_credit = new_variables(df_credit, dummy_cols)
    df_credit = del_variables(df_credit, col_list)
    df_credit['Credit amount'] = np.log(df_credit['Credit amount'])
    print(df_credit.head())
    df_credit.name = 'new_german_credit_data_risk'
    save_files([df_credit])


# prediction model methods 

# Prepare data

def split_data(df_credit):
    #Creating the x and y variables
    x = df_credit.drop('Risk_bad', 1)
    y = df_credit["Risk_bad"]

    # Spliting X and y into train and test version
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=42)

    x_train.name = 'x_train'
    x_test.name = 'x_test'
    y_train.name = 'y_train'
    y_test.name = 'y_test'

    save_files([x_train, x_test, y_train, y_test])

def prepare_model(train_list, models):
        # to feed the random state
    seed = 7
        # evaluate each model in turn
    msg = ''
    results = []
    names = []
    scoring = 'recall'

    x_train = train_list[0]
    y_train = train_list[1]

    for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = msg + "%s: %f (%f) \n" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Models

def rf_model(train_list):
    #Seting the Hyper Parameters
    x_train = train_list[0]
    y_train = train_list[1]
    param_grid = {"max_depth": [3,5, 7, 10,None],
                  "n_estimators":[3,5,10,25,50,150],
                  "max_features": [4,7,15,20]}

    #Creating the classifier
    model = RandomForestClassifier(random_state=2)

    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='recall', verbose=4)
    grid_search.fit(x_train, y_train)
    print(grid_search.best_score_)
    print(grid_search.best_params_)
    rf = RandomForestClassifier(max_depth=None, max_features=10, n_estimators=15, random_state=2)
    #trainning with the best params
    rf.fit(x_train, y_train)

    return rf

def gnb_model(train_list):
    x_train = train_list[0]
    y_train = train_list[1]

    #Creating the classifier
    GNB = GaussianNB()

    # Fitting with train data
    gnb = GNB.fit(x_train, y_train)
    # Printing the Training Score
    print("Training score data: ")
    print(gnb.score(x_train, y_train))

    return gnb

def pipe_model(train_list):
        #Seting the Hyper Parameters
    x_train = train_list[0]
    y_train = train_list[1]
    features = []
    features.append(('pca', PCA(n_components=2)))
    features.append(('select_best', SelectKBest(k=6)))
    feature_union = FeatureUnion(features)
    # create pipeline
    estimators = []
    estimators.append(('feature_union', feature_union))
    estimators.append(('logistic', GaussianNB()))
    pipe = Pipeline(estimators)
    # evaluate pipeline
    seed = 7
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    results = cross_val_score(pipe, x_train, y_train, cv=kfold)
    print(results.mean())
    # Fitting with train data
    pipe.fit(x_train, y_train)
    return pipe

def grid_model(train_list):
    #Seting the Hyper Parameters
    param_test = {
     'max_depth':[3,5,6,10],
     'min_child_weight':[3,5,10],
     'gamma':[0.0, 0.1, 0.2, 0.3, 0.4],
    # 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 10],
     'subsample':[i/100.0 for i in range(75,90,5)],
     'colsample_bytree':[i/100.0 for i in range(75,90,5)]
    }
    x_train = train_list[0]
    y_train = train_list[1]

    #Creating the classifier
    model_xg = XGBClassifier(random_state=2)

    grid_search = GridSearchCV(model_xg, param_grid=param_test, cv=5, scoring='recall')
    grid_search.fit(x_train, y_train)
    print(grid_search.best_score_)
    print(grid_search.best_params_)
    return grid_search

#Model prediction 

def test_model(model, test_list):
    x_test = test_list[0]
    y_test = test_list[1]
    #Testing the model 
    #Predicting using our  model
    y_pred = pd.DataFrame(model.predict(x_test), columns=['y_pred'])

    # Check the results obtained
    print(accuracy_score(y_test,y_pred))
    print("\n")
    print(confusion_matrix(y_test, y_pred))

    return y_pred


def apply_model(train_model, test_list, filename):
    model = train_model
    y_pred = test_model(model, test_list)
    write_model(model, y_pred, filename)



default_args= {
    'owner': 'Yemile Chavez',
    'email_on_failure': False,
    'email': ['yemilec@yahoo.com.mx'],
    'start_date': datetime(2023, 1, 26)
}

with DAG(
    "ml_pipeline",
    description='End-to-end ML pipeline example',
    schedule_interval='@daily',
    default_args=default_args, 
    catchup=False) as dag:

# Variables

    file_list = ['german_credit_data_risk']

    dummy_cols= [('Purpose', 'Purpose', True), ('Sex', 'Sex', True), 
                ('Housing', 'Housing', True), ('Saving accounts', 'Savings', True),
                ('Risk','Risk' , False), ('Checking account', 'Check', True), ('Age_cat', 'Age_cat', True)]

    col_list =['Saving accounts', 'Checking account', 'Purpose', 'Sex', 'Housing', 'Age_cat', 'Risk', 'Risk_good'] 

    new_file_list =  ['new_german_credit_data_risk']
    
    train_list = ['x_train', 'y_train']

    models = [('LR', LogisticRegression()), ('LDA', LinearDiscriminantAnalysis()),
            ('KNN', KNeighborsClassifier()), ('CART', DecisionTreeClassifier()),
            ('NB', GaussianNB()), ('RF', RandomForestClassifier()),
            ('SVM', SVC(gamma='auto')) ,('XGB', XGBClassifier())
            ]
    test_list = ['x_test', 'y_test']

# task: 1
    
    loading_data = PythonOperator(
        task_id='loading_data',
        python_callable=first_look,
        op_kwargs = {'df_credit':load_files(file_list, 'input')[0]},
        provide_context=True
        )

# task: 2

    preprocess_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess,
        op_kwargs = {'df_credit':load_files(file_list, 'input')[0], 
                        'dummy_cols': dummy_cols,
                        'col_list': col_list},
        )

# task: 3

    split_data = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        op_kwargs = {'df_credit':load_files(new_file_list, 'output')[0]},
        )

# task: 4

    prepare_model = PythonOperator(
        task_id='prepare_model',
        python_callable=prepare_model,
        op_kwargs = {'train_list':load_files(train_list, 'output'), 'models': models}
        )

# task: 5

    first_model = PythonOperator(
        task_id='first_model',
        python_callable=apply_model,
        op_kwargs = {'train_model': rf_model(load_files(train_list, 'output')) , 
                    'test_list': load_files(test_list, 'output'),
                    'filename' : 'rf_model' })

# task: 6

    second_model = PythonOperator(
        task_id = 'second_model',
        python_callable = apply_model,
        op_kwargs = {'train_model': gnb_model(load_files(train_list, 'output')) , 
                    'test_list': load_files(test_list, 'output'),
                    'filename' : 'gnb_model' })

# task: 7

    pipeline_model = PythonOperator(
        task_id='pipeline_model',
        python_callable = apply_model,
        op_kwargs = {'train_model': pipe_model(load_files(train_list, 'output')) , 
                    'test_list': load_files(test_list, 'output'),
                    'filename' : 'pipeline_model' })
    # task: 8

    last_model = PythonOperator(
        task_id='last_model',
        python_callable=apply_model,
        op_kwargs = {'train_model': grid_model(load_files(train_list, 'output')) , 
                    'test_list': load_files(test_list, 'output'),
                    'filename' : 'grid_model' })


    loading_data >> preprocess_data >> split_data >> [prepare_model, first_model]
    first_model >> second_model >> pipeline_model >> last_model
