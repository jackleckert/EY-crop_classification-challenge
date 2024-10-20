#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 20:06:57 2023

@author: jackleckert
"""

# Visualization
import ipyleaflet
import matplotlib.pyplot as plt
from IPython.display import Image
import seaborn as sns

# Data Science
import numpy as np
import pandas as pd
import datasets
# Feature Engineering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Machine Learning
from sklearn.metrics import f1_score, accuracy_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression

# Planetary Computer Tools
import pystac
import pystac_client
from pystac_client import Client
from pystac.extensions.eo import EOExtension as eo
import odc
import toolz
from odc.stac import stac_load
import planetary_computer as pc
pc.settings.set_subscription_key('8e9b6659b924459a988554b473a6a8e7')

# Others
import requests
import rich.table
from itertools import cycle
from tqdm import tqdm
tqdm.pandas()


#Get vh_vv function

def get_sentinel_data(latlong,time_slice,assets):
    '''
    Returns VV and VH values for a given latitude and longitude 
    Attributes:
    latlong - A tuple with 2 elements - latitude and longitude
    time_slice - Timeframe for which the VV and VH values have to be extracted
    assets - A list of bands to be extracted
    '''

    latlong=latlong.replace('(','').replace(')','').replace(' ','').split(',')
    bbox_of_interest = (float(latlong[1]) , float(latlong[0]), float(latlong[1]) , float(latlong[0]))
    time_of_interest = time_slice

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-1-rtc"], bbox=bbox_of_interest, datetime=time_of_interest
    )
    items = list(search.get_all_items())
    bands_of_interest = assets
    data = stac_load([items[0]], patch_url=pc.sign, bbox=bbox_of_interest).isel(time=0)
    vh = data["vh"].astype("float").values.tolist()[0][0]
    vv = data["vv"].astype("float").values.tolist()[0][0]
    return vh,vv

def combine_two_datasets(dataset1,dataset2):
    '''
    Returns a  vertically concatenated dataset.
    Attributes:
    dataset1 - Dataset 1 to be combined 
    dataset2 - Dataset 2 to be combined
    '''
    data = pd.concat([dataset1,dataset2], axis=1)
    return data

def plot_confusion_matrix(true_value,predicted_value,title,labels):
    '''
    Plots a confusion matrix.
    Attributes:
    true_value - The ground truth value for comparision.
    predicted_value - The values predicted by the model.
    title - Title of the plot.
    labels - The x and y labels of the plot.
    '''
    cm = confusion_matrix(true_value,predicted_value)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Blues');
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title(title); 
    ax.xaxis.set_ticklabels(labels); 
    ax.yaxis.set_ticklabels(labels);
    
if __name__ == "__main__":
    
    crop_presence_data = pd.read_csv("Crop_Location_Data.csv")
    crop_presence_data.head()
    
    # Function call to extract VV,VH Values
    time_slice = "2020-03-20/2020-03-21"
    assests = ['vh','vv']
    vh_vv = []
    for coordinates in tqdm(crop_presence_data['Latitude and Longitude']):
        vh_vv.append(get_sentinel_data(coordinates,time_slice,assests))
    vh_vv_data = pd.DataFrame(vh_vv,columns =['vh','vv'])
    
    #Combine Latitude, Longitude and VV/VH
    crop_data = combine_two_datasets(crop_presence_data,vh_vv_data)
    crop_data.head()
    
    #Model building
    crop_data = crop_data[['vh','vv','Class of Land']]
    X = crop_data.drop(columns=['Class of Land']).values
    y = crop_data ['Class of Land'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y,random_state=40)
    
    #TIP 4
    #Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    #Model training
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train,y_train)
    
    #Out sample evaluation
    outsample_predictions = model.predict(X_test)
    print("Accuracy {0:.2f}%".format(100*accuracy_score(outsample_predictions, y_test)))
    print(classification_report(y_test, outsample_predictions))
    plot_confusion_matrix(y_test, outsample_predictions,"Model Level 1: Logistic\nRegression Model Out-Sample Results",['Rice', 'Non Rice'])
    
    
    
    
    
    #Submission test 
    
    test_file = pd.read_csv('challenge_1_submission_template.csv')
    test_file.head()
    
    ## Get Sentinel-1-RTC Data
    time_slice = "2020-03-20/2020-03-21"
    assests = ['vh','vv']
    vh_vv = []
    for coordinates in tqdm(test_file['id']):
        vh_vv.append(get_sentinel_data(coordinates,time_slice,assests))
    submission_vh_vv_data = pd.DataFrame(vh_vv,columns =['vh','vv'])
    submission_vh_vv_data.head()
    
    # Feature Scaling 
    submission_vh_vv_data = submission_vh_vv_data.values
    transformed_submission_data = sc.transform(submission_vh_vv_data)
    
    #Making predictions
    final_predictions = model.predict(transformed_submission_data)
    final_prediction_series = pd.Series(final_predictions)
    
    #Combining the results into dataframe
    submission_df = pd.DataFrame({'Latitude and Longitude':test_file['id'].values, 'Class of Land':final_prediction_series.values})
    
    #Displaying the sample submission dataframe
    #display(submission_df)
    
    #Submission file
    #Dumping the predictions into a csv file.
    submission_df.to_csv("challenge_1_submission_rice_crop_prediction.csv",index = False)
    
    