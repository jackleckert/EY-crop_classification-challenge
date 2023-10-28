#!/usr/bin/env python
# coding: utf-8

# # Time series classification using Random Forest 
# 
# This notebook proposes a solution to the EY Open Science Data Challenge - Level 1, with the objective to identify rice fields in a region of Vietnam using only Sentinel 1 satellite VH and VV time series values.
# 




# Visualization
import ipyleaflet
import matplotlib.pyplot as plt
from IPython.display import Image
import seaborn as sns

# Data Science
import numpy as np
import pandas as pd
# Feature Engineering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Machine Learning
from sklearn.metrics import f1_score, accuracy_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle


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
    
    box_size_deg = 0.0009 # Surrounding box in degrees, yields approximately 5x5 pixel region

    min_lon = float(latlong[1])-box_size_deg/2
    min_lat = float(latlong[0])-box_size_deg/2
    max_lon = float(latlong[1])+box_size_deg/2
    max_lat = float(latlong[0])+box_size_deg/2

    bbox_of_interest = (min_lon, min_lat, max_lon, max_lat)
    #bbox_of_interest = (float(latlong[1]) , float(latlong[0]), float(latlong[1]) , float(latlong[0]))
    
    time_of_interest = time_slice

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-1-rtc"], bbox=bbox_of_interest, datetime=time_of_interest
    )
    items = list(search.get_all_items())
    
    
    
    # Define the pixel resolution for the final product
    # Define the scale according to our selected crs, so we will use degrees

    resolution = 10  # meters per pixel 
    scale = resolution / 111320.0 # degrees per pixel for crs=4326 
    
    # Load the data using Open Data Cube
    data = stac_load(items,bands=["vv", "vh"], patch_url=pc.sign, bbox=bbox_of_interest, crs="EPSG:4326", resolution=scale)
   
    # Calculate the mean of the data across the sample region
    mean = data.mean(dim=['latitude','longitude']).compute()
    
    # Calculate V
    v = mean.vv + mean.vh
    
    
    return v.values

def get_new_features(data,L):
    #data = 600 time series list of values
    
    
    new_features=[]
    if 'mean' in L:
        the_means=[]
        for col in data.columns:
            mean_v = float(np.mean(data[col]))
            the_means.append(mean_v)
        new_features.append(the_means)
    
    if 'std' in L:
        the_stds=[]
        for col in data.columns:
            sigma_v = float(np.std(data[col]))
            the_stds.append(sigma_v)
        new_features.append(the_stds)
    
    if 'CV' in L:
        the_CVs=[]
        for col in data.columns:
            mean_v = float(np.mean(data[col]))
            sigma_v = float(np.std(data[col]))
            CV_v = float(mean_v/sigma_v)
            the_CVs.append(CV_v)
        new_features.append(the_CVs)
    
    if 'max' in L:
        the_maxs=[]
        for col in data.columns:
            max_v = float(max(data[col]))
            the_maxs.append(max_v)
        new_features.append(the_maxs)
        
    if 'min' in L:
        the_mins=[]
        for col in data.columns:
            min_v = float(min(data[col]))
            the_mins.append(min_v)
        new_features.append(the_mins)
        
    if 'median' in L:
        the_medians=[]
        for col in data.columns:
            median_v = float(np.median(data[col]))
            the_medians.append(median_v)
        new_features.append(the_medians)
    
    if 'amplitude' in L:
        the_amplis=[]
        for col in data.columns:
            max_v = float(max(data[col]))
            min_v = float(min(data[col]))
            ampli_v = abs(max_v-min_v)
            the_amplis.append(ampli_v)
        new_features.append(the_amplis)
        
    if 'absolute energy' in L:
        the_abs_Es=[]
        for col in data.columns:
            abs_E_v = float(sum((data[col])**2))
            the_abs_Es.append(abs_E_v)
        new_features.append(the_abs_Es)
    
    if 'mean diff' in L:
        the_mean_diffs=[]
        col_1 = np.delete(data[col].values,0)
        col_2 = np.delete(data[col].values,len(data[col])-1)
        delta_v = col_1-col_2
        for col in data.columns:
            mean_delta_v = float(np.mean(delta_v))
            the_mean_diffs.append(mean_delta_v)
        new_features.append(the_mean_diffs)
        
    if 'autocorrelation lag 1' in L:
        the_auto_lag_1s=[]
        for col in data.columns: 
            the_auto=sm.tsa.acf(data[col])[1]
            the_auto_lag_1s.append(the_auto)
        new_features.append(the_auto_lag_1s)
    
    #median_delta_v = float(np.median(delta_v))
    #sum_delta_v = float(sum(abs(delta_v)))
    #dist_v = float(sum(np.sqrt(1+delta_v**2)))#signal distance 
    
    return np.transpose(pd.DataFrame(new_features))
        
def anomaly_detection(list_vf, crop_presence_data, value):
    to_drop=[]
    for i in range(np.shape(list_vf)[0]):
        for j in range(np.shape(list_vf)[1]):
            if abs(list_vf[j][i])>=value:
                to_drop.append(i)
    list_vf=list_vf.drop(index=list_vf.index[to_drop])
    crop_presence_data=crop_presence_data.drop(index=crop_presence_data.index[to_drop])
    list_vf=list_vf.reset_index(drop=True)
    crop_presence_data=crop_presence_data.reset_index(drop=True)
    return list_vf, crop_presence_data
    
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

    #Load the crop locations
    crop_presence_data = pd.read_csv("Crop_Location_Data.csv")
    crop_presence_data.head()
    
    
    # ## Index and box size choice
    # The first step was to choose wisely a vegetation index as a combination of VV and VH values. By plotting several indexes (RVI, SNI, vh+vv, vh-vv, vh/vv and vh+vv/vh-vv) for a single day, the difference between the 'Non Rice' and 'Rice' values could be analyzed. The vh+vv values gave the best results and were therefore chosen.
    # 
    # The second parameter to choose was the box size on which the VV and VH values would be averaged in order to prevent overfitting. By testing 3x3, 5x5, 7x7, 8x8, 9x9, 10x10 and 11x11 box sizes, the best accuracy results were obtained with a box of 10x10.
    # 
    # Finally, a whole year of Sentinel 1 remote sensing was taken to build time series with enough data points.
    
    
    
    
    # Function call to extract VV,VH Values
    #time_slice = "2020-01-01/2020-12-31"
    #assests = ['vh','vv']
    #list_v = []
    #for coordinates in tqdm(crop_presence_data['Latitude and Longitude']):
       #v=get_sentinel_data(coordinates,time_slice,assests)
       #list_v.append(v)
    #list_v = np.transpose(pd.DataFrame(list_v))
    
    
    
    
    
    #Use this box if you have already saved the extracted time series in a csv file
    
    #list_v.to_csv('list_vh+vv_mean10x10.csv', index=False)
    list_v = pd.read_csv('list_vh+vv_mean10x10.csv')
    #list_v
    
    
    # ## Feature extraction
    # 
    # This step constitute the main work of the project. A function (named get_new_features) was created to compute basic and more complex characteristics of the time series in order to better classify them. Selected features include the mean, the standard deviation, the absolute energy, the median, the amplitude, the maximum, the minimum, the coefficient of variation (CV), the autocorrelation lag 1 and the mean value differences.
    

    
    list_vf=get_new_features(list_v,['mean','std','median','max','min','CV', 'absolute energy'])
    #list_vf
    
    pd.set_option('display.max_rows', 10)
    #list_vf
    
    
    # ## Anomaly detection
    # By comparing to the submission dataset values, some of the training set values reach very large values. An attempt to remove those 'anomalies' has been done in order to be closer to the final submission dataset, but this has not improved the accuracy.

    #For best score, do not execute
    
    # list_vf, crop_presence_data = anomaly_detection(list_vf, crop_presence_data, 70)
    
 
    
    
    #Combine Latitude, Longitude and VV/VH
    crop_data = combine_two_datasets(crop_presence_data,list_vf)
    

    
    
    #Model building
    #crop_data = crop_data[['vh','vv','Class of Land']]
    X = crop_data.drop(columns=['Class of Land','Latitude and Longitude']).values
    y = crop_data ['Class of Land'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y,random_state=50)
    
    
    # ## Feature Scaling
    # Various scaling methods have been tested, including the StandardScaler, MinMaxScaler, MaxAbsScaler, and RobustScaler. The scaling method with the best results was the StandardScaler. 

    
    #Feature scaling 1
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
    
    # ## Classification model selection
    # 
    # Several classification models have been tested, but the Random Forest Classifier has given the best results so far.
    
    
    from sklearn.ensemble import RandomForestClassifier
    model=RandomForestClassifier(n_estimators=400)
    model.fit(X_train,y_train)
    
    #Save model with pickle
    filename = 'Leckert_Jack_RFC.pkl'
    pickle.dump(model, open(filename, 'wb'))
 
 
    # load the model from disk
    model = pickle.load(open(filename, 'rb'))
    
  
    
    
    #Out sample evaluation
    outsample_predictions = model.predict(X_test)
    print("Accuracy {0:.2f}%".format(100*accuracy_score(outsample_predictions, y_test)))
    print(classification_report(y_test, outsample_predictions))
    plot_confusion_matrix(y_test, outsample_predictions,"Model Level 1: Logistic\nRegression Model Out-Sample Results",['Rice', 'Non Rice'])
    
    
    
    
    #Submission test 
    test_file = pd.read_csv('challenge_1_submission_template_correct_columns_fixed.csv')
    #test_file
    
    
    
    
    ## Get Sentinel-1-RTC Data
    # time_slice = "2020-01-01/2020-12-31"
    # assests = ['vh','vv']
    # list_vs = []
    # for coordinates in tqdm(test_file['id']):
    #     vs=get_sentinel_data(coordinates,time_slice,assests)
    #     list_vs.append(vs)
    # submission_v_data = np.transpose(pd.DataFrame(list_vs))
    

    
    
    #Use this if you have already saved the extracted time series in a csv file
    
    #submission_v_data.to_csv('submission_vh+vv_ampli_mean10x10.csv', index=False)
    submission_v_data= pd.read_csv('submission_vh+vv_ampli_mean10x10.csv')
    #submission_v_data
    
    
    
    
    
    submission_v_data_f=get_new_features(submission_v_data,['mean','std','median','max','min','CV','absolute energy'])
    #submission_v_data_f
 
    
    
    # Feature Scaling 
    transformed_submission_data = sc.transform(submission_v_data_f)
    
    #Making predictions
    final_predictions = model.predict(transformed_submission_data)
    final_prediction_series = pd.Series(final_predictions)
        
    #Combining the results into dataframe
    submission_df = pd.DataFrame({'id':test_file['id'].values, 'target':final_prediction_series.values})
    #Displaying the sample submission dataframe
    display(submission_df)
    
    

    #Submission file
    #Dumping the predictions into a csv file.
    submission_df.to_csv("challenge_1_submission_rice_crop_prediction.csv",index = False)
    
    

    
    
    
    
