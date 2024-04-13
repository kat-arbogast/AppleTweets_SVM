'''
Author: Katrina Arbogast

This script models an SVM for text classification. 

The dataset I will employ consists of tweets related to Apple Inc., collected from various sources and then 
aggregated [2]. These tweets will serve as the basis for sentiment analysis, where the SVM models will
predict the target classifications (positive, negative, or neutral sentiments) regarding Apple products,
services, or events.
'''

import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
import seaborn as sns
import random as rd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.svm import LinearSVC

from collections import Counter

#-------------------------------------------------------------------------
## Setting the folder path to the cleaned but not formatted data
appleTweets_folder = "./Data/appleTweets_vectorized.csv"

## Setting the filename
appleTweets_filename = "AppleTweets"
#-------------------------------------------------------------------------

def main():
    '''
    This is the main function for the decision tree modeling and visualizations on wildfire data
    '''
    
    print("\n ---------- Ingesting Data ---------- \n")
    appleTweetsdf = pd.read_csv(appleTweets_folder)
    
    # print("\n ---------- Reformatting Some of the Target Labels ---------- \n")
    
   
    print("\n\n ---------- Selecting Train and Test Data ---------- \n")
    appleTweets_sample_dict = setup_train_test_data(appleTweetsdf, label_col="labels")
    
    print("\n\n ---------- SVM  ---------- \n")
    run_svm(appleTweets_sample_dict, appleTweets_filename)
    
    # print("\n\n ---------- Further Visuals ---------- \n")
    # label_counts(fires_season_dict_pred, f"{us_fires_burn_monthly_filename}_season")


def setup_train_test_data(df, label_col, cols_of_interst_plus_label=None, test_size=0.2, seed_val=1):

    if cols_of_interst_plus_label is None:
        df2 = df.copy()
    else:
        df2 = df[cols_of_interst_plus_label]
    
    rd.seed(seed_val)
    train_df, test_df, train_labels, test_labels = train_test_split(
        df2.drop(label_col, axis=1), 
        df2[label_col], 
        test_size=test_size, 
        stratify=df2[label_col])
    
    sample_dict = {
        "train_labels" : train_labels,
        "train_df" : train_df,
        "test_labels" : test_labels,
        "test_df" : test_df
    }
    
    return sample_dict


def run_svm(sample_dict, filename, visual_folder="./CreatedVisuals/svm"):
    
    ## Data
    train_labels = sample_dict["train_labels"]
    train_df = sample_dict["train_df"]
    test_labels = sample_dict['test_labels']
    test_df = sample_dict['test_df']
    

    SVM_Model1=LinearSVC(C=50, dual="auto")
    SVM_Model1.fit(train_df, train_labels)
    
    score = accuracy_score(test_labels, SVM_Model1.predict(test_df))
    print(f"\nThe accurary is for Linear SVM {filename} is : {score}\n")

    SVM_matrix = confusion_matrix(test_labels, SVM_Model1.predict(test_df))
    print("\nThe confusion matrix for Linear SVM is:")
    print(SVM_matrix)
    print("\n\n")
    
    print(f"Making linear visual for {filename}")
    disp = ConfusionMatrixDisplay(confusion_matrix=SVM_matrix, display_labels=SVM_Model1.classes_)
    plt.figure(figsize=(18, 15))
    disp.plot(cmap='magma')
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Confusion Matrix Linear SVM\n- {filename} -\nAccuracy = {score}") 
    plt.tight_layout()
    plt.savefig(f"{visual_folder}/{filename}_linear_svm_cm.png", dpi=300)
    plt.close()
    
    
    #--------------other kernels
    ## RBF
    print("--- Starting RBF ---")
    SVM_Model2=sklearn.svm.SVC(C=1.0, kernel='rbf', gamma="auto")
    SVM_Model2.fit(train_df, train_labels)

    score = accuracy_score(test_labels, SVM_Model2.predict(test_df))
    print(f"\nThe accurary is for rbf SVM {filename} is : {score}\n")

    
    SVM_matrix = confusion_matrix(test_labels, SVM_Model2.predict(test_df))
    print("\nThe confusion matrix for rbf SVM is:")
    print(SVM_matrix)
    print("\n\n")
    
    print(f"Making rbf visual for {filename}")
    disp = ConfusionMatrixDisplay(confusion_matrix=SVM_matrix, display_labels=SVM_Model1.classes_)
    plt.figure(figsize=(18, 15))
    disp.plot(cmap='magma')
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Confusion Matrix RBF SVM\n- {filename} \nAccuracy = {score}-") 
    plt.tight_layout()
    plt.savefig(f"{visual_folder}/{filename}_rbf_svm_cm.png", dpi=300)
    plt.close()


    for d in range(1, 5): # [1, 5, 10, 20, 50]:
        ## POLY
        print("--- Starting Poly ---")
        SVM_Model3=sklearn.svm.SVC(C=1, kernel='poly', degree=d, gamma="scale")
        SVM_Model3.fit(train_df, train_labels)
        
        score = accuracy_score(test_labels, SVM_Model3.predict(test_df))
        print(f"\nThe accurary is for poly {d} SVM {filename} is : {score}\n")

        SVM_matrix = confusion_matrix(test_labels, SVM_Model3.predict(test_df))
        print(f"\nThe confusion matrix for poly d = {d} and C = 1 SVM is:")
        print(SVM_matrix)
        print("\n\n")
        
        print(f"Making poly visual for {filename} - {d}")
        disp = ConfusionMatrixDisplay(confusion_matrix=SVM_matrix, display_labels=SVM_Model1.classes_)
        plt.figure(figsize=(18, 15))
        disp.plot(cmap='magma')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Confusion Matrix Poly SVM - {d}\n - {filename} -\nAccuracy = {score}") 
        plt.tight_layout()
        plt.savefig(f"{visual_folder}/{filename}_poly_svm_cm_{d}.png", dpi=300)
        plt.close()


# DO NOT REMOVE!!!
if __name__ == "__main__":
    main()