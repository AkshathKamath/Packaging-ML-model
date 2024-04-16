import pandas as pd
import numpy as np 
from config import config  
from processing.data_handling import load_dataset,save_pipeline
import processing.pre_processing as pp 
import pipeline as pipe 
import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

def model_training_saving():
    df_train = load_dataset(config.TRAIN_FILE)
    y_train = df_train[config.TARGET]
    y_train = y_train.map({'N':0, 'Y':1})
    pipe.classification_pipeline.fit(df_train[config.FEATURES], y_train)
    save_pipeline(pipe.classification_pipeline)

if __name__=='__main__': ## For automating
    model_training_saving()