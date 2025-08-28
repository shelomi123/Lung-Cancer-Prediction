import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image
from sklearn import datasets


#data = pd.read_csv('/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/lung cancer data.csv')
#data.head()

st.title('Predicting Lung Cancer')

image = Image.open('/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/BIProject/lung.jpg')
st.image(image,use_column_width=True)

st.write(
        # data app with streamlit 
        )

st.write(
        # different classifiers and datasets 
        )

dataset_name=st.sidebar.selectbox('Select datasets',('Breast Cancer','Iris','Wine'))

classifier_=st.sidebar.selectbox('Select classifier',('SVM','KNN'))
#data = pd.read_csv('/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/lung cancer data.csv')

def get_dataset(name):
    data=None
    data=data = pd.read_csv('/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/lung cancer data.csv')

    x= data.data
    y= data.target
    
    return x,y

x,y=get_dataset(dataset_name)

st.dataframe(x)

st.write('Shape of your dataset',x.shape)
st.write('unique target variables:', len(np.unique(y)))

st.write("Data columns:", x.columns)
st.write("Selected feature columns:", x)
st.write("Input DataFrame columns:", x.columns)
