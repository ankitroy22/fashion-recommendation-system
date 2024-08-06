import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2

# Ensure 'uploads' directory exists
os.makedirs('uploads', exist_ok=True)

# Load feature list and filenames
feature_list = np.array(pickle.load(open('embedding.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Define the model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tensorflow.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def extract_features(img_path, model):
    try:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = np.array(img)
        expand_img = np.expand_dims(img, axis=0)
        pre_img = preprocess_input(expand_img)
        result = model.predict(pre_img).flatten()
        normalized = result / norm(result)
        return normalized
    except Exception as e:
        st.error(f"Error in feature extraction: {e}")
        return None

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        
        # Feature extraction
        features = extract_features(os.path.join("uploads", uploaded_file.name), model)
        if features is not None:
            indices = recommend(features, feature_list)
            
            # Show recommendations
            col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occurred in file upload")
