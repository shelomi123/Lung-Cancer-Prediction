import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from keras.models import load_model
import webbrowser

# Set page configuration
st.set_page_config(page_title="Cancer Prediction System", layout="wide")

# Load the trained model and scaler
model = load_model('/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/App/my_model.h5')
with open('/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/App/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'show_result' not in st.session_state:
    st.session_state['show_result'] = False

# Function to handle navigation
def navigate(page):
    st.session_state['page'] = page
    st.session_state['show_result'] = False

# Custom CSS to style only the navigation buttons with a yellow background and non-curved rectangles
st.markdown(
    """
    <style>
        /* Header styling */
        .header {
            background-color: #23395d;
            padding: 15px;
            color: white;
            font-size: 24px;
            text-align: center;
            font-weight: bold;
        }

        /* Button styling */
        .stButton>button {
            background-color: #f0c040 !important;
            color: #23395d !important;
            font-size: 18px !important;
            font-weight: bold !important;
            border: none !important;
            border-radius: 0 !important; /* Non-curved, rectangular */
            padding: 10px 20px !important;
            margin: 5px 0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown("<div class='header'>Cancer Prediction System</div>", unsafe_allow_html=True)

# Sidebar with navigation buttons
with st.sidebar:
    st.markdown("<h2>Navigation</h2>", unsafe_allow_html=True)
    if st.button("Home"):
        navigate("home")
    if st.button("Services"):
        navigate("services")
    if st.button("Blog"):
        navigate("blog")
    if st.button("About"):
        navigate("about")
    if st.button("FAQ"):
        navigate("faq")


# Function to reset form and go back to input page
def reset():
    st.session_state['show_result'] = False

# Function to open Google Maps for nearby medical centers
def consult_doctor():
    url = "https://www.google.com/maps/search/medical+centers+near+me"
    webbrowser.open_new_tab(url)

# Function to make prediction and go to result page
def predict():
    chronic_lung_disease = 1 if st.session_state['chronic_lung_disease'] == "Yes" else 0
    gender = 1 if st.session_state['gender'] == "Male" else 0
    input_data = np.array([[
        st.session_state['air_pollution'], st.session_state['weight_loss'], st.session_state['alcohol_use'], 
        st.session_state['passive_smoker'], st.session_state['age'], chronic_lung_disease, 
        st.session_state['fatigue'], st.session_state['frequent_cold'], st.session_state['dry_cough'], gender
    ]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    
    # Calculate the likelihood percentage for "High Risk" of lung cancer
    high_risk_percentage = prediction[0][2] * 100  # Confidence score for "High Risk" class

    # Interpret the result
    predicted_class = np.argmax(prediction, axis=1)[0]
    label_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    st.session_state['result'] = label_mapping.get(predicted_class, "Unknown")
    st.session_state['high_risk_percentage'] = high_risk_percentage
    st.session_state['show_result'] = True

# Display content based on selected page
if st.session_state['page'] == 'home':
    if not st.session_state['show_result']:
        # Layout using columns to align form and image side by side
        col1, col2 = st.columns([2, 1])

        # Left column: Input form
        with col1:
            st.markdown("<div class='input-form'><h2>Enter Patient Details</h2>", unsafe_allow_html=True)
            st.session_state['air_pollution'] = st.slider("Air Pollution Level (0-10)", 0, 10, 5)
            st.session_state['weight_loss'] = st.slider("Weight Loss Level (0-10)", 0, 10, 5)
            st.session_state['alcohol_use'] = st.slider("Alcohol Use Level (0-10)", 0, 10, 5)
            st.session_state['passive_smoker'] = st.slider("Passive Smoker Level (0-10)", 0, 10, 5)
            st.session_state['age'] = st.number_input("Age", min_value=0, max_value=120, value=30)
            st.session_state['chronic_lung_disease'] = st.selectbox("Chronic Lung Disease", ["No", "Yes"])
            st.session_state['fatigue'] = st.slider("Fatigue Level (0-10)", 0, 10, 5)
            st.session_state['frequent_cold'] = st.slider("Frequent Cold Level (0-10)", 0, 10, 5)
            st.session_state['dry_cough'] = st.slider("Dry Cough Level (0-10)", 0, 10, 5)
            st.session_state['gender'] = st.selectbox("Gender", ["Male", "Female"])
            st.button("Predict", on_click=predict)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Right column: Image with container for styling
        with col2:
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image("/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/App/lung.jpg", use_column_width=True)  # Replace with actual image path
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Prediction result page
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"<h2>Prediction Result: {st.session_state['result']}</h2>", unsafe_allow_html=True)
            st.write(f"### Estimated Risk of Developing Lung Cancer: {st.session_state['high_risk_percentage']:.2f}%")
            
            # Back button
            st.button("Back", on_click=reset)
            
            # Consult a Doctor button
            if st.button("Consult a Doctor in Your Area"):
                consult_doctor()

        with col2:
            st.image("/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/App/lung.jpg", use_column_width=True)  # Replace with actual image path

elif st.session_state['page'] == 'services':
    st.markdown("<h2>Our Services</h2>", unsafe_allow_html=True)
    st.markdown("<div class='service-bar'>Predicting the risk of lung cancer</div>", unsafe_allow_html=True)
    st.markdown("<div class='service-bar'>Guiding patients to the nearest medical center</div>", unsafe_allow_html=True)
    st.markdown("<div class='service-bar'>Providing knowledge on lung cancer</div>", unsafe_allow_html=True)

elif st.session_state['page'] == 'blog':
    st.markdown("<h2>Blog</h2>", unsafe_allow_html=True)
    st.write("Visit our blog at [Lung Cancer Blogs](https://www.verywellhealth.com/lung-cancer-blogs-2249220)")

elif st.session_state['page'] == 'about':
    st.markdown("<h2>About This App</h2>", unsafe_allow_html=True)
    st.markdown("<p class='about-text'>This application is designed to help users assess their risk of lung cancer and guide them to resources for further assistance.</p>", unsafe_allow_html=True)

elif st.session_state['page'] == 'faq':
    st.markdown("<h2>FAQ</h2>", unsafe_allow_html=True)
    st.write("Here you can find answers to frequently asked questions.")
