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
            width : 103%;
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

# Navigation bar with functional buttons
with st.container():
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("Home"):
            navigate("home")
    with col2:
        if st.button("Services"):
            navigate("services")
    with col3:
        if st.button("Blog"):
            navigate("blog")
    with col4:
        if st.button("About"):
            navigate("about")
    with col5:
        if st.button("FAQ"):
            navigate("faq")

# The rest of your code for page content and other elements can stay the same


# The rest of your code for page content and other elements can stay the same

# The rest of your code for each page content can remain the same, and you can continue to use `st.session_state` for page navigation
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
    
    st.markdown("### Understanding Lung Cancer")
    st.image("/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/App/images/1.png", width=600)
    st.write("Learn about the types, stages, and symptoms of lung cancer. This article provides a comprehensive guide for understanding lung cancer, which may help you interpret the results from our prediction system.")
    st.markdown("[Read more](https://www.verywellhealth.com/")

    st.markdown("### Lifestyle Changes to Reduce Cancer Risk")
    st.image("/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/App/images/2.jpg", width=600)
    st.write("While some risk factors for cancer are unavoidable, lifestyle changes like quitting smoking, reducing alcohol consumption, and adopting a balanced diet can help lower your risk. This article covers effective strategies for healthier living.")
    st.markdown("[Read more](https://www.cancer.org/)")

    st.markdown("### The Role of Air Pollution in Lung Cancer")
    st.image("/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/App/images/3.jpg", width=600)
    st.write("Air pollution is a significant factor in respiratory diseases and lung cancer. This article discusses how pollutants impact lung health and contribute to the development of lung cancer.")
    st.markdown("[Read more](https://www.cancer.gov")

    st.markdown("### How to Support a Loved One with Cancer")
    st.image("/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/App/images/4.jpg", width=600)
    st.write("If someone close to you has been diagnosed with cancer, it's essential to know how to provide emotional and practical support. This article offers guidance on how to be there for a loved one during this challenging time.")
    st.markdown("[Read more](https://www.cancercare.org/")

    st.markdown("### Innovations in Lung Cancer Treatment")
    st.image("/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/App/images/5.jpg", width=600)
    st.write("Explore the latest advancements in lung cancer treatments, including immunotherapy, targeted therapy, and other cutting-edge approaches. This article explains how these treatments work and who may benefit from them.")
    st.markdown("[Read more](https://www.cancercenter.com/")

elif st.session_state['page'] == 'about':
    st.markdown("<h2>About This App</h2>", unsafe_allow_html=True)
    st.markdown("<p class='about-text'>This application is designed to help users assess their risk of lung cancer and guide them to resources for further assistance.</p>", unsafe_allow_html=True)

elif st.session_state['page'] == 'faq':
    st.markdown("<h2>FAQ</h2>", unsafe_allow_html=True)
    
    st.markdown("### 1. What is the purpose of this Cancer Prediction System?")
    st.write("This app is designed to help users assess their risk level of developing lung cancer based on several factors such as age, air pollution exposure, smoking habits, and more. It provides a quick and easy way for individuals to understand their risk, but it does not replace professional medical advice.")

    st.markdown("### 2. How accurate is this prediction?")
    st.write("The predictions provided by this app are based on a trained machine learning model, but they are not 100% accurate. The model has limitations and may not consider all unique health conditions. It is always recommended to consult with a medical professional for an accurate diagnosis.")

    st.markdown("### 3. What data do I need to provide for the prediction?")
    st.write("You will need to enter information such as age, levels of air pollution exposure, smoking status, alcohol use, presence of chronic lung disease, and other factors that can influence lung cancer risk.")

    st.markdown("### 4. Can this app diagnose lung cancer?")
    st.write("No, this app cannot diagnose lung cancer. It only provides a risk assessment based on input data. A medical diagnosis can only be made by a healthcare professional.")

    st.markdown("### 5. Is my data safe?")
    st.write("This app does not store or share any data entered by users. All information is processed within the app to generate a prediction and is not saved or transmitted.")

    st.markdown("### 6. Should I rely on this app for making health decisions?")
    st.write("No, this app should not be solely relied upon for making health decisions. It is meant to raise awareness of potential risk factors. For medical decisions, please consult a healthcare provider.")

    st.markdown("### 7. How is the risk level calculated?")
    st.write("The risk level is calculated using a machine learning model that has been trained on a dataset with similar features. The model evaluates each input feature, processes it, and outputs a risk category: Low, Medium, or High.")

    st.markdown("### 8. What should I do if my risk is high?")
    st.write("If your risk level is high, it is advisable to consult with a medical professional for further testing and advice. High risk does not necessarily mean you have cancer, but it indicates that you may want to seek professional guidance.")

    st.markdown("### 9. Can I consult a doctor through this app?")
    st.write("This app provides a link to find nearby medical centers. However, it does not offer direct consultation services. You can use the link to locate a medical facility or professional in your area.")

    st.markdown("### 10. Is this app free to use?")
    st.write("Yes, this app is free to use for risk assessment purposes.")
