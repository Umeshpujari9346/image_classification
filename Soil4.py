# import pickle
# import numpy as np
# import cv2
# import streamlit as st
# import plotly.express as px
# import pandas as pd
# import os 
# from sklearn.metrics import accuracy_score


# # Define categories (soil types)
# SOIL_CATEGORIES = ["Alluvial soil", "Black Soil", "Cinder Soil", "Clay soil", "Laterite Soil", "Peat Soil", "Red soil", "Yellow Soil"]

# # Soil information for descriptions and recommendations
# SOIL_INFO = {
#     "Alluvial soil": {"desc": "Fertile soil found near riverbanks, rich in nutrients.", "rec": "Ideal for rice, wheat, and sugarcane."},
#     "Black Soil": {"desc": "High clay content, retains moisture well.", "rec": "Great for cotton and cereals."},
#     "Cinder Soil": {"desc": "Volcanic soil, porous and well-draining.", "rec": "Suitable for root crops like potatoes."},
#     "Clay soil": {"desc": "Dense and sticky, retains water.", "rec": "Good for rice with proper drainage."},
#     "Laterite Soil": {"desc": "Rich in iron, poor in nutrients.", "rec": "Best for tea, coffee with fertilization."},
#     "Peat Soil": {"desc": "Organic-rich, acidic soil.", "rec": "Good for berries and acid-loving plants."},
#     "Red soil": {"desc": "Rich in iron oxides, well-drained.", "rec": "Suitable for millets and groundnuts."},
#     "Yellow Soil": {"desc": "Loamy with moderate fertility.", "rec": "Supports vegetables and orchards."}
# }


# # Load trained model and scaler
# @st.cache_resource
# def load_model_and_scaler():
#     try:
#         with open("soil_quality_rf.pkl", "rb") as f:
#             model = pickle.load(f)
#     except Exception as e:
#         st.error(f"Error loading soil_quality_rf.pkl: {e}")
#         return None, None
    
#     try:
#         with open("scaler.pkl", "rb") as f:
#             scaler = pickle.load(f)
#     except Exception as e:
#         st.error(f"Error loading scaler.pkl: {e}")
#         return model, None
    
#     return model, scaler

# model, scaler = load_model_and_scaler()

# def predict_soil(image):
#     if model is None or scaler is None:
#         st.error("Model or scaler not loaded. Check the files.")
#         return None, None, None
    
#     img = cv2.resize(image, (128, 128))
#     img = img.flatten().reshape(1, -1)  # Convert to 1D feature vector
#     img = scaler.transform(img)  # Apply standardization
#     prediction = model.predict(img)[0]
#     probs = model.predict_proba(img)[0]
#     return SOIL_CATEGORIES[prediction], probs[prediction], probs

# # Streamlit UI with Navigation Bar
# st.title("🌱 Soil Classification App")

# # Create tabs for navigation
# tabs = st.tabs(["Home", "Classify Soil (Upload)", "Classify Soil (Webcam)", "Bulk Classification", "Dataset Info", "Model Accuracy"])

# # Tab 1: Home
# with tabs[0]:
#     st.header("Welcome to the Soil Classification App")
#     st.write("""
#     This app helps you classify soil types using machine learning. Navigate through the tabs to:
#     - **Classify Soil (Upload):** Upload a single soil image to classify its type.
#     - **Classify Soil (Webcam):** Use your webcam to capture and classify a soil image.
#     - **Bulk Classification:** Upload multiple images and download the results.
#     - **Dataset Info:** Learn about the dataset used to train the model.
#     - **Model Accuracy:** View the model's performance metrics.
#     """)

# # Tab 2: Classify Soil (Upload)
# with tabs[1]:
#     st.header("Classify Soil (Upload)")
#     st.write("Upload an image of soil to determine its type.")
#     uploaded_file = st.file_uploader("Choose a soil image...", type=["jpg", "jpeg", "png"], key="single_upload")
    
#     if uploaded_file is not None:
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
#         if img is None:
#             st.error("Invalid image file. Please upload a valid image.")
#         else:
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             st.image(img_rgb, caption="Uploaded Soil Image")
            
#             soil_type, confidence, all_probs = predict_soil(img)
            
#             if soil_type:
#                 st.subheader("Results")
#                 st.write(f"**Soil Type:** {soil_type} (Confidence: {confidence:.2f})")
#                 st.write(f"**Description:** {SOIL_INFO[soil_type]['desc']}")
#                 st.write(f"**Recommendations:** {SOIL_INFO[soil_type]['rec']}")
#                 fig = px.bar(x=SOIL_CATEGORIES, y=all_probs, labels={"x": "Soil Type", "y": "Probability"})
#                 st.plotly_chart(fig)

# # Tab 3: Classify Soil (Webcam)
# with tabs[2]:
#     st.header("Classify Soil (Webcam)")
#     st.write("Use your webcam to capture a soil image and classify its type.")
#     camera_file = st.camera_input("Take a picture of the soil")
    
#     if camera_file is not None:
#         file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
#         if img is None:
#             st.error("Invalid camera input.")
#         else:
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             st.image(img_rgb, caption="Captured Soil Image", use_container_width=True)
            
#             soil_type, confidence, all_probs = predict_soil(img)
            
#             if soil_type:
#                 st.subheader("Results")
#                 st.write(f"**Soil Type:** {soil_type} (Confidence: {confidence:.2f})")
#                 st.write(f"**Description:** {SOIL_INFO[soil_type]['desc']}")
#                 st.write(f"**Recommendations:** {SOIL_INFO[soil_type]['rec']}")
#                 fig = px.bar(x=SOIL_CATEGORIES, y=all_probs, labels={"x": "Soil Type", "y": "Probability"})
#                 st.plotly_chart(fig)

# # Tab 4: Bulk Classification
# with tabs[3]:
#     st.header("Bulk Classification")
#     st.write("Upload multiple soil images to classify them and download the results.")
#     uploaded_files = st.file_uploader("Choose soil images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="bulk_upload")
    
#     if uploaded_files:
#         results = []
#         for uploaded_file in uploaded_files:
#             file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#             img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
#             if img is None:
#                 st.error(f"Invalid image file: {uploaded_file.name}")
#             else:
#                 img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 st.image(img_rgb, caption=f"Uploaded Soil Image: {uploaded_file.name}", use_container_width=True)
#                 soil_type, confidence, _ = predict_soil(img)
#                 if soil_type:
#                     st.write(f"**File:** {uploaded_file.name}")
#                     st.write(f"**Soil Type:** {soil_type} (Confidence: {confidence:.2f})")
#                     results.append({"File": uploaded_file.name, "Soil Type": soil_type, "Confidence": confidence})
        
#         if results:
#             df = pd.DataFrame(results)
#             csv = df.to_csv(index=False)
#             st.download_button("Download Results", csv, "soil_classification_results.csv", "text/csv")

# # Tab 5: Dataset Info
# with tabs[4]:
#     st.header("Dataset Information")
#     st.write("The model was trained on a dataset containing images of the following soil types:")
#     for category in SOIL_CATEGORIES:
#         st.write(f"- **{category}**: {SOIL_INFO[category]['desc']}")
#     st.write("Each soil type folder contains images resized to 128x128 pixels and flattened into feature vectors for training.")

# # Tab 6: Model Accuracy
# # with tabs[5]:
# #     st.header("Model Accuracy")
# #     st.write("Accuracy score is:", f"{accuracy_score(y_test, rf_model.predict(X_test)) * 100:.2f}%")
# with tabs[5]:
#     st.header("Model Accuracy")
#     st.write("The Random Forest model's accuracy is precomputed and not available in this deployed version. Please run the training script locally to compute accuracy.")
import pickle
import numpy as np
import cv2
import streamlit as st
import plotly.express as px
import pandas as pd
import os

# Define categories (soil types)
SOIL_CATEGORIES = ["Alluvial soil", "Black Soil", "Cinder Soil", "Clay soil", "Laterite Soil", "Peat Soil", "Red soil", "Yellow Soil"]

# Soil information for descriptions and recommendations
SOIL_INFO = {
    "Alluvial soil": {"desc": "Fertile soil found near riverbanks, rich in nutrients.", "rec": "Ideal for rice, wheat, and sugarcane."},
    "Black Soil": {"desc": "High clay content, retains moisture well.", "rec": "Great for cotton and cereals."},
    "Cinder Soil": {"desc": "Volcanic soil, porous and well-draining.", "rec": "Suitable for root crops like potatoes."},
    "Clay soil": {"desc": "Dense and sticky, retains water.", "rec": "Good for rice with proper drainage."},
    "Laterite Soil": {"desc": "Rich in iron, poor in nutrients.", "rec": "Best for tea, coffee with fertilization."},
    "Peat Soil": {"desc": "Organic-rich, acidic soil.", "rec": "Good for berries and acid-loving plants."},
    "Red soil": {"desc": "Rich in iron oxides, well-drained.", "rec": "Suitable for millets and groundnuts."},
    "Yellow Soil": {"desc": "Loamy with moderate fertility.", "rec": "Supports vegetables and orchards."}
}

# Load trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        with open("soil_quality_rf.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading soil_quality_rf.pkl: {e}")
        return None, None
    
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading scaler.pkl: {e}")
        return model, None
    
    return model, scaler

model, scaler = load_model_and_scaler()

def predict_soil(image):
    if model is None or scaler is None:
        st.error("Model or scaler not loaded. Check the files.")
        return None, None, None
    
    img = cv2.resize(image, (128, 128))
    img = img.flatten().reshape(1, -1)  # Convert to 1D feature vector
    img = scaler.transform(img)  # Apply standardization
    prediction = model.predict(img)[0]
    probs = model.predict_proba(img)[0]
    return SOIL_CATEGORIES[prediction], probs[prediction], probs

# Streamlit UI with Navigation Bar
st.title("🌱 Soil Classification App")

# Create tabs for navigation
tabs = st.tabs(["Home", "Classify Soil (Upload)", "Classify Soil (Webcam)", "Bulk Classification", "Dataset Info", "Model Accuracy"])

# Tab 1: Home
with tabs[0]:
    st.header("Welcome to the Soil Classification App")
    st.write("""
    This app helps you classify soil types using machine learning. Navigate through the tabs to:
    - **Classify Soil (Upload):** Upload a single soil image to classify its type.
    - **Classify Soil (Webcam):** Use your webcam to capture and classify a soil image.
    - **Bulk Classification:** Upload multiple images and download the results.
    - **Dataset Info:** Learn about the dataset used to train the model.
    - **Model Accuracy:** View the model's performance metrics.
    """)

# Tab 2: Classify Soil (Upload)
with tabs[1]:
    st.header("Classify Soil (Upload)")
    st.write("Upload an image of soil to determine its type.")
    uploaded_file = st.file_uploader("Choose a soil image...", type=["jpg", "jpeg", "png"], key="single_upload")
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error("Invalid image file. Please upload a valid image.")
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Uploaded Soil Image")  # Removed use_container_width
            
            soil_type, confidence, all_probs = predict_soil(img)
            
            if soil_type:
                st.subheader("Results")
                st.write(f"**Soil Type:** {soil_type} (Confidence: {confidence:.2f})")
                st.write(f"**Description:** {SOIL_INFO[soil_type]['desc']}")
                st.write(f"**Recommendations:** {SOIL_INFO[soil_type]['rec']}")
                fig = px.bar(x=SOIL_CATEGORIES, y=all_probs, labels={"x": "Soil Type", "y": "Probability"})
                st.plotly_chart(fig)

# Tab 3: Classify Soil (Webcam)
with tabs[2]:
    st.header("Classify Soil (Webcam)")
    st.write("Use your webcam to capture a soil image and classify its type.")
    camera_file = st.camera_input("Take a picture of the soil")
    
    if camera_file is not None:
        file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error("Invalid camera input.")
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Captured Soil Image")  # Removed use_container_width
            
            soil_type, confidence, all_probs = predict_soil(img)
            
            if soil_type:
                st.subheader("Results")
                st.write(f"**Soil Type:** {soil_type} (Confidence: {confidence:.2f})")
                st.write(f"**Description:** {SOIL_INFO[soil_type]['desc']}")
                st.write(f"**Recommendations:** {SOIL_INFO[soil_type]['rec']}")
                fig = px.bar(x=SOIL_CATEGORIES, y=all_probs, labels={"x": "Soil Type", "y": "Probability"})
                st.plotly_chart(fig)

# Tab 4: Bulk Classification
with tabs[3]:
    st.header("Bulk Classification")
    st.write("Upload multiple soil images to classify them and download the results.")
    uploaded_files = st.file_uploader("Choose soil images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="bulk_upload")
    
    if uploaded_files:
        results = []
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                st.error(f"Invalid image file: {uploaded_file.name}")
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption=f"Uploaded Soil Image: {uploaded_file.name}")  # Removed use_container_width
                soil_type, confidence, _ = predict_soil(img)
                if soil_type:
                    st.write(f"**File:** {uploaded_file.name}")
                    st.write(f"**Soil Type:** {soil_type} (Confidence: {confidence:.2f})")
                    results.append({"File": uploaded_file.name, "Soil Type": soil_type, "Confidence": confidence})
        
        if results:
            df = pd.DataFrame(results)
            csv = df.to_csv(index=False)
            st.download_button("Download Results", csv, "soil_classification_results.csv", "text/csv")

# Tab 5: Dataset Info
with tabs[4]:
    st.header("Dataset Information")
    st.write("The model was trained on a dataset containing images of the following soil types:")
    for category in SOIL_CATEGORIES:
        st.write(f"- **{category}**: {SOIL_INFO[category]['desc']}")
    st.write("Each soil type folder contains images resized to 128x128 pixels and flattened into feature vectors for training.")

# Tab 5: Model Accuracy
with tabs[5]:
    st.header("Model Accuracy")
    st.write("The Random Forest model's accuracy is precomputed and not available in this deployed version. Please run the training script locally to compute accuracy.")
