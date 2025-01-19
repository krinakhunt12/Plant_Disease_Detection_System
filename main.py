import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions), predictions[0]  # Return index of max element and the prediction probabilities

# Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "ABOUT", "DISEASE RECOGNITION", "FAQ", "FEEDBACK", "CONTACT US"])

# Display an introductory image
img = Image.open("Diseases.png")
st.image(img)

# Main Page (HOME)
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)
    st.markdown("""
    Welcome to the **Plant Disease Recognition System**! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. With early identification, we can mitigate losses and ensure a healthier harvest. Together, let's protect our crops and preserve sustainable agriculture!

    ### Why is Early Disease Detection Important?
    Early disease detection is critical to protecting plants and crops from various diseases that can significantly reduce agricultural yield. By quickly diagnosing plant diseases, we can:
    - Minimize crop damage
    - Implement timely treatments (like pesticides or fungicides)
    - Increase crop production efficiency
    - Ensure food security and sustainable farming practices

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced machine learning models to identify potential diseases.
    3. **Results:** View the results and recommended actions based on the diagnosis.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for highly accurate disease detection.
    - **User-Friendly:** A simple, intuitive interface designed for both professionals and beginners in agriculture.
    - **Fast and Efficient:** Get results in seconds, allowing for quick decision-making and intervention.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about our project, our team, and our goals on the **About** page.
    """)

# About Page
elif app_mode == "ABOUT":
    st.header("About")
    st.markdown("""
                 #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this [GitHub repo](https://github.com). It consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The dataset is divided into training, validation, and test sets, which are used to train and test the model.

                #### Key Features:
                - **Diversity of plants**: Our model can recognize a variety of plant diseases.
                - **Large dataset**: The dataset contains 87,000 images, ensuring robust training and validation.
                - **Augmented Data**: To avoid overfitting, offline augmentation techniques like rotation, flipping, and scaling were applied to the images.

                #### Dataset Breakdown:
                1. **Train (70,295 images)**
                2. **Test (33 images)**
                3. **Validation (17,572 images)**
                4. **Prediction (33 test images)**

                #### How the Model Works:
                The model is a Convolutional Neural Network (CNN) trained on plant images to detect diseases based on visible patterns. CNNs are well-suited for image classification tasks due to their ability to learn spatial hierarchies of features.
    """)

# Disease Recognition Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

        # Predict button
        if st.button("Predict"):
            st.snow()
            result_index, prediction_probabilities = model_prediction(test_image)
            
            # Reading Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
                          'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
                          'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                          'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
                          'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                          'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite',
                          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
            
            # Display prediction
            disease = class_name[result_index]
            confidence = np.max(prediction_probabilities) * 100
            st.success(f"Our Prediction: **{disease}**")
            st.write(f"Confidence Level: {confidence:.2f}%")

            # Suggest treatment (example)
            st.markdown(f"### Recommended Actions for **{disease}**:")
            treatment_suggestions = {
                'Apple___Apple_scab': 'Use a fungicide that targets fungal infections, ensure proper spacing between trees to allow airflow.',
                'Potato___Early_blight': 'Apply a copper-based fungicide and avoid overwatering.',
                'Tomato___Bacterial_spot': 'Remove affected leaves, treat with a bacterial-resistant fungicide, and avoid overhead irrigation.'
                # Add more diseases and treatments as needed
            }

            treatment = treatment_suggestions.get(disease, "No specific treatment suggestions available.")
            st.write(treatment)

# FAQ Page
elif app_mode == "FAQ":
    st.header("Frequently Asked Questions (FAQ)")
    st.markdown("""
    ### Q1: How accurate is this system?
    **A1:** Our model is trained on a large dataset of plant diseases, and we continuously improve it. While it provides highly accurate predictions, it is always recommended to verify with agricultural experts.

    ### Q2: Can this system detect all plant diseases?
    **A2:** This system can detect a variety of diseases, but it may not cover all possible diseases. The dataset includes 38 different diseases.

    ### Q3: What should I do if my plant is infected?
    **A3:** Based on the detected disease, we will suggest possible actions like applying specific fungicides, improving watering practices, or consulting with an expert.

    ### Q4: Can I use this for any plant?
    **A4:** The model works with the plants included in the dataset. If you have other types of plants, the model may not recognize them accurately.

    ### Q5: Is there a mobile version of the app?
    **A5:** Currently, the app is web-based. However, we are working on optimizing it for mobile use.
    """)



# Contact Us Page
elif app_mode == "CONTACT US":
    st.header("Contact Us")
    st.markdown("""
    We'd love to hear from you! Whether you have a question, suggestion, or need assistance, feel free to get in touch with us.

    **Email:** info@plantdisease.com  
    **Phone:** +91 93754 45267  
    **Address:** 45 Sustainable Agriculture St., Ahmedabad, India

    Or, fill out the form below to send us a message:
    """)
    contact_message = st.text_area("Your Message:")
    if st.button("Send Message"):
        if contact_message:
            st.success("Thank you for your message! We will get back to you shortly.")
        else:
            st.warning("Please write a message before sending.")
