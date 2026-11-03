
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------
st.set_page_config(
    page_title="Pollution Detection System",
    page_icon="🌍",
    layout="centered"
)

# --------------------------------
# TITLE
# --------------------------------
st.title("🌍 Pollution Detection System")
st.subheader("CNN Based Image Classification")
st.write("Upload an image to identify the type of pollution.")

st.divider()

# --------------------------------
# LOAD TRAINED MODEL
# --------------------------------
@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "CAT.keras",
        compile=False,
        safe_mode=True
    )

    with open("classes.json", "r") as f:
        class_names = json.load(f)

    return model, class_names


# --------------------------------
# CHECK MODEL FILES
# --------------------------------
if not os.path.exists("CAT.keras"):
    st.error("CAT.keras file not found!")
    st.stop()

if not os.path.exists("classes.json"):
    st.error("classes.json file not found!")
    st.stop()


# --------------------------------
# LOAD MODEL
# --------------------------------
try:
    with st.spinner("Loading CNN Model..."):
        model, class_names = load_model()

    st.success("CNN Model Loaded Successfully!")

except Exception as e:
    st.error("Unable to load model. Check TensorFlow/Keras versions.")
    st.exception(e)
    st.stop()


# --------------------------------
# MODEL INFORMATION
# --------------------------------
with st.expander("📋 Model Information"):

    st.write("Model Name: CNN")
    st.write("Image Size: 64 x 64")
    st.write("Number of Classes:", len(class_names))
    st.write("Available Classes:", class_names)


# --------------------------------
# IMAGE UPLOAD
# --------------------------------
st.divider()

st.header("📸 Upload Pollution Image")

uploaded_file = st.file_uploader(
    "Select an image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------
# IMAGE PREDICTION
# --------------------------------
if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button(
        "🔍 Detect Pollution",
        use_container_width=True
    ):

        with st.spinner("Analyzing image..."):

            # Resize image
            resized_img = img.resize((64, 64))

            # Convert to NumPy array
            x = np.array(
                resized_img,
                dtype=np.float32
            )

            # Expand dimensions
            x = np.expand_dims(x, axis=0)

            # Predict
            prediction = model.predict(
                x,
                verbose=0
            )

            # Predicted class
            predicted_index = int(
                np.argmax(prediction[0])
            )

            predicted_class = class_names[predicted_index]

            confidence = float(
                prediction[0][predicted_index] * 100
            )

        # --------------------------------
        # DISPLAY RESULT
        # --------------------------------
        st.divider()

        st.header("📊 Prediction Result")

        st.success(
            f"Detected Pollution: {predicted_class}"
        )

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

        # --------------------------------
        # CLASS PROBABILITIES
        # --------------------------------
        st.subheader("📈 Class-wise Prediction")

        for i, name in enumerate(class_names):

            probability = float(
                prediction[0][i] * 100
            )

            st.write(
                f"**{name}** : {probability:.2f}%"
            )

            st.progress(
                min(probability / 100, 1.0)
            )

        # --------------------------------
        # FINAL MESSAGE
        # --------------------------------
        st.divider()

        st.info(
            f"The CNN model predicts this image as {predicted_class}."
        )


# --------------------------------
# FOOTER
# --------------------------------
st.divider()

st.caption(
    "🌍 Pollution Detection System | Deep Learning | CNN"
)
