import streamlit as st
from transformers import pipeline

# Load robust SMS spam detection model
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="mariagrandury/roberta-base-finetuned-sms-spam-detection"
    )

classifier = load_model()

# Streamlit UI
st.title("📩 Spam SMS Detector")
st.write("Type an SMS message below and the AI will predict if it's **Spam** or **Ham**.")

# User input
sms_input = st.text_area("Enter SMS message here:")
# Label mapping

# sms = st.text_area("Enter SMS here:")

# Predict button
if st.button("Check Spam"):
    sms = sms_input.strip()
    if sms == "":
        st.warning("Please enter a message!")
    else:
        result = classifier(sms)[0]
        # Correct mapping
        label_map = {"LABEL_0": "ham", "LABEL_1": "spam"}
        label = label_map[result['label']]
        score = round(result['score'] * 100, 2)
        st.success(f"Prediction: {label} ({score}%)")