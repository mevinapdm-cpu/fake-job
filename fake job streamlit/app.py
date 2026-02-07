import streamlit as st
import pickle
import numpy as np


# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_model()


# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="💼",
    layout="centered"
)


# -------------------------------
# Custom Style
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

h1 {
    color: #ffffff;
}

label {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Title
# -------------------------------
st.markdown("## 💼 Fake Job Detection System")
st.markdown("Check whether a job posting is **Real or Fake** using Machine Learning")


# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns(2)


# -------------------------------
# Inputs (With Options)
# -------------------------------
with col1:

    job_title = st.selectbox(
        "Job Title",
        [
            "Software Engineer",
            "Data Analyst",
            "HR Manager",
            "Web Developer",
            "Marketing Executive",
            "Accountant",
            "Other"
        ]
    )

    job_type = st.radio(
        "Job Type",
        ["Full-Time", "Part-Time", "Internship", "Contract"]
    )

    remote = st.checkbox("Remote Job Available")

    experience = st.slider(
        "Required Experience (Years)",
        0, 15, 2
    )


with col2:

    location = st.selectbox(
        "Job Location",
        [
            "India",
            "USA",
            "UK",
            "Canada",
            "Remote",
            "Other"
        ]
    )

    salary = st.selectbox(
        "Salary Range",
        [
            "Below 3 LPA",
            "3 - 6 LPA",
            "6 - 10 LPA",
            "Above 10 LPA"
        ]
    )

    company_type = st.radio(
        "Company Type",
        ["Startup", "MNC", "Private", "Government"]
    )


# -------------------------------
# Text Inputs
# -------------------------------
st.subheader("Job Description")

description = st.text_area(
    "Enter Job Description",
    height=120
)

requirements = st.text_area(
    "Enter Requirements",
    height=100
)

benefits = st.text_area(
    "Enter Benefits",
    height=80
)


# -------------------------------
# Example Button
# -------------------------------
if st.button("📌 Fill Sample Data"):

    st.session_state["description"] = "Looking for skilled developer with good communication skills."
    st.session_state["requirements"] = "Python, SQL, Machine Learning"
    st.session_state["benefits"] = "Work from home, Health insurance"


# -------------------------------
# Prediction
# -------------------------------
st.markdown("---")

if st.button("🔍 Predict Job"):

    if description == "":

        st.warning("⚠️ Please enter Job Description")

    else:

        # Convert inputs to text
        text = f"""
        {job_title}
        {job_type}
        {location}
        {salary}
        {company_type}
        {'Remote' if remote else 'Onsite'}
        {experience} years experience
        {description}
        {requirements}
        {benefits}
        """

        vector = vectorizer.transform([text])

        result = model.predict(vector)[0]

        prob = model.predict_proba(vector)[0]


        # Output
        st.subheader("Result")

        if result == 1:

            st.error("🚨 This Job is FAKE")

            st.progress(float(prob[1]))

            st.write(f"Confidence: {round(prob[1]*100,2)} %")

        else:

            st.success("✅ This Job is REAL")

            st.progress(float(prob[0]))

            st.write(f"Confidence: {round(prob[0]*100,2)} %")


# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("### 📘 ML Project | Fake Job Detection")
st.markdown("Developed using Logistic Regression & Streamlit")




