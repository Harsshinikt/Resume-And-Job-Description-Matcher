import streamlit as st
from cohere_utils import get_similarity_score, generate_resume_suggestions

st.set_page_config(page_title="Resume + JD Matcher", layout="centered")

st.title("📄 Resume + JD Matching Assistant")
st.write("Upload your Resume and Job Description to get a match score and improvement suggestions.")

resume_file = st.file_uploader("Upload Resume (.txt)", type="txt")
jd_file = st.file_uploader("Upload Job Description (.txt)", type="txt")

if resume_file and jd_file:
    resume_text = resume_file.read().decode("utf-8")
    jd_text = jd_file.read().decode("utf-8")

    with st.spinner("Analyzing..."):
        score = get_similarity_score(resume_text, jd_text)
        feedback = generate_resume_suggestions(resume_text, jd_text)

    st.subheader("📊 Match Score")
    st.success(f"Resume matches the JD by **{round(score * 100, 2)}%**")

    st.subheader("💡 Suggestions to Improve Resume")
    st.write(feedback)

else:
    st.info("Please upload both a resume and a job description.")
