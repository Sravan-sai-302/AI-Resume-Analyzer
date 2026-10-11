import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🤖 AI Resume Analyzer")
st.write("Analyze your resume against a job description and find your match score.")

st.divider()

# ---------------- SKILLS ----------------
SKILLS = [
    "python", "java", "javascript", "html", "css",
    "react", "node.js", "express", "sql", "mysql",
    "mongodb", "django", "flask", "fastapi",
    "machine learning", "deep learning", "artificial intelligence",
    "generative ai", "nlp", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch",
    "git", "github", "docker", "aws",
    "rest api", "api", "data structures", "algorithms"
]


# ---------------- PDF TEXT EXTRACTION ----------------
def extract_resume_text(pdf_file):
    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ---------------- SKILL EXTRACTION ----------------
def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in SKILLS:

        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text):
            found_skills.append(skill)

    return sorted(found_skills)


# ---------------- UI ----------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("📄 Upload Resume")

    resume_file = st.file_uploader(
        "Upload your resume PDF",
        type=["pdf"]
    )

with col2:

    st.subheader("💼 Job Description")

    job_description = st.text_area(
        "Paste the job description here",
        height=250
    )


# ---------------- ANALYSIS ----------------

if st.button("🔍 Analyze Resume", type="primary"):

    if resume_file is None:

        st.warning("Please upload your resume.")

    elif not job_description.strip():

        st.warning("Please enter the job description.")

    else:

        with st.spinner("Analyzing your resume..."):

            # Extract resume
            resume_text = extract_resume_text(resume_file)

            # Calculate similarity
            vectorizer = TfidfVectorizer(
                stop_words="english"
            )

            vectors = vectorizer.fit_transform(
                [resume_text, job_description]
            )

            similarity = cosine_similarity(
                vectors[0:1],
                vectors[1:2]
            )[0][0]

            match_score = round(similarity * 100, 2)

            # Extract skills
            resume_skills = extract_skills(resume_text)

            job_skills = extract_skills(job_description)

            matching_skills = sorted(
                set(resume_skills) & set(job_skills)
            )

            missing_skills = sorted(
                set(job_skills) - set(resume_skills)
            )

        # ---------------- RESULTS ----------------

        st.divider()

        st.subheader("📊 Resume Analysis")

        score_col, skill_col = st.columns(2)

        with score_col:

            st.metric(
                "Resume Match Score",
                f"{match_score}%"
            )

            st.progress(
                min(match_score / 100, 1.0)
            )

        with skill_col:

            st.metric(
                "Skills Found",
                len(resume_skills)
            )

        # ---------------- SKILLS ----------------

        st.subheader("✅ Matching Skills")

        if matching_skills:

            st.write(
                ", ".join(
                    skill.title()
                    for skill in matching_skills
                )
            )

        else:

            st.info("No matching skills detected.")


        st.subheader("❌ Missing Skills")

        if missing_skills:

            st.write(
                ", ".join(
                    skill.title()
                    for skill in missing_skills
                )
            )

        else:

            st.success(
                "Great! No missing skills detected."
            )


        # ---------------- RECOMMENDATIONS ----------------

        st.subheader("💡 Resume Improvement Suggestions")

        if match_score < 40:

            st.warning(
                "Your resume has a low match with this job. "
                "Consider adding relevant technical skills and project experience."
            )

        elif match_score < 70:

            st.info(
                "Your resume has a moderate match. "
                "Try adding the missing skills and relevant keywords."
            )

        else:

            st.success(
                "Your resume has a strong match with this job description!"
            )

        if missing_skills:

            st.write("Focus on these skills:")

            for skill in missing_skills:

                st.write(f"• {skill.title()}")

        # ---------------- RESUME SKILLS ----------------

        st.subheader("🧠 Skills Detected in Your Resume")

        if resume_skills:

            st.write(
                ", ".join(
                    skill.title()
                    for skill in resume_skills
                )
            )

        else:

            st.info(
                "No predefined technical skills were detected."
            )