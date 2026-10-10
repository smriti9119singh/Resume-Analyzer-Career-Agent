"""
Streamlit User Interface Entry Point for AI Resume Analyzer & Career Agent.
Run locally via: `streamlit run app.py`
"""
import streamlit as st
from modules.parser import validate_and_extract_pdf_bytes, parse_resume_text
from modules.agent import generate_grounded_resume_feedback, run_career_agent_pipeline
from modules.report_generator import build_markdown_report

st.set_page_config(
    page_title="AI Resume Analyzer & Career Agent",
    layout="wide",
)

st.title("AI Resume Analyzer & Career Agent")
st.caption(
    "Data Science Portfolio Project: PDF Parsing · Scikit-learn TF-IDF Cosine Similarity · "
    "Gemini API Grounded Feedback · Tool-Calling Career Agent"
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume (PDF) or Paste Text")
    uploaded_pdf = st.file_uploader("Upload Resume PDF (max 5 MB)", type=["pdf"])
    resume_text_input = st.text_area("Or paste resume text directly:", height=220)

with col2:
    st.subheader("2. Target Job Description & Study Plan")
    jd_text_input = st.text_area("Paste Target Job Description:", height=220)
    study_hours = st.slider("Available Study Hours / Week", min_value=2, max_value=30, value=10)
    study_weeks = st.slider("Roadmap Duration (Weeks)", min_value=2, max_value=8, value=4)

if st.button("Analyze Resume & Run Career Agent", type="primary"):
    try:
        if uploaded_pdf is not None:
            parsed = validate_and_extract_pdf_bytes(uploaded_pdf.read(), uploaded_pdf.name)
        elif resume_text_input.strip():
            parsed = parse_resume_text(resume_text_input)
        else:
            st.error("Please upload a PDF resume or paste your resume text.")
            st.stop()

        if not jd_text_input.strip():
            st.error("Please paste a target job description.")
            st.stop()

        with st.spinner("Running TF-IDF Cosine Similarity, Skill Extraction, and Career Agent..."):
            agent_output = run_career_agent_pipeline(
                parsed["sanitized_text"],
                jd_text_input,
                study_hours_per_week=study_hours,
                weeks=study_weeks,
            )
            match_res = agent_output["match_result"]
            feedback = generate_grounded_resume_feedback(
                parsed["sanitized_text"], jd_text_input, match_res
            )

        st.success("Analysis Complete!")
        m1, m2, m3 = st.columns(3)
        m1.metric("Overall Match Score", f"{match_res['overall_match_score']}%")
        m2.metric("TF-IDF Cosine Similarity", f"{match_res['tfidf_cosine_similarity']}%")
        m3.metric("Skill Overlap Score", f"{match_res['skill_overlap_score']}%")
        st.info(match_res["disclaimer"])

        st.subheader("Matched vs. Missing Skills")
        st.write("**Matched Skills:**", ", ".join(match_res["matched_skills"]) or "None")
        st.write("**Missing Skills:**", ", ".join(match_res["missing_skills"]) or "None")

        st.subheader("Grounded Resume Feedback")
        st.write(feedback.get("grounded_summary"))

        st.subheader("Prioritized Learning Roadmap")
        for week in agent_output["roadmap"]:
            st.markdown(
                f"**Week {week['week']}: {week['focus_skill']}** ({week['allocated_hours']} hrs) — {week['deliverable']}"
            )

        report_md = build_markdown_report(
            parsed["candidate_name"], match_res, feedback, agent_output
        )
        st.download_button(
            "Download Analysis Report (.md)",
            data=report_md,
            file_name="career_analysis_report.md",
            mime="text/markdown",
        )
    except Exception as err:
        st.error(f"Validation / Processing Error: {err}")
