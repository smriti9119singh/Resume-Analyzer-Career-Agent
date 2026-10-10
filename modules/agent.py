"""
Generative AI Resume Feedback and Agentic Career Assistant modules using google-genai SDK.
"""
import json
import os
from typing import Dict, List
from .config import GEMINI_MODEL
from .matcher import evaluate_resume_job_match
from .skill_extractor import lookup_verified_resources


def generate_grounded_resume_feedback(resume_text: str, jd_text: str, match_data: Dict) -> Dict:
    """
    Calls Google Gemini API using the supported `google-genai` SDK to generate
    grounded resume feedback without inventing qualifications.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    matched = match_data.get("matched_skills", [])
    missing = match_data.get("missing_skills", [])

    if not api_key or api_key == "MY_GEMINI_API_KEY":
        return {
            "grounded_summary": (
                f"Data-driven candidate skilled in {', '.join(matched[:5]) or 'analytical problem solving'}, "
                "seeking to apply quantitative modeling and software engineering practices to the target role."
            ),
            "strengths": [
                f"Matches {len(matched)} required skills ({', '.join(matched[:6])}).",
                f"Achieves {match_data.get('tfidf_cosine_similarity', 0)}% TF-IDF lexical alignment with the job description.",
            ],
            "improvements": [
                f"Close skill gaps in {', '.join(missing[:5]) or 'deployment workflows'} through targeted portfolio projects.",
                "Ensure every experience bullet pairs technical tools with validation metrics (e.g., F1-score, RMSE).",
            ],
            "used_fallback": True,
            "notice": "GEMINI_API_KEY not found in environment. Returned deterministic structured feedback.",
        }

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        prompt = (
            "Analyze the candidate resume against the job description.\n"
            "Do NOT invent credentials, employers, or metrics.\n\n"
            f"Matched Skills: {matched}\nMissing Skills: {missing}\n"
            f"Resume:\n{resume_text[:8000]}\n\nJob Description:\n{jd_text[:6000]}"
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as exc:
        return {
            "grounded_summary": f"Candidate with verified experience in {', '.join(matched[:4]) or 'data analysis'}.",
            "strengths": [f"Demonstrated alignment on {', '.join(matched[:5])}."],
            "improvements": [f"Add projects covering {', '.join(missing[:4])}."],
            "used_fallback": True,
            "notice": f"Gemini API error handled gracefully: {exc}",
        }


def run_career_agent_pipeline(
    resume_text: str,
    jd_text: str,
    study_hours_per_week: int = 10,
    weeks: int = 4,
) -> Dict:
    """
    Executes the multi-step tool-calling Career Agent workflow:
    1. Deterministic Skill & TF-IDF Matching Tool
    2. Verified Learning Resource Lookup Tool
    3. Prioritized Roadmap & Interview Prep Generation
    4. Recommendation Alignment Verification Tool
    """
    tool_trace: List[Dict] = []

    # Step 1: Run deterministic matcher
    match_result = evaluate_resume_job_match(resume_text, jd_text)
    tool_trace.append({
        "tool": "evaluate_resume_job_match",
        "type": "deterministic_python_tool",
        "summary": f"Overall Score: {match_result['overall_match_score']}%, Missing: {len(match_result['missing_skills'])} skills",
    })

    # Step 2: Lookup verified URLs for missing skills
    missing_skills = match_result["missing_skills"]
    target_skills = missing_skills if missing_skills else match_result["matched_skills"][:4]
    resources = lookup_verified_resources(target_skills)
    tool_trace.append({
        "tool": "lookup_verified_resources",
        "type": "deterministic_python_tool",
        "summary": f"Found {len(resources)} verified documentation URLs for target skills.",
    })

    # Step 3: Build prioritized learning roadmap
    roadmap = []
    pool = target_skills if target_skills else ["Python", "Scikit-learn", "Docker", "FastAPI"]
    for w in range(weeks):
        focus = pool[w % len(pool)]
        skill_res = lookup_verified_resources([focus])
        roadmap.append({
            "week": w + 1,
            "focus_skill": focus,
            "allocated_hours": study_hours_per_week,
            "deliverable": f"Build and test a reproducible {focus} mini-project on GitHub.",
            "verified_resources": skill_res,
        })

    # Step 4: Verify that generated roadmap covers identified missing skills
    roadmap_text = " ".join(r["focus_skill"].lower() for r in roadmap)
    covered = [s for s in missing_skills if s.lower() in roadmap_text]
    validation = {
        "is_valid": len(covered) > 0 or len(missing_skills) == 0,
        "covered_missing_skills": covered,
        "verified_urls_count": sum(len(r["verified_resources"]) for r in roadmap),
    }
    tool_trace.append({
        "tool": "validate_agent_recommendations",
        "type": "deterministic_python_tool",
        "summary": f"Verified {len(covered)}/{len(missing_skills)} missing skills scheduled across {weeks} weeks.",
    })

    return {
        "match_result": match_result,
        "roadmap": roadmap,
        "tool_trace": tool_trace,
        "validation": validation,
    }
