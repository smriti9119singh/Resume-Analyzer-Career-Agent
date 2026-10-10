"""
Deterministic NLP skill extraction and alias normalization module.
"""
import re
from typing import Dict, List
from .config import SKILL_TAXONOMY, VERIFIED_RESOURCES


def extract_skills_from_text(text: str) -> List[Dict]:
    """
    Extracts canonical skills from raw text using token-boundary regular expressions.
    """
    if not text or not isinstance(text, str):
        return []

    lowered = text.lower()
    extracted: List[Dict] = []

    for item in SKILL_TAXONOMY:
        total_hits = 0
        matched_aliases: List[str] = []
        for alias in item["aliases"]:
            escaped = re.escape(alias.lower())
            pattern = re.compile(rf"(?:^|[^a-z0-9_+#])({escaped})(?=$|[^a-z0-9_+#])", re.I)
            hits = pattern.findall(lowered)
            if hits:
                total_hits += len(hits)
                matched_aliases.append(alias)

        if total_hits > 0:
            extracted.append({
                "canonical": item["canonical"],
                "category": item["category"],
                "matched_aliases": sorted(list(set(matched_aliases))),
                "frequency": total_hits,
            })

    extracted.sort(key=lambda x: (-x["frequency"], x["canonical"]))
    return extracted


def compare_resume_and_job_skills(resume_text: str, jd_text: str) -> Dict:
    """
    Compares skills found in the resume against skills required in the job description.
    """
    resume_skills = extract_skills_from_text(resume_text)
    jd_skills = extract_skills_from_text(jd_text)

    resume_map = {s["canonical"]: s for s in resume_skills}
    matched = [resume_map[s["canonical"]] for s in jd_skills if s["canonical"] in resume_map]
    missing = [s for s in jd_skills if s["canonical"] not in resume_map]

    overlap_ratio = len(matched) / len(jd_skills) if jd_skills else 0.0

    return {
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "matched_skills": matched,
        "missing_skills": missing,
        "skill_overlap_ratio": round(overlap_ratio, 4),
    }


def lookup_verified_resources(skill_names: List[str]) -> List[Dict[str, str]]:
    """
    Returns verified learning resource URLs for the specified canonical skill names.
    """
    results: List[Dict[str, str]] = []
    for skill in skill_names:
        if skill in VERIFIED_RESOURCES:
            results.append({
                "skill": skill,
                "title": VERIFIED_RESOURCES[skill]["title"],
                "url": VERIFIED_RESOURCES[skill]["url"],
            })
    return results
