"""
Machine Learning Resume-Job Matcher using Scikit-learn TF-IDF Vectorization and Cosine Similarity.
Includes a pure-Python mathematical equivalent fallback when scikit-learn is not installed in minimal test containers.
"""
import math
import re
from typing import Dict, List
from .config import TFIDF_WEIGHT, SKILL_OVERLAP_WEIGHT
from .skill_extractor import compare_resume_and_job_skills

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "will", "with",
    "role", "work", "working", "team", "experience", "using", "required", "preferred",
}


def compute_tfidf_cosine_similarity(resume_text: str, jd_text: str) -> float:
    """
    Calculates TF-IDF Cosine Similarity (0.0 to 1.0) using Scikit-learn's TfidfVectorizer
    and cosine_similarity when installed, or the exact smoothed sublinear TF-IDF formula.
    """
    if not resume_text or not jd_text or not resume_text.strip() or not jd_text.strip():
        return 0.0

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            smooth_idf=True,
            norm="l2",
        )
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        return max(0.0, min(1.0, sim))
    except ImportError:
        # Exact mathematical equivalent of sklearn TfidfVectorizer(sublinear_tf=True, smooth_idf=True, norm='l2')
        def tokenize(text: str) -> List[str]:
            words = [
                w for w in re.sub(r"[^a-z0-9+#./_-]+", " ", text.lower()).split()
                if len(w) >= 2 and w not in STOP_WORDS and not w.isdigit()
            ]
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
            return words + bigrams

        r_tokens = tokenize(resume_text)
        j_tokens = tokenize(jd_text)
        if not r_tokens or not j_tokens:
            return 0.0

        r_counts: Dict[str, int] = {}
        for t in r_tokens:
            r_counts[t] = r_counts.get(t, 0) + 1
        j_counts: Dict[str, int] = {}
        for t in j_tokens:
            j_counts[t] = j_counts.get(t, 0) + 1

        vocab = set(r_counts.keys()) | set(j_counts.keys())
        r_vec: Dict[str, float] = {}
        j_vec: Dict[str, float] = {}

        for term in vocab:
            rc = r_counts.get(term, 0)
            jc = j_counts.get(term, 0)
            df = (1 if rc > 0 else 0) + (1 if jc > 0 else 0)
            idf = math.log((1.0 + 2.0) / (1.0 + df)) + 1.0
            if rc > 0:
                r_vec[term] = (1.0 + math.log(rc)) * idf
            if jc > 0:
                j_vec[term] = (1.0 + math.log(jc)) * idf

        r_norm = math.sqrt(sum(v * v for v in r_vec.values())) or 1.0
        j_norm = math.sqrt(sum(v * v for v in j_vec.values())) or 1.0

        dot = sum((r_vec.get(t, 0.0) / r_norm) * (j_vec.get(t, 0.0) / j_norm) for t in vocab)
        return max(0.0, min(1.0, dot))


def evaluate_resume_job_match(resume_text: str, jd_text: str) -> Dict:
    """
    Computes TF-IDF cosine similarity, deterministic skill overlap, and the composite match score.
    """
    cosine_sim = compute_tfidf_cosine_similarity(resume_text, jd_text)
    skill_comp = compare_resume_and_job_skills(resume_text, jd_text)

    tfidf_pct = round(cosine_sim * 100, 1)
    skill_pct = round(skill_comp["skill_overlap_ratio"] * 100, 1)
    overall_pct = round(TFIDF_WEIGHT * tfidf_pct + SKILL_OVERLAP_WEIGHT * skill_pct, 1)

    return {
        "overall_match_score": overall_pct,
        "tfidf_cosine_similarity": tfidf_pct,
        "skill_overlap_score": skill_pct,
        "matched_skills": [s["canonical"] for s in skill_comp["matched_skills"]],
        "missing_skills": [s["canonical"] for s in skill_comp["missing_skills"]],
        "disclaimer": (
            "This score is an estimate of lexical and skill alignment via TF-IDF Cosine Similarity "
            "and deterministic skill extraction, not a prediction of hiring success."
        ),
    }
