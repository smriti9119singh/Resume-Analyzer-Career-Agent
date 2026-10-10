"""
Configuration, skill taxonomy, verified learning resource catalog, and security sanitization.
"""
import re
from typing import Dict, List, Tuple

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
MIN_RESUME_CHARS = 40
MAX_TEXT_CHARS = 35000
GEMINI_MODEL = "gemini-2.5-flash"
TFIDF_WEIGHT = 0.45
SKILL_OVERLAP_WEIGHT = 0.55

SKILL_TAXONOMY: List[Dict] = [
    {"canonical": "Python", "category": "Programming & Querying", "aliases": ["python", "python3", "py"]},
    {"canonical": "SQL", "category": "Programming & Querying", "aliases": ["sql", "mysql", "postgresql", "postgres", "sqlite"]},
    {"canonical": "R", "category": "Programming & Querying", "aliases": ["r programming", "rstudio"]},
    {"canonical": "Pandas", "category": "Programming & Querying", "aliases": ["pandas", "dataframe"]},
    {"canonical": "NumPy", "category": "Programming & Querying", "aliases": ["numpy"]},
    {"canonical": "SciPy", "category": "Programming & Querying", "aliases": ["scipy"]},
    {"canonical": "Scikit-learn", "category": "Machine Learning & Statistics", "aliases": ["scikit-learn", "sklearn", "scikit learn"]},
    {"canonical": "PyTorch", "category": "Machine Learning & Statistics", "aliases": ["pytorch", "torch"]},
    {"canonical": "TensorFlow", "category": "Machine Learning & Statistics", "aliases": ["tensorflow", "keras"]},
    {"canonical": "XGBoost", "category": "Machine Learning & Statistics", "aliases": ["xgboost", "lightgbm", "gradient boosting"]},
    {"canonical": "A/B Testing", "category": "Machine Learning & Statistics", "aliases": ["a/b testing", "ab testing", "hypothesis testing"]},
    {"canonical": "Statistical Modeling", "category": "Machine Learning & Statistics", "aliases": ["statistical modeling", "regression", "statistics"]},
    {"canonical": "Time Series Forecasting", "category": "Machine Learning & Statistics", "aliases": ["time series", "arima", "prophet", "forecasting"]},
    {"canonical": "Feature Engineering", "category": "Machine Learning & Statistics", "aliases": ["feature engineering", "pca"]},
    {"canonical": "Model Evaluation", "category": "Machine Learning & Statistics", "aliases": ["model evaluation", "cross-validation", "roc-auc", "f1-score", "cosine similarity"]},
    {"canonical": "Natural Language Processing", "category": "Generative AI & NLP", "aliases": ["nlp", "natural language processing", "tf-idf", "tokenization", "spacy"]},
    {"canonical": "Large Language Models", "category": "Generative AI & NLP", "aliases": ["llm", "llms", "large language models", "gemini", "gemini api", "transformers"]},
    {"canonical": "RAG Architectures", "category": "Generative AI & NLP", "aliases": ["rag", "retrieval-augmented generation", "vector database", "faiss"]},
    {"canonical": "Agentic AI & Tool Calling", "category": "Generative AI & NLP", "aliases": ["agentic ai", "ai agents", "tool calling", "function calling"]},
    {"canonical": "Apache Spark", "category": "Data Engineering & Cloud", "aliases": ["spark", "pyspark", "apache spark"]},
    {"canonical": "Airflow", "category": "Data Engineering & Cloud", "aliases": ["airflow", "apache airflow", "etl"]},
    {"canonical": "AWS", "category": "Data Engineering & Cloud", "aliases": ["aws", "amazon web services", "sagemaker"]},
    {"canonical": "Google Cloud Platform", "category": "Data Engineering & Cloud", "aliases": ["gcp", "google cloud", "bigquery", "vertex ai"]},
    {"canonical": "Snowflake", "category": "Data Engineering & Cloud", "aliases": ["snowflake", "dbt"]},
    {"canonical": "Tableau", "category": "Analytics & Visualization", "aliases": ["tableau"]},
    {"canonical": "Power BI", "category": "Analytics & Visualization", "aliases": ["power bi", "powerbi"]},
    {"canonical": "Matplotlib & Seaborn", "category": "Analytics & Visualization", "aliases": ["matplotlib", "seaborn", "plotly"]},
    {"canonical": "Streamlit", "category": "Analytics & Visualization", "aliases": ["streamlit"]},
    {"canonical": "Docker", "category": "Software Engineering & MLOps", "aliases": ["docker", "containerization"]},
    {"canonical": "MLflow", "category": "Software Engineering & MLOps", "aliases": ["mlflow", "mlops"]},
    {"canonical": "FastAPI", "category": "Software Engineering & MLOps", "aliases": ["fastapi", "flask", "rest api"]},
    {"canonical": "Git & CI/CD", "category": "Software Engineering & MLOps", "aliases": ["git", "github", "ci/cd", "pytest"]},
]

VERIFIED_RESOURCES: Dict[str, Dict[str, str]] = {
    "Python": {"title": "Official Python 3 Tutorial", "url": "https://docs.python.org/3/tutorial/"},
    "SQL": {"title": "Mode SQL Analytics Tutorial", "url": "https://mode.com/sql-tutorial/"},
    "Pandas": {"title": "Pandas User Guide", "url": "https://pandas.pydata.org/docs/user_guide/index.html"},
    "Scikit-learn": {"title": "Scikit-learn User Guide", "url": "https://scikit-learn.org/stable/user_guide.html"},
    "PyTorch": {"title": "PyTorch Basics Tutorial", "url": "https://pytorch.org/tutorials/beginner/basics/intro.html"},
    "Natural Language Processing": {"title": "Hugging Face NLP Course", "url": "https://huggingface.co/learn/nlp-course/"},
    "Large Language Models": {"title": "Google Gemini API Documentation", "url": "https://ai.google.dev/gemini-api/docs"},
    "Agentic AI & Tool Calling": {"title": "Gemini Function Calling Guide", "url": "https://ai.google.dev/gemini-api/docs/function-calling"},
    "Docker": {"title": "Docker Get Started Guide", "url": "https://docs.docker.com/get-started/"},
    "FastAPI": {"title": "FastAPI Official Tutorial", "url": "https://fastapi.tiangolo.com/tutorial/"},
    "MLflow": {"title": "MLflow Quickstart Guide", "url": "https://mlflow.org/docs/latest/getting-started/index.html"},
    "Apache Spark": {"title": "PySpark Getting Started", "url": "https://spark.apache.org/docs/latest/api/python/getting_started/index.html"},
    "Airflow": {"title": "Apache Airflow Tutorial", "url": "https://airflow.apache.org/docs/apache-airflow/stable/tutorial/index.html"},
    "Time Series Forecasting": {"title": "Forecasting: Principles and Practice", "url": "https://otexts.com/fpp3/"},
}


def sanitize_untrusted_text(raw_text: str) -> Tuple[str, List[str]]:
    """
    Sanitizes untrusted resume and job description text against prompt injection
    and strips non-printable control characters.
    """
    if not raw_text or not isinstance(raw_text, str):
        return "", []

    flags: List[str] = []
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", raw_text)[:MAX_TEXT_CHARS]

    patterns = [
        ("System prompt override attempt", re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)", re.I)),
        ("Role hijacking attempt", re.compile(r"you\s+are\s+now\s+(a|an|in)\s+developer\s+mode|system\s*:\s*override", re.I)),
    ]

    for label, pat in patterns:
        if pat.search(cleaned):
            flags.append(label)
            cleaned = pat.sub("[REDACTED_UNTRUSTED_DIRECTIVE]", cleaned)

    return cleaned.strip(), flags
