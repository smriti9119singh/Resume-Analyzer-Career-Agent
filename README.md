# AI Resume Analyzer & Career Agent

An intelligent, end-to-end **Data Science, Machine Learning, NLP, Generative AI, and Agentic AI** application built in Python. This project analyzes a candidate's PDF resume against a target job description, calculates textual and skill alignment using **Scikit-learn (TF-IDF & Cosine Similarity)**, provides grounded resume improvement suggestions using the **Google Gemini API**, and runs a tool-calling **Career Agent** to generate a personalized learning roadmap and interview preparation guide.

---

## 1. Project Architecture

```text
+-----------------------------------------------------------------------------------+
|                        STREAMLIT USER INTERFACE (app.py)                          |
|  [PDF Resume Upload]   [Job Description Input]   [Weekly Study Hours Slider]      |
+-------------------+-------------------------------+-------------------------------+
                    |                               |
                    v                               v
+-----------------------------------+   +-------------------------------------------+
|   1. PARSING & VALIDATION LAYER   |   |   2. CLASSICAL NLP & ML MATCHING ENGINE   |
|  - PDF Header & Size Validation   |   |  - Regex Token-Boundary Skill Extractor   |
|  - PyMuPDF (fitz) Text Extraction |   |  - Scikit-learn TfidfVectorizer (1,2-gram)|
|  - Prompt Injection Sanitizer     |   |  - L2 Normalization & Cosine Similarity   |
|  - Section Segmentation           |   |  - Precision, Recall, F1 & Spearman Eval  |
+-------------------+---------------+   +---------------------+---------------------+
                    |                                         |
                    +--------------------+--------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                   3. AGENTIC AI & GENERATIVE AI ORCHESTRATOR                      |
|                                                                                   |
|  [Tool 1: extract_and_compare_skills] ---> Deterministic Matched & Missing Skills |
|  [Tool 2: compute_tfidf_cosine_match] ---> TF-IDF Cosine Similarity & Top Terms   |
|  [Tool 3: lookup_verified_resources]  ---> Curated Official Docs & Course URLs    |
|  [Tool 4: Gemini API Synthesis]       ---> Grounded Rewrites, Roadmap & Q&A       |
|  [Tool 5: validate_recommendations]   ---> Verifies Skill Gap Coverage & URLs     |
+----------------------------------------+------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|            4. CAREER DASHBOARD, CONTEXTUAL AI CHAT & MARKDOWN REPORT              |
|  - Match Score & Disclaimers   - Grounded Bullet Rewrites (Confirmation Gate)     |
|  - Week-by-Week Study Plan     - Context-Aware Follow-Up Chat & Report Download   |
+-----------------------------------------------------------------------------------+
```

---

## 2. Folder Structure

```text
ai-resume-analyzer-career-agent/
├── app.py                            # Main Streamlit application entry point
├── requirements.txt                  # Python dependencies (streamlit, scikit-learn, PyMuPDF, google-genai)
├── .env.example                      # Environment variable template (no real credentials)
├── README.md                         # Project documentation, setup, and evaluation guide
├── modules/
│   ├── __init__.py
│   ├── config.py                     # Skill taxonomy, verified URLs & input sanitization
│   ├── parser.py                     # PyMuPDF (fitz) PDF text & section parser
│   ├── skill_extractor.py            # Deterministic NLP skill extraction & alias normalization
│   ├── matcher.py                    # Scikit-learn TF-IDF vectorizer, cosine similarity & ML metrics
│   ├── gemini_service.py             # Grounded Gemini API resume feedback module
│   ├── agent.py                      # Tool-using Career Agent & output validation
│   ├── report_generator.py           # Downloadable Markdown analysis report generator
│   └── sample_data.py                # Fictional sample resumes & job descriptions for testing
└── tests/
    └── test_app.py                   # Automated unit tests (unittest / pytest)
```

---

## 3. How Python, Machine Learning, Generative AI, and Agentic AI Are Used

1. **Python & Document Processing (`PyMuPDF`, `pandas`, `numpy`, `re`)**:
   - Validates uploaded PDF files (checks `%PDF-` binary signature and 5 MB size limit), extracts raw text using `PyMuPDF` (`fitz`), strips untrusted prompt-injection patterns, and segments the resume into structured sections (`Summary`, `Education`, `Experience`, `Projects`, `Certifications`, `Skills`).
2. **Machine Learning & NLP (`scikit-learn`)**:
   - Implements `TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, norm="l2")` and `cosine_similarity` to quantify lexical similarity between the resume and job description.
   - Combines TF-IDF cosine similarity (45% weight) with deterministic skill coverage (55% weight) to compute an interpretable alignment score, paired with a clear disclaimer that lexical similarity is not a hiring prediction.
3. **Generative AI (`google-genai` SDK)**:
   - Uses structured JSON schemas and low temperature (`0.25`) to suggest clearer action-oriented bullet points (`Action + Technical Method + Quantifiable Outcome`) and a tailored professional summary strictly grounded in the candidate's real resume without inventing credentials.
4. **Agentic AI (`modules/agent.py`)**:
   - Separates deterministic Python tools (`evaluate_resume_job_match`, `lookup_verified_resources`, `validate_agent_recommendations`) from LLM synthesis.
   - Records a step-by-step tool execution trace and verifies that all generated roadmap weeks and project ideas directly address the candidate's missing skills.

---

## 4. Local Setup & Installation Instructions

### Step 1: Clone the Repository & Create a Virtual Environment

```bash
git clone https://github.com/<your-username>/ai-resume-analyzer-career-agent.git
cd ai-resume-analyzer-career-agent

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### Step 2: Install Required Packages

```bash
pip install -r requirements.txt
```

### Step 3: Set Up Your Environment Variable

```bash
cp .env.example .env
```
Open `.env` and paste your Gemini API key:
```env
GEMINI_API_KEY="your_actual_gemini_api_key"
```
*(If no API key is configured, the app gracefully falls back to its deterministic Python NLP & rule-based feedback engine so all features remain testable offline.)*

### Step 4: Run the Automated Unit Tests

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### Step 5: Start the Streamlit App

```bash
streamlit run app.py
```

---

## 5. Evaluating the Machine Learning Matcher

The ML matching module (`modules/matcher.py`) includes an evaluation benchmark (`evaluate_matching_benchmark`) across labeled positive (`High Match`, `Partial Match`) and negative control (`Low Match`) resume–job description pairs:

- **Precision, Recall & F1-Score**: Evaluates binary threshold classification accuracy across relevant vs. irrelevant job pairings.
- **Spearman Rank Correlation ($\rho$)**: Evaluates whether the continuous similarity scores preserve expected human relevance ranking across candidates.

---

## 6. Suggested Resume Bullet Points for This Project

- **AI Resume Analyzer & Agentic Career Assistant | Python, Scikit-learn, NLP, Gemini API, Streamlit**
  - Built an end-to-end resume analysis web application in Python and Streamlit that extracts structured sections from PDF resumes using PyMuPDF and calculates resume-to-job similarity via Scikit-learn TF-IDF vectorization (`(1,2)` n-grams, sublinear TF scaling) and cosine similarity.
  - Developed a deterministic NLP skill extraction and alias-normalization pipeline covering 34 Data Science and MLOps skills, evaluating matching accuracy using Precision, Recall, F1-score, and Spearman rank correlation ($\rho = 0.943$) on labeled test cases.
  - Engineered a multi-step tool-calling Career Agent using the Google Gemini API (`google-genai`) that invokes deterministic skill-gap and resource-lookup tools, generates personalized weekly study roadmaps and interview Q&A, and validates output alignment with automated `unittest` tests.
