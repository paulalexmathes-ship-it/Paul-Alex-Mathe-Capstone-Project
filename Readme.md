
# Zepto Data & AI Platform — Capstone Project

**Certificate Program in Artificial Intelligence and Machine Learning**

An end-to-end AI/ML platform comprising three interconnected modules: a data-engineering pipeline, an analytics pipeline, and a GenAI support assistant.

---

## Repository Structure
Paul-Alex-Mathe-Capstone-Project/ │ ├── README.md ← You are here (root documentation) ├── requirements.txt ← Combined dependencies for all modules │ ├── data_pipeline/ ← Module 1: Scrape → Clean → Store → Query (25 marks) │ ├── README.md │ ├── data_pipeline.py │ ├── scraped_books_raw.csv │ └── books_pipeline.db │ ├── analytics/ ← Module 2: EDA → Modeling → Evaluation (50 marks) │ ├── README.md │ ├── 01_eda.py │ ├── 02_modeling.py │ ├── titanic.csv │ ├── best_pipeline.joblib │ └── charts/ │ └── support_assistant/ ← Module 3: GenAI Support Assistant (25 marks) ├── README.md ├── Dockerfile ├── app/ │ ├── init.py │ ├── config.py │ ├── models.py │ ├── embeddings.py │ ├── prompt_template.py │ ├── graph.py │ └── main.py └── docs/ ├── doc_01.txt ... doc_08.txt

---

## Modules Overview

### Module 1 — Data Pipeline (25 marks)

A complete web-scraping and data-engineering pipeline that:
- Scrapes 60+ books from books.toscrape.com across 5 categories
- Cleans all fields to proper types
- Converts prices from GBP to INR (1 GBP = 105.50 INR — fixed project constant)
- Creates a normalized SQLite database with 2 tables
- Executes 6 SQL queries demonstrating all required clauses (WHERE, ORDER BY, GROUP BY, HAVING, JOIN, subquery)
- Verifies SQL JOIN output matches pandas merge

**Tech Stack:** Python, BeautifulSoup, SQLite, pandas

---

### Module 2 — Analytics Pipeline (50 marks)

An analyst-to-data-scientist workflow using the Titanic dataset (891 rows × 15 columns):

**Part A — Profiling, Cleaning, and EDA:**
- Missing-value handling with threshold-based strategy (drop column >30%, impute 5–30%, drop rows <5%)
- Univariate analysis with IQR outlier detection and skewness determination
- Bivariate analysis with survival rates by sex, class, and combined
- Correlation matrix with top-2 interpretations
- Multivariate data story (4 charts with written interpretations)

**Part B — Predictive Modeling:**
- Stratified 80/20 train/test split preserving class balance
- sklearn Pipeline with ColumnTransformer (median imputation + StandardScaler for numeric; most frequent + OneHotEncoder for categorical)
- Three classifiers: Logistic Regression, Decision Tree, Random Forest
- Full evaluation: confusion matrix, accuracy, precision, recall, F1, ROC/AUC
- Imbalance handling comparison (baseline vs class_weight='balanced' vs SMOTE)
- GridSearchCV hyperparameter tuning with OOB score
- Regression side-task (predict fare) with heteroscedasticity analysis
- Final model recommendation with justification

**Best Model:** Random Forest (AUC ~0.87, F1 ~0.76)

**Tech Stack:** Python, pandas, NumPy, scikit-learn, seaborn, matplotlib, joblib

---

### Module 3 — Support Assistant (25 marks)

A RAG-powered customer support assistant built with LangGraph, FastAPI, ChromaDB, and sentence-transformers.

**RAG Pipeline:**
1. **Ingestion** — Loads 8 policy documents from `docs/` directory
2. **Embedding** — Generates 384-dim vectors using `all-MiniLM-L6-v2` model, stored in ChromaDB
3. **Retrieval** — Queries top-3 most similar chunks via cosine similarity
4. **Generation** — Produces answers via canned templates (MOCK_LLM=1) or real LLM (MOCK_LLM=0)

**LangGraph Flow:**
[START] → [classify_intent] ──┬── policy_question ──→ [retrieve_and_answer] → [END] └── general_question ─→ [direct_answer] ────→ [END]


**Deployment:** FastAPI on port 7860, containerized with Docker (`python:3.11-slim`)

**Tech Stack:** Python, FastAPI, LangGraph, ChromaDB, sentence-transformers, Pydantic, Docker

---

## Quick Start

### Prerequisites
- Python 3.11 (recommended)
- pip
- Git
- Docker Desktop (for Module 3 containerized run)

### Setup

```bash
# Clone the repository
git clone https://github.com/Paul-Alex-Mathe/Paul-Alex-Mathe-Capstone-Project.git
cd Paul-Alex-Mathe-Capstone-Project

# Create virtual environment
py -3.11 -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
Run Each Module
# Module 1: Data Pipeline
python data_pipeline/data_pipeline.py

# Module 2: Analytics
cd analytics
python 01_eda.py
python 02_modeling.py
cd ..

# Module 3: Support Assistant
cd support_assistant
uvicorn app.main:app --host [IP_ADDRESS] --port 7860

Design Decisions Summary

| Module | Key Decisions |
|--------|--------------|
| Data Pipeline | Median imputation for failed parses; 5 categories scraped for safety margin above 60 books; 0.5s polite delay between requests |
| Analytics | Threshold-based missing value strategy; stratified split to preserve class balance; Random Forest chosen for best AUC/F1 trade-off |
| Support Assistant | Document-level chunking (short docs); MOCK_LLM=1 default for graded baseline; keyword heuristic for intent classification; Pydantic validation with retry for real LLM mode |

Currency Conversion Rate
1 GBP = 105.50 INR — This is a fixed, project-defined constant for Module 1. It is not a live or historical market rate and requires no external API.

Git Workflow
This repository demonstrates proper Git workflow:

Feature branches created for each module
Multiple commits per branch
Merged back into main via merge commits
Viewable via git log --graph --all

Author
Paul Alex Mathe Certificate Program in Artificial Intelligence and Machine Learning