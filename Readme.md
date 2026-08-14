
# Zepto Data & AI Platform — Capstone Project

**Certificate Program in Artificial Intelligence and Machine Learning**

An end-to-end AI/ML platform comprising three interconnected modules: a data-engineering pipeline, an analytics pipeline, and a GenAI support assistant.

---

## Repository Structure

My Assistant
You need to first check what's inside C:\capstone_temp after cloning. The folder name depends on your GitHub repository name. Let's find it:

powershell





# See what's in capstone_temp
cd C:\capstone_temp
dir
This will show you the actual folder name that was created when you cloned. It'll be whatever your repo is named on GitHub (e.g., AIML-Capstone-Project, capstone-project, etc.).

Then:

powershell





# Replace YOUR_REPO_FOLDER with the actual name shown by 'dir'
cd YOUR_REPO_FOLDER

# Now verify your existing modules are here
dir

# NOW create support_assistant
mkdir support_assistant
cd support_assistant
mkdir docs
mkdir app
OR if you haven't cloned yet and just created C:\capstone_temp as empty, do the clone first:

powershell





cd C:\capstone_temp
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
dir
Then dir will show the cloned folder name — cd into it, and continue from Step 4 onward.

Tell me: What does dir show when you run it inside C:\capstone_temp? I'll give you the exact next command.

when I run dir inside C:\capstone_temp I got the output as shown in Image

image-12.png
My Assistant
Completed 1 step
Your repo folder is Paul-Alex-Mathe-Capstone-Project. Here's what to do next:

Continue from here:
Step 1: Navigate into your repo
powershell





cd Paul-Alex-Mathe-Capstone-Project
Step 2: Verify your existing modules are there
powershell





dir
You should see your Module 1 and Module 2 folders listed. They're safe.

Step 3: Create the support_assistant folder structure
powershell





mkdir support_assistant
cd support_assistant
mkdir docs
mkdir app
Step 4: Create all the files
powershell





New-Item -Path "app\__init__.py" -ItemType File -Force
New-Item -Path "app\config.py" -ItemType File -Force
New-Item -Path "app\models.py" -ItemType File -Force
New-Item -Path "app\embeddings.py" -ItemType File -Force
New-Item -Path "app\prompt_template.py" -ItemType File -Force
New-Item -Path "app\graph.py" -ItemType File -Force
New-Item -Path "app\main.py" -ItemType File -Force
New-Item -Path "docs\doc_01.txt" -ItemType File -Force
New-Item -Path "docs\doc_02.txt" -ItemType File -Force
New-Item -Path "docs\doc_03.txt" -ItemType File -Force
New-Item -Path "docs\doc_04.txt" -ItemType File -Force
New-Item -Path "docs\doc_05.txt" -ItemType File -Force
New-Item -Path "docs\doc_06.txt" -ItemType File -Force
New-Item -Path "docs\doc_07.txt" -ItemType File -Force
New-Item -Path "docs\doc_08.txt" -ItemType File -Force
New-Item -Path "requirements.txt" -ItemType File -Force
New-Item -Path "Dockerfile" -ItemType File -Force
New-Item -Path "README.md" -ItemType File -Force
New-Item -Path ".gitignore" -ItemType File -Force
Step 5: Open in VS Code
powershell





cd ..
code.
This opens your entire repo in VS Code. You'll see your existing modules AND the new support_assistant/ folder in the left panel.

Step 6: Paste code into each file
In VS Code, expand support_assistant/ in the left panel and paste the code I gave you earlier into each file. Don't forget the .gitignore — paste this into it:

venv/
__pycache__/
*.pyc
chroma_db/
.env
Step 7: Set up venv and install packages
powershell





cd support_assistant
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Step 8: Test the app
powershell





uvicorn app.main:app --host 0.0.0.0 --port 7860
Step 9: Push to GitHub
powershell





# Stop server with Ctrl+C, go to repo root
cd C:\capstone_temp\Paul-Alex-Mathe-Capstone-Project

git add support_assistant/
git status
git commit -m "Add Module 3: Support Assistant - RAG pipeline with LangGraph + FastAPI"
git push origin main
Your Module 1 and Module 2 are completely untouched — you're only adding new files. Let me know what you see after running dir in Step 2!

http://0.0.0.0:7860/ is not opening

My Assistant
That's a Windows-specific issue. 0.0.0.0 doesn't work in a browser on Windows. Use localhost instead.

Fix: Open this URL in your browser
http://localhost:7860/
Or specifically test the endpoints:

http://localhost:7860/health
http://localhost:7860/docs
/docs opens the FastAPI Swagger UI where you can test the /ask endpoint directly in your browser!

To test via PowerShell (open a second terminal):
powershell





# Health check
Invoke-RestMethod -Uri "http://localhost:7860/health" -Method Get

# Policy question
$body = '{"query": "What is the delivery fee?"}'
Invoke-RestMethod -Uri "http://localhost:7860/ask" -Method Post -Body $body -ContentType "application/json"

# General question
$body = '{"query": "What is the weather today?"}'
Invoke-RestMethod -Uri "http://localhost:7860/ask" -Method Post -Body $body -ContentType "application/json"
Quick Explanation
Table



Address


Works on Windows?


http://0.0.0.0:7860	❌ No (0.0.0.0 means "listen on all interfaces" — it's for the server, not the browser)
http://localhost:7860	✅ Yes
http://127.0.0.1:7860	✅ Yes
View more
The server is running correctly — you just need to access it via localhost. Try opening http://localhost:7860/docs in your browser and let me know what you see!

ERR_ADDRESS_INVALID (-108)

Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given

Failed to send telemetry event ClientCreateCollectionEvent: capture() takes 1 positional argument but 3 were given

Failed to send telemetry event CollectionAddEvent: capture() takes 1 positional argument but 3 were given

My Assistant
The telemetry errors are just warnings from ChromaDB — they don't affect functionality. The real issue is the browser address. Let me clarify:

The telemetry messages are harmless
Those Failed to send telemetry event messages are just ChromaDB trying (and failing) to send anonymous usage stats. Your app is still running fine. They don't break anything.

The actual issue: Browser URL
You typed [IP_ADDRESS] in your browser. That won't work. Instead open:

http://[IP_ADDRESS]:7860/docs
But if [IP_ADDRESS]:7860 also gives an error:
Check what your terminal actually shows when uvicorn starts. Look for the line that says:

INFO:     Uvicorn running on http://...
If it shows an error before that line, the server didn't start. In that case, try running with 127.0.0.1 instead:

powershell





# Stop the current server with Ctrl+C first, then:
uvicorn app.main:app --host 127.0.0.1 --port 7860
Then open in browser: http://127.0.0.1:7860/docs

To test via PowerShell (most reliable on Windows):
Open a second PowerShell window and run:

powershell





Invoke-RestMethod -Uri "http://127.0.0.1:7860/health" -Method Get
If you get back {"status":"healthy","mock_llm":true} — your app is working perfectly!

Then test the main endpoint:

powershell





$body = '{"query": "What is the delivery fee?"}'
Invoke-RestMethod -Uri "http://127.0.0.1:7860/ask" -Method Post -Body $body -ContentType "application/json"
Tell me:

Does the uvicorn terminal show 🟢 Support Assistant ready! ?
What happens when you run Invoke-RestMethod from a second terminal?
I want to remove the existing requirements.txt files from each module in the git repository and add a new requirements.txt file for the overall project, which contains all the dependancies. what can I do?

My Assistant
Here's exactly how to do that, Paul Alex:

Step-by-Step: Consolidate into One Root requirements.txt
Step 1: Navigate to your repo root
powershell





cd C:\capstone_temp\Paul-Alex-Mathe-Capstone-Project
Step 2: Remove individual requirements.txt files from each module
powershell





# Remove from module folders (adjust folder names to match yours)
# First check what folders you have:
dir

# Then remove requirements.txt from each module that has one:
git rm module_1/requirements.txt
git rm module_2/requirements.txt
git rm support_assistant/requirements.txt
⚠️ Replace module_1, module_2 with your actual folder names. If any doesn't have a requirements.txt, just skip it — the command will tell you "did not match any files".

Step 3: Create one root-level requirements.txt
powershell





New-Item -Path "requirements.txt" -ItemType File -Force
Step 4: Open it in VS Code and paste all combined dependencies
powershell





code requirements.txt
Paste this (combines all modules into one file):

Root requirements.txt (All Modules Combined)






# ============================================
# Combined requirements for all modules
# ============================================

# --- Core / Shared ---
numpy
pandas
pydantic==2.9.2

# --- Module 1 & 2 dependencies ---
# (Add any packages your Module 1 and 2 use here)
# Example:
# scikit-learn
# matplotlib
# jupyter

# --- Module 3: Support Assistant ---
fastapi==0.115.0
uvicorn==0.30.6
sentence-transformers==3.1.1
chromadb==0.5.5
langgraph==0.2.28
langchain-core==0.3.6
groq==0.11.0
Important: Check your old Module 1 and Module 2 requirements.txt files BEFORE deleting them. Add any packages they listed into the combined file above (under the "Module 1 & 2 dependencies" section).

Step 5: Check your old requirements before they're gone
If you want to see what was in them first:

powershell





# View contents before removing (adjust folder names)
cat module_1/requirements.txt
cat module_2/requirements.txt
Copy any packages you see and add them to the root requirements.txt.

Step 6: Commit and push
powershell





git add requirements.txt
git status

# You should see:
#   deleted: module_1/requirements.txt
#   deleted: module_2/requirements.txt
#   deleted: support_assistant/requirements.txt
#   new file: requirements.txt

git commit -m "Consolidate all requirements.txt into single root-level file"
git push origin main
Final Repo Structure
Paul-Alex-Mathe-Capstone-Project/
├── module_1/              ← no more requirements.txt inside
├── module_2/              ← no more requirements.txt inside
├── support_assistant/     ← no more requirements.txt inside
├── requirements.txt       ← ONE file for everything ✅
└── README.md
One thing to note
Your support_assistant/Dockerfile references requirements.txt with COPY requirements.txt . — you'll need to update the Dockerfile to copy from the parent directory. Update this line in support_assistant/Dockerfile:

dockerfile





# Change this line:
COPY requirements.txt.

# To this:
COPY ../requirements.txt.
OR keep a copy of requirements.txt in support_assistant/ just for Docker, and remove it from git tracking. Let me know which approach you prefer!

I need a readme.md file for the overall project. attached are the 3 readme.md files for the each 3 modules. Give the content that I can directly paste in the Consolidated project readme.md file.

Readme.md
README.md
1+
My Assistant
Completed 1 step
Now I have all the content from your three module READMEs. Here's the consolidated root-level README.md:

Root README.md - Consolidated Project






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