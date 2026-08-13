
# Zepto Data & AI Platform — Capstone Project

**Certificate Program in Artificial Intelligence and Machine Learning**

An end-to-end AI/ML platform comprising three interconnected modules: a data-engineering pipeline, an analytics pipeline, and a GenAI support assistant.

---

## Repository Structure

```
zepto-ai-platform/
│
├── README.md                  ← You are here (root documentation)
├── data_pipeline/             ← Module 1: Scrape → Clean → Store → Query (25 marks)
│   ├── README.md
│   ├── requirements.txt
│   ├── data_pipeline.py       ← Main pipeline script
│   ├── scraped_books_raw.csv  ← Generated: raw scraped data
│   └── books_pipeline.db      ← Generated: SQLite database
│
├── analytics/                 ← Module 2: EDA → Modeling → Evaluation (50 marks)
│   └── (coming soon)
│
└── support_assistant/         ← Module 3: GenAI Support Assistant (25 marks)
    └── (coming soon)
```

---

## Quick Start

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/zepto-ai-platform.git
cd zepto-ai-platform

# Install Module 1 dependencies
pip install -r data_pipeline/requirements.txt
```

### Run Module 1 (Data Pipeline)

```bash
python data_pipeline/data_pipeline.py
```

This will:
1. Scrape 60+ books from books.toscrape.com across 5 categories
2. Clean all fields to proper types
3. Convert prices from GBP to INR (1 GBP = 105.50 INR — fixed project constant)
4. Create a normalized SQLite database with 2 tables
5. Run 6 SQL queries demonstrating all required clauses
6. Verify SQL JOIN matches pandas merge

---

## Currency Conversion Rate

**1 GBP = 105.50 INR** — This is a fixed, project-defined constant for this assignment. It is not a live or historical market rate and requires no external API, network access, or date reference.

---

## Design Decisions Summary

| Module | Key Decisions |
|--------|--------------|
| Data Pipeline | Median imputation for failed parses; 5 categories scraped for safety margin above 60 books; 0.5s polite delay between requests |
| Analytics | (To be added) |
| Support Assistant | (To be added) |

---

## Git Workflow

This repository demonstrates proper Git workflow:
- Feature branches created for development
- Multiple commits per branch
- Merged back into `main` via merge commits
- Viewable via `git log --graph --all`
