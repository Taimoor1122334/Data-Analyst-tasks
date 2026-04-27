#  Data Science & Analytics Internship Tasks

Welcome to your internship project! This workspace contains everything you need to complete the five data science tasks. Each task is in its own folder with all the files you'll need.

---

## Project Overview

You're going to work through 5 different machine learning projects. You need to complete at least 3 of them, but we've set up all 5 so you can do as many as you want. These projects cover the main skills you'll use as a data scientist—loading data, exploring it, building models, and understanding your results.

**Deadline:** May 15, 2026

---

## What's Inside Each Task Folder

We've organized things so each task has its own folder. Here's what you'll find:

### Task 1: Exploring and Visualizing a Simple Dataset
**Folder:** `task01_exploring_and_visualizing_a_simple_dataset/`

Start here if you're new to data exploration. You'll work with the Iris dataset (flowers) and learn how to:
- Load and inspect data using pandas
- Look at basic stats and summaries
- Create scatter plots, histograms, and box plots to understand your data

**Files in this folder:**
- `README.md` - full instructions
- `task_notes.md` - space for your notes and planning
- `task01_exploring_and_visualizing_a_simple_dataset.py` - helper functions for data loading and plotting
- `task01_exploring_and_visualizing_a_simple_dataset.ipynb` - Jupyter notebook where you'll run the code

---

### Task 2: Credit Risk Prediction
**Folder:** `task02_credit_risk_prediction/`

This one's about predicting whether a loan applicant will default. You'll learn:
- How to handle missing data and clean datasets
- Build a classification model (Logistic Regression or Decision Tree)
- Evaluate your model using accuracy and confusion matrix
- Visualize key features like loan amount and income

**Files in this folder:**
- `README.md` - task details and requirements
- `task_notes.md` - your workspace for notes
- `task02_credit_risk_prediction.py` - code for loading loan data and training models
- `task02_credit_risk_prediction.ipynb` - notebook to run everything

---

### Task 3: Customer Churn Prediction
**Folder:** `task03_customer_churn_prediction/`

Predict which customers will leave the bank. You'll work with:
- Real banking customer data
- Encoding categorical variables (like location and gender)
- Random Forest classifier
- Feature importance analysis to understand what drives churn

**Files in this folder:**
- `README.md` - instructions
- `task_notes.md` - for your thoughts and ideas
- `task03_customer_churn_prediction.py` - helper functions for data prep and modeling
- `task03_customer_churn_prediction.ipynb` - notebook for execution

---

### Task 4: Predicting Insurance Claim Amounts
**Folder:** `task04_predicting_insurance_claim_amounts/`

Switch gears to regression—predicting how much someone's medical insurance will cost. You'll cover:
- Linear Regression modeling
- Visualizing relationships (age, BMI, smoking status vs. charges)
- Evaluating performance with MAE and RMSE
- Understanding how different factors impact insurance costs

**Files in this folder:**
- `README.md` - full breakdown
- `task_notes.md` - your working space
- `task04_predicting_insurance_claim_amounts.py` - functions for loading and visualizing insurance data
- `task04_predicting_insurance_claim_amounts.ipynb` - notebook

---

### Task 5: Personal Loan Acceptance Prediction
**Folder:** `task05_personal_loan_acceptance_prediction/`

The final task—predict which customers will accept a personal loan offer. You'll:
- Explore customer demographics (age, job, marital status)
- Build a classification model
- Extract business insights about which customer groups are most likely to accept

**Files in this folder:**
- `README.md` - instructions
- `task_notes.md` - notes area
- `task05_personal_loan_acceptance_prediction.py` - data loading and model code
- `task05_personal_loan_acceptance_prediction.ipynb` - main notebook

---

## How to Use This Setup

1. **Pick a task** - Start with any one that interests you (or start with Task 1 if you're new)

2. **Open the Jupyter notebook** - Find the `.ipynb` file in that task folder

3. **Follow the cells** - Each notebook has:
   - An introduction explaining the problem
   - Code cells that import from the `.py` file
   - Cells that load and explore the data
   - Cells for building and testing the model
   - Cells for visualization and evaluation

4. **Run the code** - Execute each cell and see the results

5. **Take notes** - Use the `task_notes.md` file to jot down what you learned or any questions

6. **Document your work** - The notebook is where everything comes together—your exploration, your model results, your conclusions

---

## File Structure

```
data analyst tasks/
├── Main Readme file.md                          ← You are here
├── TASKS_SUMMARY.md                             ← Quick summary of all tasks
├── setup_tasks.py                               ← Helper script for setup
│
├── task01_exploring_and_visualizing_a_simple_dataset/
│   ├── README.md
│   ├── task_notes.md
│   ├── task01_exploring_and_visualizing_a_simple_dataset.py
│   └── task01_exploring_and_visualizing_a_simple_dataset.ipynb
│
├── task02_credit_risk_prediction/
│   ├── README.md
│   ├── task_notes.md
│   ├── task02_credit_risk_prediction.py
│   └── task02_credit_risk_prediction.ipynb
│
├── task03_customer_churn_prediction/
│   ├── README.md
│   ├── task_notes.md
│   ├── task03_customer_churn_prediction.py
│   └── task03_customer_churn_prediction.ipynb
│
├── task04_predicting_insurance_claim_amounts/
│   ├── README.md
│   ├── task_notes.md
│   ├── task04_predicting_insurance_claim_amounts.py
│   └── task04_predicting_insurance_claim_amounts.ipynb
│
└── task05_personal_loan_acceptance_prediction/
    ├── README.md
    ├── task_notes.md
    ├── task05_personal_loan_acceptance_prediction.py
    └── task05_personal_loan_acceptance_prediction.ipynb
```

---

## What You Need to Have Installed

Make sure you have these libraries before you start:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

If you're missing something, just run that install command and you're good to go.

---

## What You'll Learn

- **Data Loading & Cleaning** - How to load datasets and handle missing/messy data
- **Exploratory Data Analysis (EDA)** - Techniques for understanding data through visualizations
- **Classification Models** - Logistic Regression, Decision Trees, Random Forests
- **Regression Models** - Linear Regression for predicting continuous values
- **Model Evaluation** - Using metrics like accuracy, confusion matrix, MAE, RMSE
- **Visualizations** - Creating meaningful plots with matplotlib and seaborn
- **Feature Importance** - Understanding which variables matter most

---

## Getting Started

1. Open any task folder you want to start with
2. Read the `README.md` for that task
3. Open the `.ipynb` notebook
4. Run each cell step by step
5. The `.py` file in the folder will provide helper functions—don't worry about writing everything from scratch

Each task takes about 1-2 hours if you follow along and understand what's happening.

---

## Tips for Success

- **Run the code** - Just reading won't help. Actually execute the cells and see what happens
- **Experiment** - Try changing parameters, creating different visualizations, testing different models
- **Take notes** - Use the `task_notes.md` to write down what you learn
- **Ask questions** - If something doesn't make sense, that's totally normal. Reach out to mentors
- **Document your findings** - Add comments to your notebook explaining your conclusions

Good luck! You've got this. 🚀
