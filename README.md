# Fraud Detection — IEEE-CIS Dataset

> A machine learning project to find fraudulent transactions, using real data from Vesta Corporation's payment system.

\---

## What this project is about



This project tries to detect fraud in 590,540 real bank transactions. Only 3.5% of them are fraud.

Here the challenge is not just building a model, but understanding *why* every decision is made along the way.

This README explains all of my understanding during this project.

\---

## Phase 1 — Exploring the data



Like always before building anything, I looked at the raw data to understand what I was working with.

### What I found

|Thing I looked at|What I found|
|-|-|
|Fraud rate|Only 3.5% of transactions are fraud|
|Highest fraud rate by product|Category C|
|Highest transaction volume|Category W|
|Highest fraud rate by card type|Discover|
|Highest card volume|Visa|
|Columns with missing data|374 out of 394|

### Findings:



Most fraud transactions are for small amounts. Fraudsters do this on purpose — small transactions are less likely to trigger bank alerts. This showed up clearly when I plotted fraud vs legit transaction amounts side by side.



Category W has the most fraud transactions in total. But Category C has the highest *rate* of fraud. That is a big difference:

* High count just means that category is popular — more transactions overall
* High rate means that category is actually riskier per transaction

The model needs to learn from rates, not just raw counts.

\---

## Why accuracy is the wrong metric here



This is one of the most important things I learned in this project.

Only 3.5% of transactions are fraud. So if a model just says "legit" for every single transaction, it would be right 96.5% of the time. That sounds good but it would catch zero fraud. It would be completely useless.

So instead of accuracy, I used these four metrics:

**Precision** — out of all the transactions the model flagged as fraud, how many actually were fraud?

* Low precision = the model is blocking innocent customers all the time

**Recall** — out of all the real fraud cases, how many did the model catch?

* Low recall = the model is missing fraud and the bank is losing money

**F1 Score** — a single number that balances precision and recall together

**ROC-AUC** — measures how well the model separates fraud from legit overall. A score of 1.0 is perfect, 0.5 is no better than random guessing.



Precision and recall pull against each other. If you make the model more aggressive, it catches more fraud (recall goes up) but also blocks more innocent customers (precision goes down). Finding the right balance is a business decision, not just a math problem.

\---

## Phase 2 — Cleaning and preparing the data

### Merging two files into one



The data came in two separate files — transactions and identity info. I joined them together using a **left join** on TransactionID.



Why left join and not the other options?

* `inner join` would have thrown away 446,000 transactions that had no identity info which is too much data to lose
* `outer join` gives the same result as left join 
* `left join` keeps all 590,540 transactions and just fills in the identity columns where they exist

Result: 590,540 rows, 434 columns.

\---

### Filling in missing data



Problem: 374 out of 434 columns had missing values after merging. Most models crash if you give them missing data, so I had to fill everything in.



Learning:

**For number columns → I used the median (middle value)**

I should not use the average because financial data has big outliers. One $50,000 transaction can pull the average way up and make it a bad guess for missing values. The median is the middle value so extreme numbers do not affect it.



**For text columns → I used the mode (most common value)**

You cannot take a median of "Visa" or "Mastercard" since they are words. The most common value is the safest guess for what a missing entry probably was.

\---

### Converting text to numbers



Models only understand numbers and not words like "Visa" or "W". I used a Label Encoder to turn each unique word into a number:

```
"Amex"       → 0
"Discover"   → 1
"Mastercard" → 2
"Visa"       → 3
```

31 text columns were converted this way.

\---

### Splitting into training and test data



I split the data 80% for training and 20% for testing, 

Learning: 

Using `stratify=y`.

Why stratify? Without it, the random split might put most fraud cases in one set. Stratify makes sure both sets have the same 3.5% fraud rate, so now the test results reflect what would happen in the real world.

```
Training set:  472,432 rows — 3.50% fraud
Test set:      118,108 rows — 3.50% fraud
```

\---

### Fixing the class imbalance with SMOTE



Problem: Even after splitting, the training data only has 3.5% fraud. If I trained a model on that, it would barely see any fraud examples and would just learn to say "legit" for everything.



Learning:

SMOTE (Synthetic Minority Oversampling Technique) fixes this by creating new fake-but-realistic fraud transactions:

1. It takes a real fraud transaction
2. Finds other similar fraud transactions nearby
3. Creates a new fraud transaction somewhere in between them



**Important:** SMOTE should only go on the training data. The test data must stay 100% real. If I tested on fake transactions, the results would be misleading.

```
Before SMOTE: 16,530 fraud  vs 455,902 legit
After SMOTE:  455,902 fraud vs 455,902 legit
```

\---

## Phase 3 — First models (baselines)

I started with two simple models to get a starting point to compare better models against.

### Model 1 — Logistic Regression



Learning: 

This model draws a straight line to separate fraud from legit. It is simple and fast but struggles with complex patterns.

|Metric|Score|
|-|-|
|Fraud Precision|0.10|
|Fraud Recall|0.67|
|Fraud F1|0.18|
|ROC-AUC|0.7902|



Problem: It catches 67% of fraud (decent recall) but for every 10 fraud flags it raises, only 1 is actually fraud (terrible precision). 



Learning: 

Why terrible precision? Fraud patterns are not a straight line. A transaction is suspicious because of a *combination* of a lot of things, could be small amount AND foreign email AND Discover card AND unusual hour. A straight line cannot capture combinations.

\---

### Model 2 — Decision Tree

Instead of one straight line, a Decision Tree asks a series of yes/no questions:

```
Is the amount < $50?
├── YES → Is it a Discover card?
│         ├── YES → FRAUD
│         └── NO  → Legit
└── NO  → Is ProductCD = C?
          ├── YES → FRAUD
          └── NO  → Legit
```

Findings using (max\_depth=12, random\_state=42)



|Metric|Score|
|-|-|
|Fraud Precision|0.30|
|Fraud Recall|0.47|
|Fraud F1|0.37|
|ROC-AUC|0.8146|



What this means: Precision jumped from 0.10 to 0.30 (3x better). But recall dropped from 0.67 to 0.47. The tree is more careful about what it flags, so it misses more fraud but causes fewer false alarms. This was the precision-recall tradeoff in action.



### Baseline comparison

|Model|Precision|Recall|F1|ROC-AUC|
|-|-|-|-|-|
|Logistic Regression|0.10|0.67|0.18|0.7902|
|Decision Tree|0.30|0.47|0.37|0.8146|



Learning:

Neither model is good enough yet. 30% precision still means 7 out of 10 fraud flags are innocent customers.

\---



## Tools used

|Tool|What it does|
|-|-|
|Python|Main programming language|
|Pandas|Loading and working with data|
|NumPy|Math operations|
|Matplotlib / Seaborn|Charts and visualizations|
|Scikit-learn|Models, data prep, and evaluation|
|XGBoost|Advanced gradient boosting model|
|imbalanced-learn|SMOTE for fixing class imbalance|
|SHAP|Explaining model decisions|

\---

## How to run this project

```bash
git clone https://github.com/piyaashraya/Fraud-detection.git
cd Fraud-detection
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap imbalanced-learn jupyter
```

Download the data from [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and put `train\_transaction.csv` and `train\_identity.csv` in the `data/` folder.



\---

*Built by Ashraya Piya — Data Science @ Drexel University*

