# Fraud Detection — IEEE-CIS Dataset

> A machine learning project to find fraudulent transactions, using real data from Vesta Corporation's payment system.

\---

## What this project is about

This project tries to detect fraud in 590,540 real bank transactions. Only 3.5% of them are fraud.

The challenge is not just building a model, but understanding *why* every decision is made along the way.

This README explains everything I learned and understood during this project.

\---



## Phase 1 — Exploring the data

Before building anything, I looked at the raw data to understand what I was working with.

### What I found

|Thing I looked at|What I found|
|-|-|
|Fraud rate|Only 3.5% of transactions are fraud|
|Highest fraud rate by product|Category C|
|Highest transaction volume|Category W|
|Highest fraud rate by card type|Discover|
|Highest card volume|Visa|
|Columns with missing data|374 out of 394|

### Findings

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

Precision and recall pull against each other. If you make the model more aggressive, it catches more fraud (recall goes up) but also blocks more innocent customers (precision goes down). ---

## Phase 2 — Cleaning and preparing the data

### Merging two files into one

The data came in two separate files — transactions and identity info. I joined them together using a **left join** on TransactionID.

Why left join and not the other options?

* `inner join` would have thrown away 446,000 transactions that had no identity info — too much data to lose
* `outer join` gives the same result as left join here anyway
* `left join` keeps all 590,540 transactions and just fills in the identity columns where they exist

Result: 590,540 rows, 434 columns.

\---

### Filling in missing data

374 out of 434 columns had missing values after merging. Most models crash if you give them missing data, so I had to fill everything in.

**For number columns → I used the median (middle value)**

I did not use the average because financial data has big outliers. One $50,000 transaction can pull the average way up and make it a bad guess for missing values. The median is the middle value so extreme numbers do not affect it.

Example:

```
Transactions: $10, $12, $9, $11, $10,000
Average: $2,008  ← pulled way up by the big transaction
Median:  $11     ← actually represents a typical transaction
```

**For text columns → I used the mode (most common value)**

You cannot take a median of "Visa" or "Mastercard" as they are words. The most common value is the safest guess for what a missing entry probably was.

Result: 0 missing values remaining.

\---

### Converting text to numbers

Models only understand numbers, not words like "Visa" or "W". I used a Label Encoder to turn each unique word into a number alphabetically:

```
"Amex"       → 0
"Discover"   → 1
"Mastercard" → 2
"Visa"       → 3
```

31 text columns were converted this way.

\---

### Splitting into training and test data

I split the data 80% for training and 20% for testing using `stratify=y`.

Why stratify? Without it, the random split might put most fraud cases in one set. Stratify makes sure both sets have the same 3.5% fraud rate, so the test results reflect what would happen in the real world.

```
Training set:  472,432 rows — 3.50% fraud
Test set:      118,108 rows — 3.50% fraud
```

\---

### Fixing the class imbalance with SMOTE

Even after splitting, the training data only has 3.5% fraud. If I trained a model on that, it would barely see any fraud examples and would just learn to say "legit" for everything.

SMOTE (Synthetic Minority Oversampling Technique) fixes this by creating new fake-but-realistic fraud transactions:

1. It takes a real fraud transaction
2. Finds other similar fraud transactions nearby
3. Creates a new fraud transaction somewhere in between them

This is smarter than just copying the same fraud rows over and over — it adds real variety.

**Important rule:** SMOTE only goes on the training data. The test data must stay 100% real. If I tested on fake transactions, the results would be misleading.

```
Before SMOTE: 16,530 fraud  vs 455,902 legit
After SMOTE:  455,902 fraud vs 455,902 legit  ← perfectly balanced
```

\---

## Phase 3 — Baseline models

I started with two simple models to get a starting point to compare better models against.

### Model 1 — Logistic Regression

This model draws a straight line to separate fraud from legit. It is simple and fast but struggles with complex patterns.

|Metric|Score|
|-|-|
|Fraud Precision|0.10|
|Fraud Recall|0.67|
|Fraud F1|0.18|
|ROC-AUC|0.7902|

It catches 67% of fraud but for every 10 fraud flags it raises, only 1 is actually fraud. It is crying wolf constantly.

Why? Fraud patterns are not a straight line. A transaction is suspicious because of a *combination* of things — small amount AND foreign email AND Discover card AND unusual hour. A straight line cannot capture combinations.

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

|Metric|Score|
|-|-|
|Fraud Precision|0.30|
|Fraud Recall|0.47|
|Fraud F1|0.37|
|ROC-AUC|0.8146|

Precision jumped from 0.10 to 0.30 thats 3x better. But recall dropped. The tree is more careful about what it flags, so it misses more fraud but causes fewer false alarms. This is the precision-recall tradeoff in action.

### Baseline comparison

|Model|Precision|Recall|F1|ROC-AUC|
|-|-|-|-|-|
|Logistic Regression|0.10|0.67|0.18|0.7902|
|Decision Tree|0.30|0.47|0.37|0.8146|

Neither model is good enough yet. 30% precision still means 7 out of 10 fraud flags are innocent customers.

\---

## Phase 4 — Advanced models

### Why these models are better

**Random Forest — a team of Decision Trees**

Instead of one tree asking questions, Random Forest builds hundreds of trees. Each of theses are trained on a random slice of the data. Then every tree votes and the majority wins.

Like asking 500 doctors for a diagnosis instead of 1. The group is almost always more accurate than any single doctor.

**XGBoost — trees that learn from mistakes**

XGBoost also builds many trees but with one key difference. Each new tree focuses on fixing the mistakes of the previous one:

```
Tree 1 → makes some errors
Tree 2 → focuses on what Tree 1 got wrong
Tree 3 → focuses on what Trees 1 and 2 got wrong
...and so on
```

This is called gradient boosting. It is one of the most powerful techniques in machine learning and dominates fraud detection in the real world.

\---

### XGBoost settings

```python
XGBClassifier(
    n\\\\\\\_estimators=120,    # build 120 trees total
    max\\\\\\\_depth=8,         # each tree can ask 8 levels of questions
    learning\\\\\\\_rate=0.4,   # how aggressively each tree fixes the last one's mistakes
    random\\\\\\\_state=42,     # makes results the same every time you run it
    eval\\\\\\\_metric='logloss', # what to measure internally while training
    verbosity=0          # turns off the wall of text XGBoost prints while training
)
```

**Why `learning\\\\\\\_rate=0.4`?**
A high learning rate means each tree makes big corrections to the previous one's mistakes. Lower rates are more careful but need more trees. I experimented with different values and 0.4 gave the best results here.

**Why `verbosity=0`?**
Without this, XGBoost prints hundreds of lines of numbers while training. Setting it to 0 kept the output clean and readable.

\---

### Results

|Model|Precision|Recall|F1|ROC-AUC|
|-|-|-|-|-|
|Logistic Regression|0.10|0.67|0.18|0.7902|
|Decision Tree|0.30|0.47|0.37|0.8146|
|Random Forest|0.61|0.50|0.55|0.8935|
|XGBoost|0.90|0.59|0.71|0.9525|

The XGBoost is the best model. Precision hit 0.90 — 9 out of 10 fraud flags are real fraud. ROC-AUC jumped to 0.9525.

**Which model should a bank use?**

This is a real conversation that happens in fraud teams every day but from this result I can say XGBoost is the better option than anything else.

\---

## Phase 5 — Explaining the model's decisions

Getting a good model is only half the job. In real life, banks need to explain *why* a transaction was flagged to the customer, to regulators, and to their own team.

SHAP (SHapley Additive exPlanations) answers one question: *"for this specific transaction, which features pushed the model toward fraud and by how much?"*

### How SHAP works

Imagine a transaction comes in and the model says "fraud — 92% probability." SHAP breaks that down like this:

```
Starting point (average for all transactions):  +0.01
Email domain is unusual:                        +0.31  ← pushed toward fraud
Transaction amount is $12:                      +0.18  ← pushed toward fraud
Product category is C:                          +0.15  ← pushed toward fraud
Card type is Visa:                              -0.09  ← pushed toward legit
Hour of day is 2pm:                             -0.07  ← pushed toward legit
─────────────────────────────────────────────────────
Final fraud score:                               0.49  → flagged as fraud
```

Each number tells you exactly how much that feature pushed the prediction toward fraud (positive numbers) or toward legit (negative numbers). They all add up to the final answer.

### What the charts show

**Feature importance chart** — the top 20 features the model relies on most. C14 and C13 are the most important, followed by TransactionAmt.

**Dot plot** — shows not just which features matter but in which direction. Each dot is one transaction. Red = high feature value, Blue = low feature value. Position shows whether it pushed toward fraud or legit.

Key finding from the dot plot: low transaction amounts and low C14 values are the strongest signals pushing toward fraud.

**Waterfall chart** — explains one specific transaction step by step. For the transaction I picked:

* V87 pushed fraud score up by +0.97 (biggest red flag)
* V258 pushed it up by +0.84
* D3 tried to pull it back by -0.87
* But the red flags won, final score 1.787, flagged as fraud

This is the kind of explanation a bank could give to a regulator or a customer.

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

Download the data from [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and put `train\\\\\\\_transaction.csv` and `train\\\\\\\_identity.csv` in the `data/` folder.

```bash
jupyter notebook
```

\---

*Built by Ashraya Piya — Data Science @ Drexel University*

