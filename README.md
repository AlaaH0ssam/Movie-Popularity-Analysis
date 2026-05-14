# 🎬 Movie Popularity Prediction  
### Pattern Recognition Project 2026 | Team CS_1

A machine learning project focused on predicting movie popularity using both **Regression** and **Multiclass Classification** techniques.  
The project explores how movie metadata such as genres, production companies, release dates, vote counts, and financial information influence audience popularity.

Built as part of the **Pattern Recognition / Machine Learning Projects 2026** coursework.

---

## 📌 Project Overview

This project is divided into two milestones:

### 📈 Milestone 1 → Regression
Predict the continuous numerical value of movie popularity.

### 🎯 Milestone 2 → Classification
Classify movies into one of four popularity categories:

- Very Low
- Low
- Medium
- High

---

## 🧠 Main Objectives

- Build a complete ML pipeline from raw data to prediction
- Apply preprocessing and feature engineering techniques
- Experiment with multiple regression and classification models
- Perform feature selection and hyperparameter tuning
- Evaluate models using proper metrics and visualizations
- Save trained models and preprocessing steps for unseen test datasets

---

# 📂 Dataset Features

The dataset contains movie metadata such as:

- Genres
- Production Companies
- Production Countries
- Spoken Languages
- Budget
- Revenue
- Vote Count
- Release Date
- Overview
- Posters / Backdrops / Homepage
- Adult Flag
- Runtime-related metadata
- Titles & Original Titles

---

# ⚙️ Preprocessing Pipeline

The preprocessing stage was designed to transform noisy real-world movie metadata into a clean machine-learning-ready dataset.

## 🔹 Data Cleaning
- Removed duplicate rows
- Removed rows with missing target values

## 🔹 Feature Engineering
Created additional informative features such as:

- `is_title_changed`
- `title_length`
- `overview_length`
- `has_backdrop`
- `has_homepage`
- `has_poster`
- `has_tagline`

## 🔹 Temporal Feature Extraction
Extracted:
- `release_day`
- `release_month`
- `release_year`

from the original release date.

## 🔹 Multi-label Handling
Processed:
- genres
- production companies
- countries
- languages

by extracting:
- first category
- number of categories

## 🔹 Missing Values Handling
- Numerical → Mean Imputation
- Categorical → Mode Imputation

## 🔹 Skewness Correction
Applied `log1p()` transformation on:
- budget
- revenue
- vote_count

## 🔹 Frequency Encoding
Encoded high-cardinality categorical features using frequency encoding.

## 🔹 Outlier Treatment
Applied Winsorization using:
- 1st percentile
- 99th percentile

## 🔹 Feature Scaling
Applied:
- Z-score Standardization

## 🔹 Feature Selection
Used:
- `SelectKBest(f_regression)` for Regression
- `SelectKBest(f_classif)` for Classification

---

# 📊 Milestone 1 → Regression Models

## Models Used

### 🌲 Random Forest Regressor
Ensemble-based model capable of handling nonlinear relationships and reducing overfitting.

### 🌳 Decision Tree Regressor
Tree-based regression model with interpretable decision paths.

---

## 📈 Regression Results

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Random Forest | 0.5200 | 6.4667 | 0.4539 |
| Decision Tree | 0.5267 | 7.3917 | 0.2865 |

✅ **Best Regression Model:** Random Forest Regressor

---

# 🎯 Milestone 2 → Classification Models

## Models Used

- Logistic Regression
- Linear SVC
- Decision Tree Classifier
- Tuned Decision Tree
- Random Forest Classifier
- Tuned Random Forest

---

## 🛠 Hyperparameter Tuning

Applied:
- `GridSearchCV`
- `RandomizedSearchCV`
- Cross Validation

Optimized parameters such as:
- `max_depth`
- `min_samples_leaf`
- `n_estimators`

Evaluation metric:
- `f1_weighted`

---

## 🏆 Best Classification Model

✅ **Random Forest Classifier**  
Accuracy: **82.59%**

---

# 📉 Visualizations

The project includes several visual analysis plots:

- Distribution of Popularity
- Correlation Heatmaps
- Actual vs Predicted
- Residual Plots
- Error Distribution
- Feature Importance
- Confusion Matrices
- Accuracy Comparison
- Training Time Comparison
- Testing Time Comparison

---

# 🧪 Model Evaluation Metrics

## Regression
- MAE
- RMSE
- R² Score

## Classification
- Accuracy
- Classification Report
- Confusion Matrix
- F1 Weighted Score

---

# 💾 Model Persistence

The project saves:
- trained models
- preprocessing objects
- feature selection steps
- scaling parameters

using:
```python
pickle
```

Saved files include:
- `best_classification_model.pkl`
- `stepsForPreprocessing.pkl`
- `ms1_regression_data.pkl`

---

# 🚀 Running The Project

## 1️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 2️⃣ Run Milestone 1 (Regression)

```bash
python ms1.py
```

---

## 3️⃣ Run Milestone 2 (Classification)

```bash
python ms2.py
```

---

## 4️⃣ Run Test Script On Unseen Data

```bash
python testscript.py
```

The script automatically:
- loads preprocessing steps
- loads saved models
- preprocesses unseen test data
- predicts outputs
- prints evaluation metrics

---

# 📁 Suggested Project Structure

```bash
Movie-Popularity-Prediction/
│
├── train_data.csv
├── unseenTestSample.csv
│
├── ms1.py
├── ms2.py
├── testscript.py
│
├── best_classification_model.pkl
├── stepsForPreprocessing.pkl
├── ms1_regression_data.pkl
│
├── README.md
└── report.pdf
```

---

# 🔍 Key Insights

🧠 Initial assumptions suggested that movie budget alone determines popularity.

📊 The analysis revealed that:
- vote count
- release year
- audience engagement
- metadata categories

have stronger predictive power than financial information alone.

The project demonstrated that movie popularity is a complex nonlinear problem best handled using ensemble learning techniques.

---

# 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pickle

---

# 👥 Team Information

### Team ID
**CS_1**

### Course
Pattern Recognition / Machine Learning Projects 2026

---

# 🌟 Final Conclusion

This project successfully built an end-to-end machine learning pipeline capable of handling real-world movie metadata for both regression and classification tasks.

The final tuned ensemble models achieved strong generalization performance while maintaining efficient preprocessing and scalable deployment through saved preprocessing pipelines and serialized models.

The dataset turned out to be less about “big budget = big popularity” and more like a cinematic ecosystem where timing, audience engagement, and metadata quietly pull strings behind the curtain 🎥✨
