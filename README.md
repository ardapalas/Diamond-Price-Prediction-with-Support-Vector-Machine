# 💎 Diamond Price Prediction

This project aims to predict diamond prices based on their physical characteristics (carat, cut, color, clarity, dimensions) using Machine Learning algorithms.

The project demonstrates the end-to-end data science lifecycle, from data cleaning and EDA to building models using **Linear Regression** and optimizing **Support Vector Regression (SVR)** using GridSearchCV.

## 📊 Dataset

 The dataset used in this project is the classic **Diamonds Dataset**.
* **Source:** [Kaggle - Diamonds Dataset](https://www.kaggle.com/datasets/shivam2503/diamonds)
* **Features:** Carat, Cut, Color, Clarity, Depth, Table, Price, X, Y, Z.
* **Target Variable:** Price

## 🛠️ Project Workflow

### 1. Data Cleaning & Preprocessing
* Dropped unnecessary columns (`Unnamed: 0`).
* Removed dimensionless entries (where x, y, or z were 0).
* **Outlier Removal:** Manually filtered extreme outliers in dimensions (`x`, `y`, `z`), `table`, and `depth` to improve model stability.
* **Encoding:** Applied `LabelEncoder` to categorical features (`Cut`, `Color`, `Clarity`).
* **Scaling:** Applied `StandardScaler` to normalize the data, which is crucial for SVR performance.

### 2. Exploratory Data Analysis (EDA)
* Correlation Heatmaps to understand feature relationships.
* Boxplots to analyze price distribution across clarity levels.
* Scatterplots to visualize relationships between dimensions/table/depth and price.

### 3. Modeling & Evaluation

Two main algorithms were implemented and compared:

#### A. Linear Regression
* Used as a baseline model.
* **Performance:**
    * R² Score: ~0.88
    * MAE: ~850

#### B. Support Vector Regression (SVR) - *The Highlight*
* **Initial Model:** Standard SVR yielded poor results (R² ~0.26) due to default parameters not fitting the data complexity.
* **Hyperparameter Tuning:** Used `GridSearchCV` with `StratifiedKFold` to find the best parameters:
    * `C`: [1, 10, 100, 1000]
    * `kernel`: ['rbf', 'linear']
    * `gamma`: [0.01, 0.1, 1]
* **Optimized Results:** Significant performance boost after tuning.
    * **R² Score: ~0.94** 🚀
    * **MAE: ~500**

## 📈 Results Summary

| Model | R2 Score | MAE (Mean Absolute Error) |
|-------|----------|---------------------------|
| Linear Regression | 0.88 | 850.60 |
| SVR (Base) | 0.26 | 1813.41 |
| **SVR (Tuned)** | **0.94** | **500.08** |

*The optimized SVR model significantly outperformed the Linear Regression model, reducing the error by nearly 41%.*

## 🧰 Technologies Used

* **Python 3.x**
* **Pandas & NumPy:** Data manipulation.
* **Matplotlib & Seaborn:** Data visualization.
* **Scikit-Learn:** Modeling, Preprocessing, and Tuning.
