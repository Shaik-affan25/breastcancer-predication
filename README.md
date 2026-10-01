# Task 1: Data Cleaning & Preprocessing (Breast Cancer Dataset)

## Objective
Clean and prepare raw data for machine learning.

## Dataset
- File: `data.csv` (Breast Cancer Wisconsin diagnostic data)
- Original size: 569 rows × 33 columns
- Target: `diagnosis` (M = Malignant, B = Benign)

## Steps Performed
1. **Data exploration:** used `df.shape`, `df.dtypes`, `df.info()` and `df.isnull().sum()` to check size, data types and null values.
2. **Handling missing values:** found no missing values in the real features. Dropped the empty extra column `Unnamed: 32`.
3. **Feature/target split:** removed `id` and separated `diagnosis` as the target (y) and 30 numerical columns as features (x).
4. **Encoding:** converted the categorical target using label encoding: `M → 1`, `B → 0`.
5. **Outlier detection:** plotted a boxplot of all features before cleaning (`area_mean` and `area_worst` showed the most outliers).
6. **Outlier removal:** used the IQR method (lower = Q1 − 1.5×IQR, upper = Q3 + 1.5×IQR) and removed rows with outliers. Rows reduced from **569 to 398**.
7. **Visualization after removal:** plotted the boxplot again to compare.
8. **Feature scaling:** standardized the features with `StandardScaler` (mean = 0, std = 1).

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Files
- `TASK_1.ipynb`: complete code and outputs
- `data.csv`: dataset used

## Result
Final cleaned data: **398 rows × 30 features**, fully numerical and standardized, ready for model training.
