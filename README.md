# 📊 DecodeLabs Internship — Task 1: Advanced EDA & Feature Engineering

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Ready-red?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Internship](https://img.shields.io/badge/DecodeLabs-Internship%20Task%201-purple)

> 🏢 **Organization:** DecodeLabs | 🔢 **Task:** Project 1 of Data Science Track

---


## 📌 Project Overview

This repository contains my solution for **Task 1** of the DecodeLabs Data Science Internship.
The goal is to transform a raw, chaotic E-Commerce Orders dataset into a mathematically clean,
ML-ready dataset using industry-standard data engineering techniques.

The entire pipeline follows the **Input → Process → Output (IPO) Architecture** taught in the DecodeLabs training kit.

---

## 🗂️ Repository Structure

```
decodelabs-internship-task1-eda/
│
├── 📓 Project_1EDA.ipynb              # Main Jupyter Notebook (full solution)
├── 📄 order_data.csv                  # Original raw dataset (input)
├── 📄 cleaned_orders_dataset.csv      # Final cleaned ML-ready dataset (output)
│
├── 📁 plots/
│   ├── 🖼️ missing_values.png          # Missing values bar chart
│   ├── 🖼️ distributions.png           # Numeric column distributions
│   ├── 🖼️ zscore_outliers.png         # Z-Score outlier detection
│   ├── 🖼️ outlier_boxplots.png        # Before vs After IQR Winsorization
│   └── 🖼️ correlation_matrix.png      # Multicollinearity heatmap
│
└── 📄 README.md                       # Project documentation
```

---

## 📋 Dataset Information

| Feature | Details |
|---------|---------|
| **Dataset** | E-Commerce Orders |
| **Original Rows** | 1200 |
| **Original Columns** | 14 |
| **Final Columns** | 32 |
| **Key Columns** | OrderID, Date, CustomerID, Product, Quantity, UnitPrice, TotalPrice, PaymentMethod, OrderStatus, CouponCode, ReferralSource |

---

## ✅ Tasks Completed

### 🔍 Task 1 — Exploratory Data Analysis (EDA)
- Checked dataset shape, data types and column names
- Generated full statistical summary using `describe()`
- Identified and visualized missing values per column
- Plotted distribution histograms for all numeric columns

---

### 🧹 Task 2 — Missing Value Handling
Applied the **Missing Data Decision Matrix** from the DecodeLabs training kit:

| Missing % | Strategy | Reason |
|-----------|----------|--------|
| **< 5%** | Drop rows (`dropna`) | Preserves data volume, prevents synthetic bias |
| **5% – 20%** | Median / Mode imputation | Robust against extreme outliers |
| **> 20%** | KNN Imputation | Captures multi-dimensional relationships |

---

### 📈 Task 3 — Outlier Detection & Treatment

#### Z-Score Method
- Flagged all rows where **|Z-Score| > 3** as statistical anomalies
- Visualized Z-Scores per column using scatter plots

#### IQR Method + Winsorization
- Calculated **Q1, Q3, IQR** for every numeric column
- Applied **Winsorization** using `numpy.clip()` — caps values at boundaries instead of deleting rows

```
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

---

### 🔧 Task 4 — Feature Engineering (5 New Features)

| # | Feature Name | Formula / Logic | Purpose |
|---|-------------|-----------------|---------|
| 1 | `RevenuePerItem` | `TotalPrice / Quantity` | Average value per unit sold |
| 2 | `HasCoupon` | `1` if CouponCode present else `0` | Discount usage binary flag |
| 3 | `OrderMonth` | Month extracted from Date | Capture seasonal patterns |
| 4 | `IsHighValue` | `1` if TotalPrice > median | High value order flag |
| 5 | `DayOfWeek` | Day number from Date (0=Mon, 6=Sun) | Capture weekly patterns |

---

### 🔠 Task 5 — One-Hot Encoding
Converted categorical columns into numeric binary columns so ML models can process them:

| Column Encoded | Unique Values |
|----------------|--------------|
| `PaymentMethod` | Credit Card, Debit Card, Cash, Gift Card, Online |
| `Product` | Monitor, Phone, Tablet, Chair, Printer, Laptop, Desk |
| `ReferralSource` | Instagram, Referral, Email, Facebook, Google |
| `OrderStatus` | Shipped, Cancelled, Returned, Delivered, Pending |

---

### 🔗 Task 6 — Multicollinearity Check
- Built absolute correlation matrix using Pearson correlation
- Identified all feature pairs with correlation **> 0.80**
- Dropped the weaker column to eliminate redundancy

**Result:** `RevenuePerItem` was dropped — it had **1.00 correlation** with `UnitPrice` (same formula)

---

### ✅ Task 7 — Pandera Schema Validation
Runtime data contract validation using **Pandera**:

| Column | Validation Rule |
|--------|----------------|
| `TotalPrice` | Must be ≥ 0 |
| `Quantity` | Must be > 0 |
| `OrderMonth` | Must be in range 1–12 |
| `IsHighValue` | Must be 0 or 1 |
| `HasCoupon` | Must be 0 or 1 |

---

## 📊 Output Visualizations

| File | Description |
|------|-------------|
| `missing_values.png` | Bar chart — missing % per column with 5% and 20% thresholds |
| `distributions.png` | Histograms — distribution of all numeric columns |
| `zscore_outliers.png` | Scatter plots — Z-Score values per column |
| `outlier_boxplots.png` | Box plots — before vs after IQR Winsorization |
| `correlation_matrix.png` | Heatmap — multicollinearity detection |

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.0+ | Data loading, cleaning, manipulation |
| `numpy` | 1.24+ | Vectorized math operations |
| `matplotlib` | 3.7+ | Data visualization |
| `seaborn` | 0.12+ | Heatmaps and statistical plots |
| `scipy` | 1.10+ | Z-Score outlier detection |
| `scikit-learn` | 1.2+ | KNN Imputation |
| `pandera` | 0.32+ | Runtime schema validation |

---

## 🚀 How to Run

**1. Clone this repository**
```bash
git clone https://github.com/natashafatii/decodelabs-internship-task1-eda.git
cd decodelabs-internship-task1-eda
```

**2. Install required libraries**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy pandera
```

**3. Launch Jupyter Notebook**
```bash
jupyter notebook
```

**4. Open and run**
```
Project_1EDA.ipynb → Run All Cells (Shift + Enter)
```

---

## 📤 Output

After running all cells, two output files are generated:

- ✅ **`cleaned_orders_dataset.csv`** — Final ML-ready cleaned dataset
- ✅ **`plots/`** — All 5 visualization charts saved as PNG

---

## 📜 License

This project is part of the DecodeLabs Industrial Training Program.
For educational purposes only.

---

<div align="center">
Made with ❤️ by <a href="https://github.com/natashafatii">Natasha Fatima</a> | DecodeLabs Intern | Batch 2026
</div>
