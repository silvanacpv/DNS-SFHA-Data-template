# Atlantic Canada Crime Patterns Analysis

## Overview

This project analyzes aggregated crime data for all localities in the Atlantic Canadian provinces:  
**Nova Scotia, New Brunswick, Newfoundland and Labrador, and Prince Edward Island** over the period **2020–2024**.  

The main goal is to **identify patterns in crime data** and understand structural similarities and differences in criminal activity across these localities.

---

## Objectives

1. Explore regional crime patterns across Atlantic Canada.  
2. Analyze crime levels, offence structures, and growth trends over time.  
3. Use dimensionality reduction and visualization to summarize patterns.  
4. Provide interpretable insights about regional crime profiles.

---

## Data

- **Source:** [Statistics Canada – Table 35-10-0178-01: Criminal Code violations, by province and territory, 2020–2024](https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=3510017801)  
- **Content:** Localities in Atlantic Canadian provinces, multiple offence categories, years 2020–2024  
- **Preprocessing includes:**  
  - Removing metadata rows and unnecessary columns  
  - Handling missing values   
  - Standardizing province names to abbreviations  

---

## Methods

## 1. Data Processing
- Clean the dataset (remove irrelevant rows, handle missing values)
- Rename and structure columns for analysis

## 2. Analysis & Visualization
- Scale numeric features for comparability
- Apply dimensionality reduction (PCA) to visualize patterns
- Cluster localities to identify crime levels and growth patterns
- Summarize trends in offence types and yearly changes

## 3. Outputs
- PCA scatter plots highlighting differences between localities
- Summary tables of offence levels by locality and year
- Lists of High Crime localities for each analysis

---

## How to Run

Run the main analysis script in Python (>=3.9) after ensuring the required packages are installed (`pandas`, `numpy`, `matplotlib`,  `seaborn`, `scikit-learn`):  

```bash
python crime_analysis.py




DNS-SFHA-Data-template/
├── README.md           # Project overview
├── video/              # A walkthrough video or link to video
├── answers/            # Written responses to the three questions
└── project/            # All the code, data, and outputs

