# Atlantic Canada Crime Patterns Analysis

## Overview
This project analyzes aggregated crime data for all localities in the Atlantic Canadian provinces:  
**Nova Scotia, New Brunswick, Newfoundland and Labrador, and Prince Edward Island** over the period 2020–2024.  

The goal is to identify patterns in crime data and understand similarities and differences in criminal activity across these localities.

## Objectives
- Explore regional crime patterns across Atlantic Canada.  
- Analyze crime levels, offence structures, and growth trends over time.  
- Apply clustering and dimensionality reduction to summarize patterns.  
- Identify and highlight localities with high crime levels.

## Data
- **Source:** Statistics Canada – Table 35-10-0178-01: Criminal Code violations, by province and territory, 2020–2024  
- **Content:** Localities in Atlantic Canadian provinces, multiple offence categories, years 2020–2024  
- **Preprocessing (high-level):**  
  - Cleaned dataset (removed totals, non-city entries, and irrelevant metadata)  
  - Handled missing values and standardized province names  
  - Selected relevant offence categories for analysis  

## Methods
- **Clustering:** KMeans clustering to group localities by crime patterns.  
- **Dimensionality Reduction:** PCA for 2D visualization of localities.  
- **Analysis Types:**  
  1. **Absolute Levels:** Raw offence counts to identify low, medium, and high crime areas.  
  2. **Proportional Structure:** Relative composition of offence types to find similar crime patterns.  
  3. **Growth Patterns:** Year-over-year changes to detect localities with accelerating or declining crime.  

## Outputs
- **PCA scatter plots** visualizing clusters for each analysis type.  
- **Cluster summaries** with descriptive labels (e.g., Low/Medium/High Crime).  
- **Lists of High Crime localities** for each analysis.  
- **Silhouette scores** to assess cluster quality.

## How to Run

Run the main analysis script in Python (>=3.9) after ensuring the required packages are installed (`pandas`, `numpy`, `matplotlib`,  `seaborn`, `scikit-learn`):  

```bash
python crime_analysis.py



DNS-SFHA-Data-template/
├── README.md           # Project overview
├── video/              # A walkthrough video or link to video
├── answers/            # Written responses to the three questions
└── project/            # All the code, data, and outputs

