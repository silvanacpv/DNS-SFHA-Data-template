#--------------------------------------------------------------------------
# Capstone Project:
# This project aims to identify patterns of criminal activity and group
# Atlantic Canadian provinces based on similarities in crime trends between
# 2018 and 2024.
# The central question guiding this analysis is:
# Can we identify meaningful patterns in crime data that allow Atlantic
# provinces to be grouped according to similar criminal profiles and trends?
# Rather than focusing on individual crime incidents, this project examines
# aggregated crime statistics across multiple offence categories.
# The goal is to understand structural similarities and differences between
# provinces over time.
#
#
# Student ID: DA21106
# Name: Silvana Paredes
#--------------------------------------------------------------------------

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

#--------------------------------------------------------------------------
# Loading the file
#--------------------------------------------------------------------------
df = pd.read_csv(
    "data.csv",
    sep=',',          # CSV is comma-separated
    skiprows=8,       # skip the first 8 rows (rows 0-7)
    nrows=7,          # read 8 rows starting from row 9 (rows 9-16)
    quotechar='"',    # handle commas inside quoted text
    encoding='utf-8-sig'
)

#--------------------------------------------------------------------------
# 1. Data Processing: 
#--------------------------------------------------------------------------
# Section A: Data cleansing
#--------------------------------------------------------------------------
# A. Drop rows and columns not required
#--------------------------------------------------------------------------
# Drop empty rows
df = df.drop(index=0)
df = df.drop(index=2)

# Drop columns by index range (1 to 7)
df = df.drop(df.columns[1:8], axis=1)

# Inspect the resulting DataFrame
print("1. Data Processing")
print(df.head(6))


#--------------------------------------------------------------------------
# B. Create column names in the format: offence_year.  
# Step 1: Append the year to offence names that are already present
#         in the first year.  
# Step 2: For subsequent years where the offence name is blank,
#         fill in the offence name and append the corresponding year.  
# This ensures that each column has a complete offence_year label
#--------------------------------------------------------------------------

years = df.iloc[0]             # row with years
original_columns = df.columns  # current name of columns

new_columns = []
current_crime = None

for i in range(len(original_columns)):
    col_name = original_columns[i]
    year_raw = years.iloc[i]

    # Check if year_raw is numeric
    if pd.api.types.is_number(year_raw):
        year = str(int(year_raw))  # convert 2018.0 -> "2018"
    else:
        year = str(year_raw).strip()  # keep as string (e.g., "Unnamed")

    # If column is not "Unnamed", it's the start of a new crime
    if not str(col_name).startswith("Unnamed"):
        current_crime = col_name.strip()

    new_columns.append(f"{current_crime}_{year}")

df.columns = new_columns

#--------------------------------------------------------------------------
# Section B: Data manipulation
#--------------------------------------------------------------------------
# C. Create a new dataset summarizing total counts of each offence across
# all years with the objective to:  
# identify offences with missing or non-representative values 
# (i.e., offences representing less than 0.01% of the total),  
# which will be removed in a later step.
#--------------------------------------------------------------------------
# Summarize offences across all years
rows_to_sum = slice(1, df.shape[0])  # all rows below header with actual data
num_years = 7  # years from 2018 to 2024

summary_data = {}

col_idx = 0
while col_idx < df.shape[1]:
    
    # Check if this column starts a 2018 block
    if str(df.iloc[0, col_idx]).strip() == "2018":
        # Ensure full block exists
        if col_idx + num_years - 1 >= df.shape[1]:
            break
        
        block_idx = list(range(col_idx, col_idx + num_years))
        
        # Crime name is the column name of the first column in the block
        crime_name = df.columns[col_idx]
        
        # Extract the block data (all rows of interest)
        block = df.iloc[rows_to_sum, block_idx]
        
        # Remove thousand separators (commas) and convert to float64
        block_numeric = block.replace(',', '', regex=True) \
                        .apply(pd.to_numeric, errors='coerce') \
                        .fillna(0).astype('float64')
        
        # Sum the values
        total_value = block_numeric.sum().sum()
        
        
        # Store in dictionary
        summary_data[crime_name] = [total_value]  # single row
        
        # Move to next block
        col_idx += num_years
    else:
        col_idx += 1

# Create summary DataFrame: 1 value per offence
offences_df = pd.DataFrame(summary_data)

# Remove the last 5 characters from offence names
offences_df.columns = [
    re.sub(r"_\d{4}$", "", col).rstrip()
    for col in offences_df.columns
]

# Delete the "Total" columns which came from the original file
offences_df = offences_df.iloc[:, 3:] 

# Print total number of offences
offences_df = offences_df.apply(pd.to_numeric, errors='coerce')
total = offences_df.sum(axis=1).iloc[0]
print("\nTotal number of offences:", total)
print("Total of offence types:", offences_df.shape[1])

#--------------------------------------------------------------------------
# D. Delete offences with non-representative values from the original dataset
#--------------------------------------------------------------------------
# 1. Get offences to exclude
#--------------------------------------------------------------------------
threshold = 0.01 #0.01% from the total of offences
min_todelete = total * threshold / 100

crimes_todelete = offences_df.columns[offences_df.iloc[0] < min_todelete].tolist()
crimes_todelete = [str(c).strip().lower() for c in crimes_todelete]

#print("\nExcluded offences:", crimes_todelete)
#for crime in crimes_todelete:
#    print(crime)
    
# Get number of offence types with non-representative values
num_todelete = (offences_df < min_todelete).sum().sum()
print("Total of offence types with fewer than ", min_todelete, "incidents", threshold, "%:", num_todelete)


#--------------------------------------------------------------------------
# 2. Get valid offences
#--------------------------------------------------------------------------
offences_df.drop(columns=[c for c in offences_df.columns if c.lower() in crimes_todelete], inplace=True)
print("Total of valid offences:", offences_df.shape[1])

#Print the valid offences to create the Categorization.csv file
#for col in offences_df.columns:
#    print(col)
    

#--------------------------------------------------------------------------
# 3. Delete offences from the main dataset
#--------------------------------------------------------------------------
print("\nOriginal dataset")
print("Shape before deleting offences:", df.shape)

# Identify columns to drop in df that start with the base name
cols_to_drop = [c for c in df.columns if any(c.lower().startswith(zero) for zero in crimes_todelete)]

# Drop these columns from df
df.drop(cols_to_drop, axis=1, inplace=True)

print("Shape after deleting offences:",df.shape)


#--------------------------------------------------------------------------
# E. Create offence categorization
#--------------------------------------------------------------------------

# Read the offence categories and category texts CSV with category assignments
categorization_df = pd.read_csv("categorization.csv", header=None, dtype=str)
categorization_df.columns = ["offence_name", "category_id", "category_text"]  

category_texts_df = pd.read_csv("categories.csv", header=None, dtype=str)
category_texts_df.columns = ["category_id", "category_text"] 

# Create a mapping from offence_name -> category
offence_to_category = {name.strip(): cat for name, cat in zip(categorization_df["offence_name"], categorization_df["category_id"])}

# Map category IDs to offences_df
category_ids = []
for col in offences_df:
    cat = offence_to_category.get(col.strip(), "Unknown")  # default if not found
    category_ids.append(cat)

offences_df.loc["category_id"] = category_ids

# Map category text
cat_text_dict = dict(zip(category_texts_df["category_id"], category_texts_df["category_text"]))
category_texts = [cat_text_dict.get(cat, "Unknown") if cat != "Unknown" else "Unknown" for cat in category_ids]

offences_df.loc["category_text"] = category_texts

#--------------------------------------------------------------------------
# F. Set offence categories in the main dataset
#--------------------------------------------------------------------------
# 1. Create category row in the main dataset
#--------------------------------------------------------------------------
# Add a new row to df with the category for each offence column
# Strip the last 5 characters (_year) from df to match the base name
df.loc['category_id'] = [
    offence_to_category.get(col[:-5].strip(), 'Unknown')
    for col in df.columns
]

#--------------------------------------------------------------------------
# 2. Create a new dataset summarized by category and year 
#--------------------------------------------------------------------------

# First row contains the year
years = df.iloc[0]

# Convert years to integer to avoid 2020 vs 2020.0 issue
years = pd.to_numeric(years, errors='coerce')  
years = years.dropna()                         
years = years.astype(int)                      

# Last row contains category_id
category_ids = df.iloc[-1]

# Replace category_id with category_name
category_names = category_ids.map(cat_text_dict)

# Create combined column names: CategoryName_Year
combined_cols = [
    f"{cat}_{year}"
    for cat, year in zip(category_names, years)
]

# Remove first (year) and last (category) rows
df_data = df.iloc[1:-1]
# Delete left column with labels 
df_data = df_data.drop(df_data.columns[0], axis=1)
# Convert to numeric
df_data = df_data.apply(pd.to_numeric, errors='coerce')
# Assign new column names
df_data.columns = combined_cols
# Group and sum duplicate Category_Year columns
df_data = df_data.T.groupby(df_data.columns, sort=False).sum().T

# Drop columns that start with 'nan'
cols_to_drop = [c for c in df_data.columns if str(c).startswith('nan')]
df_data.drop(cols_to_drop, axis=1, inplace=True)

# Add Provinces column
df_data.insert(
    loc=0,                        # index 0 -> first column
    column='Provinces',           # name of the new column
    value=df.iloc[1:5, 0].values  # the data to insert
)

# Optional: display all columns
pd.set_option('display.max_columns', None)
print("\nSummary of Crime Data by Category and Year")
print(df_data.shape)
print(df_data)


#--------------------------------------------------------------------------
# 2. Machine Learning: Clustering
#--------------------------------------------------------------------------


















