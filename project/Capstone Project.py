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
# 1. Data Processing: data manipulation 
#--------------------------------------------------------------------------
# A. Drop empty rows
#--------------------------------------------------------------------------
df = df.drop(index=0)
df = df.drop(index=2)

# Inspect the resulting DataFrame
print("1. Data Processing")
print(df.head(6))

#--------------------------------------------------------------------------
# B. Create number of columns in format: offence_year
# to fill the header, because in the first row is the name of the offence
# (only over the first year, but over the following years are blanks.
# and in the second row are shown the years
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
# C. Create a new dataset summarizing years by offence with the objective to
# identify offences with no values or non-representative values to delete
# them later
#--------------------------------------------------------------------------
# Row 0 has the years
# Columns are the crime names
rows_to_sum = slice(1, df.shape[0])  # all rows below header with actual data
num_years = 7  # 2018-2024

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

# Create summary DataFrame: 1 row, 1 column per crime
offences_df = pd.DataFrame(summary_data)
offences_df = offences_df.iloc[:, 4:] #delete totals
print("\nTOTALS")
print("Total of offence types:", offences_df.shape[1])

# Print total number of offences
offences_df = offences_df.apply(pd.to_numeric, errors='coerce')
total = offences_df.sum(axis=1).iloc[0]
print("Total number of offences:", total)


#--------------------------------------------------------------------------
# D. Delete offences with non-representative values from the original dataset
#--------------------------------------------------------------------------
# 1. Get offences to exclude: with non-representative values
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
print("Number of offence types with fewer than ", min_todelete, "incidents", threshold, "%:", num_todelete)


#--------------------------------------------------------------------------
# 2. Get valid offences
#--------------------------------------------------------------------------
offences_df.drop(columns=[c for c in offences_df.columns if c.lower() in crimes_todelete], inplace=True)
print("\nValid offences:", offences_df)

#Print the valid offences to create the Categorization.csv file
#for col in offences_df.columns:
#    print(col)
    

#--------------------------------------------------------------------------
# 3. Delete offences from the main dataset
#--------------------------------------------------------------------------
# Extract base name (remove last 5 characters, e.g., '_2018')
crimes_todelete_base = [c[:-5] for c in crimes_todelete]

# Identify columns to drop in df that start with the base name
cols_to_drop = [c for c in df.columns if any(c.lower().startswith(zero) for zero in crimes_todelete_base)]

# Drop these columns from df
df.drop(cols_to_drop, axis=1, inplace=True)

print("Number of rows after deleting offences:",df.shape)


#--------------------------------------------------------------------------
# E. Create offence categorization
#--------------------------------------------------------------------------

# 1. Remove the last 5 characters from offence names in offences_df for matching
offences_df.columns = [
    re.sub(r"_\d{4}$", "", col).rstrip()
    for col in offences_df.columns
]

# 2. Read the offences CSV with category assignments
categorization_df = pd.read_csv("categorization.csv", header=None, dtype=str)
categorization_df.columns = ["offence_name", "category_id", "category_text"]  # adjust column names

# 3. Read the CSV with category texts
category_texts_df = pd.read_csv("categories.csv", header=None, dtype=str)
category_texts_df.columns = ["category_id", "category_text"]  # adjust if needed

# 4. Create a mapping from offence_name -> category
offence_to_category = {name.strip(): cat for name, cat in zip(categorization_df["offence_name"], categorization_df["category_id"])}

# 4. Map category IDs to offences_df
category_ids = []
for col in offences_df:
    cat = offence_to_category.get(col.strip(), "Unknown")  # default if not found
    category_ids.append(cat)

offences_df.loc["category_id"] = category_ids

# 5. Map category text
cat_text_dict = dict(zip(category_texts_df["category_id"], category_texts_df["category_text"]))
category_texts = [cat_text_dict.get(cat, "Unknown") if cat != "Unknown" else "Unknown" for cat in category_ids]

offences_df.loc["category_text"] = category_texts

#--------------------------------------------------------------------------
# F. Set offence categories in the main dataset
#--------------------------------------------------------------------------

# Add a new row to df with the category for each offence column
# Strip the last 5 characters (_year) and any extra spaces to match the base name
df.loc['category_id'] = [
    offence_to_category.get(col[:-5].strip(), 'Unknown')
    for col in df.columns
]

#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', 2000)
#print(df.head(10))


#--------------------------------------------------------------------------
# 2. Machine Learning: Clustering
#--------------------------------------------------------------------------



















