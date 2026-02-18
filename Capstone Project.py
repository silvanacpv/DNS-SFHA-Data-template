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
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


#--------------------------------------------------------------------------
# 0. Settings
#--------------------------------------------------------------------------
# 0.1 Constants 
#--------------------------------------------------------------------------
METADATA_ROWS = 8        # Number of metadata rows at the top of the file
ROWS_PROVINCES = 4       # Number of rows (provinces)
ROWS_HEADER = 3          # Number of rows of the header in csv file
NUM_YEARS = 7            # Years from 2018 to 2024
THRESHOLD = 0.01         # 0.01% from the total of offences
LINE_SIZE = 70           # Line length in pixels
FIRST_YEAR = "2018"      # First year in csv file
LAST_YEAR = "2024"       # Last year in csv file

#--------------------------------------------------------------------------
# 0.2 Loading the file
#--------------------------------------------------------------------------

total_rows = ROWS_PROVINCES + ROWS_HEADER

df = pd.read_csv(
    "data.csv",
    sep=',',                # CSV is comma-separated
    skiprows=METADATA_ROWS, # skip the header
    nrows=total_rows,       # read the table
    quotechar='"',          # handle commas inside quoted text
    encoding='utf-8-sig'
)

# Print the raw dataset
pd.set_option('display.max_columns', None)
print("=" * LINE_SIZE)
print("CAPSTONE PROJECT")
print("1. Data Processing")
print("=" * LINE_SIZE)
print("Criminal Code violations", FIRST_YEAR, "–", LAST_YEAR, "by Province and Territory")
print("Raw Dataset:")
print(df.iloc[:5, :7])
print("-" * LINE_SIZE)


#--------------------------------------------------------------------------
# 1. Data Processing
#--------------------------------------------------------------------------
# 1.1 Data cleansing
#--------------------------------------------------------------------------
# 1.1.1 Drop rows and columns not required
#--------------------------------------------------------------------------
# Remove irrelevant row on the header
df = df.drop(index=0)

# Remove fully empty row
df = df.drop(index=2)

# Drop uncompleted total columns (from 1 to 7)
df = df.drop(df.columns[1:8], axis=1)


#--------------------------------------------------------------------------
# 1.1.2 Create column names in the format: offence_year.  
# Append the year to offence names that are present in the first year      
# For years where the offence name is blank, fill in the offence name
# and append the corresponding year.  
#--------------------------------------------------------------------------

years = df.iloc[0]             # row with years
original_columns = df.columns  # current name of columns

new_columns = []
current_crime = None

for i in range(len(original_columns)):
    col_name = original_columns[i]

    # Keep the first column unchanged
    if i == 0:
        new_columns.append(col_name)
        continue
    
    year_raw = years.iloc[i]

    # Convert year to string
    if pd.api.types.is_number(year_raw):
        year = str(int(year_raw))  # convert 2018.0 -> "2018"
    else:
        year = str(year_raw).strip()  # keep as string (e.g., "Unnamed")

    # Update current crime if column name is not "Unnamed"
    if not str(col_name).startswith("Unnamed"):
        current_crime = col_name.strip()

    # Combine crime name with year
    new_columns.append(f"{current_crime}_{year}")

df.columns = new_columns

#--------------------------------------------------------------------------
# 1.2 Data manipulation
#--------------------------------------------------------------------------
# 1.2.1 Summarize offences across all years to identify somes with
# non-representative values which will be removed in a later step
# (i.e., offences with less than 0.01% of the total)
#--------------------------------------------------------------------------

# Summarize offences across all years
rows_to_sum = slice(1, df.shape[0])  # all rows below header with actual data

summary_data = {}

col_idx = 0
while col_idx < df.shape[1]:
    
    # Check if this column starts the FIRST_YEAR block
    if str(df.iloc[0, col_idx]).strip() == FIRST_YEAR:
        # Ensure full block exists
        if col_idx + NUM_YEARS - 1 >= df.shape[1]:
            break
        
        block_idx = list(range(col_idx, col_idx + NUM_YEARS))
        
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
        col_idx += NUM_YEARS
    else:
        col_idx += 1

# Create summary DataFrame: 1 value per offence
offences_df = pd.DataFrame(summary_data)

# Remove the last 5 characters from offence names
offences_df.columns = [
    re.sub(r"_\d{4}$", "", col).rstrip()
    for col in offences_df.columns
]

# Delete the "Total" columns that came from the original file
offences_df = offences_df.iloc[:, 3:] 

# Print total number of offences
offences_df = offences_df.apply(pd.to_numeric, errors='coerce')
total = offences_df.sum(axis=1).iloc[0]
print(f"\nTotal offences: {total:,}")
print("Total offence types:", offences_df.shape[1])

#--------------------------------------------------------------------------
# 1.2.2 Delete offences with non-representative values
#--------------------------------------------------------------------------

# Get offences to exclude
# Compute absolute minimum frequency threshold to avoid rare-event distortion 
# % from the total number of offences
min_todelete = total * THRESHOLD / 100   

crimes_todelete = offences_df.columns[offences_df.iloc[0] < min_todelete].tolist()
crimes_todelete = [str(c).strip().lower() for c in crimes_todelete]
    
# Get number of offence types with non-representative values
num_todelete = (offences_df < min_todelete).sum().sum()
print("Offence types with fewer than ", THRESHOLD, "% incidents: ", num_todelete)


# Get valid offences
offences_df.drop(
    columns=[
        c
        for c in offences_df.columns
        if c.lower() in crimes_todelete
    ],
    inplace=True
)
print("Valid offences after filtering:", offences_df.shape[1])
    

# Delete offences from the main dataset
# Identify columns to drop in df that start with the base name
cols_to_drop = [
    c
    for c in df.columns
    if any(
        c.lower().startswith(zero)
        for zero in crimes_todelete
    )
]

# Drop these columns from df
df.drop(cols_to_drop, axis=1, inplace=True)

print("Shape after deleting offences:",df.shape)
print("-" * LINE_SIZE)

#--------------------------------------------------------------------------
# 1.2.3 Create offence categorization in the main dataset
#--------------------------------------------------------------------------
# 1.2.3.1 Load categorization files
#--------------------------------------------------------------------------

# Read the offence categories and category texts CSV with category assignments
categorization_df = pd.read_csv("categorization.csv", header=None, dtype=str)
categorization_df.columns = ["offence_name", "category_id", "category_text"]  

category_texts_df = pd.read_csv("categories.csv", header=None, dtype=str)
category_texts_df.columns = ["category_id", "category_text"] 

# Create a mapping from offence_name -> category
offence_to_category = {
    name.strip(): cat
    for name, cat in zip(
        categorization_df["offence_name"],
        categorization_df["category_id"]
    )
}

# Map category IDs to offences_df
category_ids = []
for col in offences_df:
    cat = offence_to_category.get(col.strip(), "Unknown")  # default if not found
    category_ids.append(cat)

offences_df.loc["category_id"] = category_ids

# Map category text
cat_text_dict = dict(zip(category_texts_df["category_id"], category_texts_df["category_text"]))
category_texts = [
    cat_text_dict.get(cat, "Unknown")
    if cat != "Unknown"
    else "Unknown"
    for cat in category_ids
]

offences_df.loc["category_text"] = category_texts
num_categories = category_texts_df["category_id"].nunique()

#--------------------------------------------------------------------------
# 1.2.3.2 Set offence categories in the main dataset
#--------------------------------------------------------------------------
# Sumarize all offences by category and year
#--------------------------------------------------------------------------

# Create category row in the main dataset
df.loc['category_id'] = [
    offence_to_category.get(col[:-5].strip(), 'Unknown')
    for col in df.columns
]

# Keep only province data (remove first and last rows)
df_data = df.iloc[1:-1].copy()
df_data = df_data.drop(df_data.columns[0], axis=1)  # drop province label column
df_data = df_data.apply(pd.to_numeric, errors='coerce')

# Drop columns starting with 'nan'
cols_to_drop = [c for c in df_data.columns if str(c).startswith('nan')]
df_data.drop(cols_to_drop, axis=1, inplace=True)

# Add Provinces column
df_data.insert(0, 'Provinces', df.iloc[1:ROWS_PROVINCES+1, 0].values)

# --------------------------------------------------------------------------
# Summary by Category
# --------------------------------------------------------------------------

df_numeric = df_data.drop(columns=['Provinces'])

# Transpose to group columns
df_t = df_numeric.T
categories = [c.split('_')[0] for c in df_t.index]
df_t['category'] = categories

# Group by category and sum
df_by_category = df_t.groupby('category').sum().T
df_by_category.insert(0, 'Provinces', df_data['Provinces'])

print("\nSummary Table by Category:")
print(df_by_category)

# --------------------------------------------------------------------------
# Summary by Year
# --------------------------------------------------------------------------

# Transpose again
df_t = df_numeric.T
years = [c.split('_')[1] for c in df_t.index]
df_t['year'] = years

# Group by year and sum
df_by_year = df_t.groupby('year').sum().T
df_by_year.insert(0, 'Provinces', df_data['Provinces'])

print("\nSummary Table by Year:")
print(df_by_year)


#--------------------------------------------------------------------------
# 2. Machine Learning: Clustering
#--------------------------------------------------------------------------
# 2.1 Prepare feature matrix 
#--------------------------------------------------------------------------

# Remove Provinces column
X = df_data.drop(columns="Provinces").T.copy()

# Scale features (robust to outliers)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Keep provinces separately
provinces = df_data["Provinces"].values  

print("=" * LINE_SIZE)
print("\n2. Machine Learning")
print("=" * LINE_SIZE)

# --------------------------------------------------------------------------
# 2.2 Select best KMeans by silhouette score
# --------------------------------------------------------------------------
def best_kmeans(X_data, n_range, random_state=42):
    best_score = -1
    best_n = None
    best_labels = None
    best_model = None
    for n in n_range:
        kmeans = KMeans(n_clusters=n, random_state=random_state)
        labels = kmeans.fit_predict(X_data)
        score = silhouette_score(X_data, labels)
        print(f"n_clusters={n} → Silhouette Score: {round(score,3)}")
        if score > best_score:
            best_score = score
            best_n = n
            best_labels = labels
            best_model = kmeans
    return best_n, best_score, best_labels, best_model

# --------------------------------------------------------------------------
# 2.3 Clustering by Category
# --------------------------------------------------------------------------
print("\nClustering: Provinces by Category")
print("-" * LINE_SIZE)

X_cat = df_by_category.drop(columns=['Provinces']).values
provinces = df_by_category['Provinces'].values

scaler = RobustScaler()
X_cat_scaled = scaler.fit_transform(X_cat)

# Range: 2 to (n_provinces-1) clusters
range_clusters_cat = range(2, len(X_cat_scaled))
n_cat, score_cat, labels_cat, kmeans_cat = best_kmeans(X_cat_scaled, range_clusters_cat)

df_by_category['cluster'] = labels_cat
print(f"\nBest n_clusters: {n_cat}, Silhouette Score: {round(score_cat,3)}")
print(df_by_category[['Provinces','cluster']])

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cat_scaled)
centroids_pca = pca.transform(kmeans_cat.cluster_centers_)

plt.figure(figsize=(7,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_cat, s=100, cmap='tab10')
plt.scatter(centroids_pca[:,0], centroids_pca[:,1], marker='X', s=200, c='black')
for i, prov in enumerate(provinces):
    plt.text(X_pca[i,0]+0.02, X_pca[i,1]+0.02, prov)
plt.title("Clusters of Provinces by Category")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# --------------------------------------------------------------------------
# 2.4 Clustering by Year
# --------------------------------------------------------------------------
print("\nClustering: Provinces by Year")
print("-" * LINE_SIZE)

X_year = df_by_year.drop(columns=['Provinces']).values
provinces = df_by_year['Provinces'].values

X_year_scaled = scaler.fit_transform(X_year)

# Range: 2 to (n_provinces-1) clusters
range_clusters_year = range(2, len(X_year_scaled))
n_year, score_year, labels_year, kmeans_year = best_kmeans(X_year_scaled, range_clusters_year)

df_by_year['cluster'] = labels_year
print(f"\nBest n_clusters: {n_year}, Silhouette Score: {round(score_year,3)}")
print(df_by_year[['Provinces','cluster']])

# PCA for visualization
pca_year = PCA(n_components=2)
X_year_pca = pca_year.fit_transform(X_year_scaled)
centroids_year_pca = pca_year.transform(kmeans_year.cluster_centers_)

plt.figure(figsize=(7,6))
plt.scatter(X_year_pca[:,0], X_year_pca[:,1], c=labels_year, s=100, cmap='tab10')
plt.scatter(centroids_year_pca[:,0], centroids_year_pca[:,1], marker='X', s=200, c='black')
for i, prov in enumerate(provinces):
    plt.text(X_year_pca[i,0]+0.02, X_year_pca[i,1]+0.02, prov)
plt.title("Clusters of Provinces by Year")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# --------------------------------------------------------------------------
# 2.5 Summary of Silhouette Scores
# --------------------------------------------------------------------------
print("\nSilhouette Scores Summary:")
print(f"By Category: {round(score_cat,3)} (n_clusters={n_cat})")
print(f"By Year: {round(score_year,3)} (n_clusters={n_year})")








