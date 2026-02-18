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
TABLE_ROWS = 7           # Number of rows corresponding to the offences table
NUM_YEARS = 7            # Years from 2018 to 2024
THRESHOLD = 0.01         # 0.01% from the total of offences
LINE_SIZE = 70           # Line length in pixels

# Clustering parameters
# Number of clusters when grouping provinces based on absolute counts or proportional structure
N_CLUSTERS_PROVINCES = 2

# Number of clusters when grouping Category_Year columns to analyze patterns across offences and years
N_CLUSTERS_CATEGORY_YEAR = 3

# Seed for the random number generator to ensure reproducible clustering results
RANDOM_STATE = 42

#--------------------------------------------------------------------------
# 0.2 Loading the file
#--------------------------------------------------------------------------

df = pd.read_csv(
    "data.csv",
    sep=',',                # CSV is comma-separated
    skiprows=METADATA_ROWS, # skip the header
    nrows=TABLE_ROWS,       # read the table
    quotechar='"',          # handle commas inside quoted text
    encoding='utf-8-sig'
)

# Print the raw dataset
pd.set_option('display.max_columns', None)
print("=" * LINE_SIZE)
print("CAPSTONE PROJECT")
print("1. Data Processing")
print("=" * LINE_SIZE)
print("Criminal Code violations (2018–2024) by Province and Territory")
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
# # Remove irrelevant row on the header
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
    
    # Check if this column starts a 2018 block
    if str(df.iloc[0, col_idx]).strip() == "2018":
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
print("\nSubset of the Clean Dataset:")
print(df.iloc[:, 22:32])
print("-" * LINE_SIZE)

#--------------------------------------------------------------------------
# 1.2.3 Create offence categorization
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
# Problem: we need to sum all offences that belong to the same
# category within the same year. Pandas does not allow horizontal
# groupby directly, so we transpose the dataframe, group by the
# combined column names (Category_Year), sum the values, and transpose back.
# Result: df_data has one column per Category_Year, aggregated across offences.
#--------------------------------------------------------------------------

# Create category row in the main dataset
# Add a new row to df with the category_id for each offence column
# Strip the last characters (_year) from df to match the base name
df.loc['category_id'] = [
    offence_to_category.get(col[:-5].strip(), 'Unknown')
    for col in df.columns
]

# Create a new dataset summarized by category and year 
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

# Keep only province data: remove first (year) and last (category) rows
df_data = df.iloc[1:-1]

# Remove the first column containing province labels
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

# Display the new dataset: Summary of Crime Data by Category and Year 
print("\nSummary of Crime Data by Category and Year:")
print(num_categories, "categories ×", NUM_YEARS, "years (plus descriptive text)")
print("Shape of summary matrix:", df_data.shape)
print(df_data.iloc[:5, :10])
print(df_data)


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
print("\nMachine Learning")
print("=" * LINE_SIZE)


#--------------------------------------------------------------------------
# Helper function to select best KMeans by silhouette score
#--------------------------------------------------------------------------
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

#--------------------------------------------------------------------------
# 2.2.1 Absolute Crime Levels
#--------------------------------------------------------------------------
print("\n--- Clustering: Absolute Crime Levels ---")
range_clusters_abs = range(2, min(len(X_scaled), 10)+1)
n_abs, score_abs, labels_abs, kmeans_abs = best_kmeans(X_scaled, range_clusters_abs)

df_features_abs = pd.DataFrame(X_scaled, columns=provinces)
df_features_abs['cluster_abs'] = labels_abs
print(f"Best n_clusters: {n_abs}, Silhouette Score: {round(score_abs,3)}")
print(df_features_abs[['cluster_abs']])

#--------------------------------------------------------------------------
# 2.2.2 Proportional Crime Structure
#--------------------------------------------------------------------------
print("\n--- Clustering: Proportional Crime Structure ---")

X_prop = df_data.drop(columns="Provinces").copy()

years = sorted({col.split("_")[-1] for col in X_prop.columns})

for year in years:
    year_cols = [c for c in X_prop.columns if c.endswith(year)]
    X_prop[year_cols] = X_prop[year_cols].div(X_prop[year_cols].sum(axis=1), axis=0)

X_prop_scaled = scaler.fit_transform(X_prop)

# Limitar n_clusters a filas - 1
range_clusters_prop = range(2, len(X_prop_scaled))
n_prop, score_prop, labels_prop, _ = best_kmeans(X_prop_scaled, range_clusters_prop)

df_data["cluster_prop"] = labels_prop
print(df_data[["Provinces", "cluster_prop"]])
print(f"Best n_clusters: {n_prop}, Silhouette Score: {round(score_prop,3)}")

#--------------------------------------------------------------------------
# 2.2.3 Crime Growth Patterns
#--------------------------------------------------------------------------
print("\n--- Clustering: Crime Growth Patterns ---")

categories = sorted({col.rsplit("_",1)[0] for col in X_prop.columns})
growth_df = pd.DataFrame(index=df_data.index)

for cat in categories:
    cat_cols = sorted([c for c in X_prop.columns if c.startswith(cat)])
    first_year = cat_cols[0]
    last_year = cat_cols[-1]
    growth_df[cat] = X_prop[last_year] - X_prop[first_year]

growth_scaled = scaler.fit_transform(growth_df)

# Limitar n_clusters a filas - 1
range_clusters_growth = range(2, len(growth_scaled))
n_growth, score_growth, labels_growth, _ = best_kmeans(growth_scaled, range_clusters_growth)

df_data["cluster_growth"] = labels_growth
print(df_data[["Provinces", "cluster_growth"]])
print(f"Best n_clusters: {n_growth}, Silhouette Score: {round(score_growth,3)}")



#--------------------------------------------------------------------------  
# 2.3 PCA Visualization 
#--------------------------------------------------------------------------  
# PCA Absolute Levels  
year = "2022"
year_cols = [c for c in df_data.columns if c.endswith(year)]
X_year = df_data[year_cols].T.copy()  # rows = features for this year
X_year_scaled = scaler.fit_transform(X_year)

# Limit number of clusters to max rows - 1 to avoid silhouette errors
range_clusters_year = range(2, len(X_year_scaled))
n_year, score_year, labels_year, kmeans_year = best_kmeans(X_year_scaled, range_clusters_year)

# PCA transformation
pca = PCA(n_components=2)
X_year_pca = pca.fit_transform(X_year_scaled)
centroids_year = pca.transform(kmeans_year.cluster_centers_)

# Scatter plot with centroids and feature labels
plt.figure(figsize=(7,6))
plt.scatter(X_year_pca[:,0], X_year_pca[:,1], c=labels_year, s=100, label="Features")
plt.scatter(centroids_year[:,0], centroids_year[:,1], marker='X', s=200, c='black', label="Centroids")
for i, feature in enumerate(year_cols):
    plt.text(X_year_pca[i,0]+0.05, X_year_pca[i,1]+0.05, feature)
plt.title(f"K-Means Clustering (Absolute Levels) - {year}")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()


# PCA Proportional
pca_prop = PCA(n_components=2)
X_prop_pca = pca_prop.fit_transform(X_prop_scaled)

plt.figure(figsize=(7,6))
plt.scatter(X_prop_pca[:,0], X_prop_pca[:,1], c=labels_prop, s=100, cmap='tab10')
for i, prov in enumerate(df_data["Provinces"]):
    plt.text(X_prop_pca[i,0]+0.02, X_prop_pca[i,1]+0.02, prov)
plt.title("PCA: Proportional Crime Structure")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# PCA Growth
pca_growth = PCA(n_components=2)
X_growth_pca = pca_growth.fit_transform(growth_scaled)

plt.figure(figsize=(7,6))
plt.scatter(X_growth_pca[:,0], X_growth_pca[:,1], c=labels_growth, s=100, cmap='tab10')
for i, prov in enumerate(df_data["Provinces"]):
    plt.text(X_growth_pca[i,0]+0.02, X_growth_pca[i,1]+0.02, prov)
plt.title("PCA: Crime Growth Patterns")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


#--------------------------------------------------------------------------
# 2.4 Silhouette Scores Summary
#--------------------------------------------------------------------------
print("\nSilhouette Scores Summary:")
print(f"Absolute Levels: {round(score_abs,3)} (n_clusters={n_abs})")
print(f"Proportional Structure: {round(score_prop,3)} (n_clusters={n_prop})")
print(f"Growth Patterns: {round(score_growth,3)} (n_clusters={n_growth})")









