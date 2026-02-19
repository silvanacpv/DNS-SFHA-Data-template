#--------------------------------------------------------------------------
# Capstone Project:
# This project aims to identify patterns of criminal activity and group
# Atlantic Canadian provinces based on similarities in crime trends between
# 2020 and 2024.
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
ROWS_PROVINCES = 279     # Number of rows (provinces)
ROWS_HEADER = 3          # Number of rows of the header in csv file
NUM_YEARS = 5            # Years in csv file
THRESHOLD = 0.01         # 0.01% from the total of offences
LINE_SIZE = 70           # Line length in pixels to show sections
FIRST_YEAR = "2020"      # First year in csv file
LAST_YEAR = "2024"       # Last year in csv file

#--------------------------------------------------------------------------
# Mapping: full name → abbreviation
province_mapping = {
    "Nova Scotia": "NS",
    "New Brunswick": "NB",
    "Newfoundland and Labrador": "NL",
    "Prince Edward Island": "PE"
}

#--------------------------------------------------------------------------
# 0.2 Loading the file
#--------------------------------------------------------------------------

total_rows = ROWS_PROVINCES + ROWS_HEADER

df = pd.read_csv(
    "data_complete.csv",
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
df = df.drop(index=1)

# Drop uncompleted total columns (from 1 to 7)
df = df.drop(df.columns[1:8], axis=1)

# Replace null values
df = df.fillna(0)


# Rename column 0 to 'Provinces'
df.rename(columns={df.columns[0]: "Provinces"}, inplace=True)

# Drop rows that start with a province name
df = df[~df["Provinces"].apply(lambda x: any(str(x).strip().startswith(p) for p in province_mapping.keys()))].reset_index(drop=True)

# Replace province names with their abbreviations
for full_name, abbr in province_mapping.items():
    # Replace full_name with abbreviation
    # Keep only the abbreviation, discard everything after it
    df["Provinces"] = df["Provinces"].str.replace(rf"{full_name}.*", abbr, regex=True)

# Strip leading/trailing whitespace
df["Provinces"] = df["Provinces"].str.strip()

#pd.set_option('display.max_rows', None)
#print(df)

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
        year = str(int(year_raw))  # convert 2024.0 -> "2024"
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

# Print total number of offences
offences_df = offences_df.apply(pd.to_numeric, errors='coerce')
total = offences_df.sum().sum()
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
# 1.2.3 Sumarize by offence and by year
#--------------------------------------------------------------------------

# Keep only province data (remove header rows)
df_data = df.iloc[1:ROWS_PROVINCES+1, :].copy()  # includes all provinces
df_data = df_data.drop(df_data.columns[0], axis=1)  # drop province label column
df_data = df_data.apply(pd.to_numeric, errors='coerce')  # convert all to numeric

# Keep province names separately
provinces = df.iloc[1:ROWS_PROVINCES+1, 0].values

# --------------------------------------------------------------------------
# Summary by Offence
# --------------------------------------------------------------------------

# Keep province data
df_data_numeric = df_data.copy()

# Transpose to have offences as rows
df_t = df_data_numeric.T

# Extract offence names (remove '_Year' suffix)
df_t['offence'] = [col.rsplit('_', 1)[0] for col in df_t.index]

# Group by offence and sum across all years
df_by_offence = df_t.groupby('offence').sum().T  # transpose back to have provinces as rows

# Add Provinces column
df_by_offence.insert(0, 'Provinces', provinces)

print("\nSummary Table by Offence:")
print(df_by_offence.head())

# --------------------------------------------------------------------------
# Summary by Year
# --------------------------------------------------------------------------

# Transpose again to have years as rows
df_t = df_data_numeric.T

# Extract years from column names
df_t['year'] = [col.split('_')[-1] for col in df_t.index]

# Group by year and sum across all offences
df_by_year = df_t.groupby('year').sum().T  # transpose back

# Add Provinces column
df_by_year.insert(0, 'Provinces', provinces)

print("\nSummary Table by Year:")
print(df_by_year.head())


#--------------------------------------------------------------------------
# 2. Machine Learning: Clustering
#--------------------------------------------------------------------------
# 2.1 Prepare feature matrix 
#--------------------------------------------------------------------------

# Use numeric data for features (df_data_numeric)
X = df_data_numeric.copy()  # no 'Provinces' column here
X = X.fillna(0)

# Keep provinces separately
provinces = provinces  # already defined before

# Scale features (robust to outliers)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print("Feature matrix shape:", X_scaled.shape)
print("Number of provinces:", len(provinces))


# --------------------------------------------------------------------------
# 2.2 Select best KMeans by silhouette score
# --------------------------------------------------------------------------
def best_kmeans(X_data, n_range, random_state=42):
    
    best_score = -1
    best_n = None
    best_labels = None
    best_model = None
    scores = []
    
    for n in n_range:
        kmeans = KMeans(n_clusters=n, random_state=random_state)
        labels = kmeans.fit_predict(X_data)
        score = silhouette_score(X_data, labels)
        scores.append(score)
        
        if score > best_score:
            best_score = score
            best_n = n
            best_labels = labels
            best_model = kmeans

    # Plot Silhouette Score vs n_clusters
    plt.figure(figsize=(8,4))
    plt.plot(list(n_range), scores, marker='o', color='skyblue')
    plt.axvline(best_n, color='red', linestyle='--', label=f'Best n={best_n}')
    plt.title("Silhouette Score vs n_clusters")
    plt.xlabel("n_clusters")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return best_n, best_score, best_labels, best_model


# --------------------------------------------------------------------------
# 2.3 Graphics
# --------------------------------------------------------------------------

# Function to plot PCA scatter
def plot_pca(X_pca, labels, centroids_pca, city_names, representative_cities=[], title="PCA Plot"):
    plt.figure(figsize=(10,7))
    
    # Scatter points
    plt.scatter(X_pca[:,0], X_pca[:,1], 
                c=labels, cmap='tab10', s=15, alpha=0.5)
    
    # Centroids
    plt.scatter(centroids_pca[:,0], centroids_pca[:,1],
                marker='X', s=200, c='black', label='Centroids')
    
    # Label centroids
    for i, (x, y) in enumerate(centroids_pca):
        plt.text(x+0.02, y+0.02, f'Cluster {i}', fontsize=10, fontweight='bold')

    # Label representative cities
    for i, city in enumerate(city_names):
        if city in representative_cities:
            plt.text(X_pca[i,0]+0.01, X_pca[i,1]+0.01, city, fontsize=8)

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------------
# PCA for Absolute Levels Clustering
# --------------------------------------------------------------------------

# Step 1: Determine best K for KMeans
n_range = range(2, 10)  # adjust as needed
n_abs, score_abs, labels_abs, kmeans_abs = best_kmeans(X_scaled, n_range)

# Step 2: PCA transformation
pca_abs = PCA(n_components=2)
X_abs_pca = pca_abs.fit_transform(X_scaled)
centroids_abs_pca = pca_abs.transform(kmeans_abs.cluster_centers_)

# Step 3: Plot PCA
index_representative = [0, 4, 9, 12, 15]  
representative_provinces = provinces[index_representative]

plot_pca(
    X_abs_pca,
    labels_abs,
    centroids_abs_pca,
    provinces,  # use the 'provinces' array, not df_data["Provinces"]
    representative_cities=representative_provinces,
    title="PCA: Absolute Crime Levels"
)


# --------------------------------------------------------------------------
# PCA for Proportional Structure
# --------------------------------------------------------------------------

# 1. Convert counts to proportions per province
X_prop = df_data_numeric.div(df_data_numeric.sum(axis=1), axis=0).fillna(0)

# 2. Scale features (robust to outliers)
scaler = RobustScaler()
X_prop_scaled = scaler.fit_transform(X_prop)

# 3. Determine best K for KMeans using silhouette score
n_range = range(2, 10)  # adjust as needed
n_prop, score_prop, labels_prop, kmeans_prop = best_kmeans(X_prop_scaled, n_range)

# 4. Apply PCA to 2 components
pca_prop = PCA(n_components=2)
X_prop_pca = pca_prop.fit_transform(X_prop_scaled)
centroids_prop_pca = pca_prop.transform(kmeans_prop.cluster_centers_)

# 5. Plot PCA
plot_pca(
    X_prop_pca,
    labels_prop,
    centroids_prop_pca,
    provinces,  # array of province names
    representative_cities=representative_provinces,
    title="PCA: Proportional Structure"
)

# --------------------------------------------------------------------------
# PCA for Growth Patterns
# --------------------------------------------------------------------------

# 1. Calculate growth rates (year-over-year percentage change)
X_growth = df_data_numeric.copy()
growth_list = []
growth_names = []

# Unique offences (base names)
offence_base_names = sorted({col.rsplit('_',1)[0] for col in X_growth.columns})

for offence in offence_base_names:
    # Columns for this offence, sorted by year
    cols = sorted([c for c in X_growth.columns if c.startswith(offence)])
    
    for i in range(1, len(cols)):
        prev = X_growth[cols[i-1]]
        curr = X_growth[cols[i]]
        
        # Compute growth, avoid divide by zero
        growth = (curr - prev) / prev.replace(0, np.nan)
        growth = growth.fillna(0)
        
        growth_list.append(growth)
        growth_names.append(f"{offence}_growth_{cols[i]}")  # name by offence + current year

# Concatenate growth columns into DataFrame
X_growth_df = pd.concat(growth_list, axis=1)
X_growth_df.columns = growth_names

# 2. Scale features
scaler = RobustScaler()
X_growth_scaled = scaler.fit_transform(X_growth_df)

# 3. Determine best KMeans
n_range = range(2, 10)
n_growth, score_growth, labels_growth, kmeans_growth = best_kmeans(X_growth_scaled, n_range)

# 4. PCA transformation
pca_growth = PCA(n_components=2)
X_growth_pca = pca_growth.fit_transform(X_growth_scaled)
centroids_growth_pca = pca_growth.transform(kmeans_growth.cluster_centers_)

# 5. Plot PCA
plot_pca(
    X_growth_pca,
    labels_growth,
    centroids_growth_pca,
    provinces,  # array of province names
    representative_cities=representative_provinces,
    title="PCA: Crime Growth Patterns"
)


# --------------------------------------------------------------------------
# 2.4 Summary of Silhouette Scores
# --------------------------------------------------------------------------
print("\nSilhouette Scores Summary:")
print(f"Absolute Levels: {round(score_abs,3)} (n_clusters={n_abs})")
print(f"Proportional Structure: {round(score_prop,3)} (n_clusters={n_prop})")
print(f"Growth Patterns: {round(score_growth,3)} (n_clusters={n_growth})")










