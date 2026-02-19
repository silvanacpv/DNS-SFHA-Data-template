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
THRESHOLD = 0.1          # % from the total of offences
LINE_SIZE = 70           # Line length in pixels to show sections
FIRST_YEAR = "2020"      # First year in csv file
LAST_YEAR = "2024"       # Last year in csv file

#--------------------------------------------------------------------------
# 0.2 Variables 
#--------------------------------------------------------------------------
# Number of years in the file
num_years = int(LAST_YEAR) - int(FIRST_YEAR) + 1

# Mapping: full name → abbreviation
province_mapping = {
    "Nova Scotia": "NS",
    "New Brunswick": "NB",
    "Newfoundland and Labrador": "NL",
    "Prince Edward Island": "PE"
}

# List of patterns to remove
remove_patterns = [
    "Canadian National Railway Police",
    "Codiac Regional"
]

#--------------------------------------------------------------------------
# 0.3 Loading the file
#--------------------------------------------------------------------------

df = pd.read_csv(
    "data_complete.csv",
    sep=',',                # CSV is comma-separated
    skiprows=METADATA_ROWS, # skip the header
    quotechar='"',          # handle commas inside quoted text
    encoding='utf-8-sig'
)

# Print the raw dataset
pd.set_option('display.max_columns', None)
print("=" * LINE_SIZE)
print("CAPSTONE PROJECT")
print("=" * LINE_SIZE)
print("Criminal Code violations", FIRST_YEAR, "–", LAST_YEAR, "by Province and Territory")
print("-" * LINE_SIZE)
print("1. Data Processing")
print("-" * LINE_SIZE)

#--------------------------------------------------------------------------
# 1. Data Processing
#--------------------------------------------------------------------------
# 1.1 Data cleansing
#--------------------------------------------------------------------------
# 1.1.1 Drop not needed rows and columns 
#--------------------------------------------------------------------------
# Remove not needed row on the header
df = df.drop(index=2).reset_index(drop=True)

# Drop total columns (from 1 to 7) with totals
df = df.drop(df.columns[1:8], axis=1)

# Replace null values
df = df.fillna(0)

#--------------------------------------------------------------------------
# 1.1.2 Drop rows containing provincial totals (redundant data)
#--------------------------------------------------------------------------
# Rename column 0 to 'Provinces'
df.rename(columns={df.columns[0]: "Provinces"}, inplace=True)

# Drop rows that start with a province name
mask = df["Provinces"].astype(str).str.strip().str.startswith(tuple(province_mapping.keys()))
df = df[~mask]
df.reset_index(drop=True, inplace=True)

#--------------------------------------------------------------------------
# 1.1.3 Drop rows with values that do not correspond to cities
#--------------------------------------------------------------------------
# Drop rows where the first column starts with defined values in remove_patterns
df = df[~df['Provinces'].astype(str).str.strip().str.startswith(tuple(remove_patterns))].reset_index(drop=True)
df = df[~df['Provinces'].str.contains('offshore', case=False, na=False)]

df.reset_index(drop=True, inplace=True)

#--------------------------------------------------------------------------
# 1.1.4 Drop footer
#--------------------------------------------------------------------------
# Drop the footer when find the text "Symbol legend"
first_col = df.columns[0]

# Convert to numeric where possible (non-numeric -> NaN)
col_numeric = pd.to_numeric(df[first_col], errors='coerce')

# Detect numeric zeros
numeric_zero_positions = col_numeric[col_numeric == 0].index

# Detect string patterns in original column
col_str = df[first_col].astype(str).str.strip()
string_patterns = ["", "Symbol legend:", ".."]
string_zero_positions = col_str[col_str.isin(string_patterns)].index

# Combine all potential end-of-table positions
end_positions = sorted(list(set(numeric_zero_positions) | set(string_zero_positions)))

# Cut table at first end-of-table position (if any)
if len(end_positions) > 0:
    first_end_pos = end_positions[0]
    if first_end_pos > 0:  # keep at least one row
        df = df.iloc[:first_end_pos].copy()

df.reset_index(drop=True, inplace=True)
        
#--------------------------------------------------------------------------
# 1.1.5 Drop rows with no data
#--------------------------------------------------------------------------
# Select only data columns (ignore the first column 'Provinces')
data_cols = df.columns[1:]

# Convert data columns to numeric (invalid parsing becomes NaN)
df.iloc[1:, 1:] = df.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce')

# Keep row 0 intact and remove rows where all data columns are NaN or 0
df = pd.concat([
    df.iloc[[0]],  # keep first row
    df.iloc[1:][
        ~(
            df.iloc[1:][data_cols].isna().all(axis=1) |
            (df.iloc[1:][data_cols] == 0).all(axis=1)
        )
    ]
]).reset_index(drop=True)

#--------------------------------------------------------------------------
# 1.1.6 Get number of rows
#--------------------------------------------------------------------------
row_provinces = df.shape[0]


#--------------------------------------------------------------------------
# 1.2 Data manipulation
#--------------------------------------------------------------------------
# 1.2.1 Replace province names with their abbreviations
#--------------------------------------------------------------------------
# Replace full_name with abbreviation, discard everything after it
for full_name, abbr in province_mapping.items():
    df["Provinces"] = df["Provinces"].str.replace(rf"{full_name}.*", abbr, regex=True)

# Strip leading/trailing whitespace
df["Provinces"] = df["Provinces"].str.strip()


#--------------------------------------------------------------------------
# 1.2.2 Create column names in the format: offence_year 
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
        #df.iloc[0] = df.iloc[0].apply(lambda x: str(int(x)) if pd.notna(x) else "")
    else:
        year = str(year_raw).strip()  # keep as string (e.g., "Unnamed")

    # Update current crime if column name is not "Unnamed"
    if not str(col_name).startswith("Unnamed"):
        current_crime = col_name.strip()

    # Combine crime name with year
    new_columns.append(f"{current_crime}_{year}")

df.columns = new_columns

#--------------------------------------------------------------------------
# 1.2.3 Summarize offences across all years to identify somes with
# non-representative values which will be removed in a later step
#--------------------------------------------------------------------------

# Summarize offences across all years
rows_to_sum = slice(1, df.shape[0])  # all rows below header with actual data
summary_data = {}

col_idx = 0
while col_idx < df.shape[1]:
    
    # Check if this column starts the FIRST_YEAR block
    if str(df.iloc[0, col_idx]).strip() == FIRST_YEAR:
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

# Print total number of offences
offences_df = offences_df.apply(pd.to_numeric, errors='coerce')
total = offences_df.sum().sum()
print(f"\nTotal offences: {total:,}")
print("Total offence types:", offences_df.shape[1])

#--------------------------------------------------------------------------
# 1.2.4 Delete offences with non-representative values
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

#--------------------------------------------------------------------------
# 1.2.5 Sumarize by offence and by year
#--------------------------------------------------------------------------
# Keep only province data (remove header rows)
df_data = df.iloc[1:row_provinces, :].copy()  # includes all provinces
df_data = df_data.drop(df_data.columns[0], axis=1)  # drop province label column
df_data = df_data.apply(pd.to_numeric, errors='coerce')  # convert all to numeric

# Keep province names separately
provinces = df.iloc[1:row_provinces, 0].values

# --------------------------------------------------------------------------
# 1.2.5.1 Summary by Offence
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
print(df_by_offence)
print(df_by_offence.shape)

# --------------------------------------------------------------------------
# 1.2.5.2 Summary by Year
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
print(df_by_year)
print(df_by_year.shape)


#--------------------------------------------------------------------------
# 2. Machine Learning: Clustering and PCA
#--------------------------------------------------------------------------

# Ensure provinces is a NumPy array
provinces = np.array(provinces)

#--------------------------------------------------------------------------
# 2.1 Generic function: find best KMeans using silhouette score
#--------------------------------------------------------------------------
def best_kmeans(X_data, n_range=range(2,10), random_state=42):
    """
    Fit KMeans for a range of cluster numbers, select the best using silhouette score,
    and plot silhouette vs n_clusters.
    """
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

#--------------------------------------------------------------------------
# 2.2 Generic function: PCA scatter plot
#--------------------------------------------------------------------------
def plot_pca(X_pca, labels, centroids_pca, province_names, 
             title="PCA Plot", label_percentile=85):
    """
    Scatter plot of PCA components colored by cluster labels.
    Only labels provinces whose distance from centroid is above a percentile.
    """

    plt.figure(figsize=(16,12))

    # Scatter points
    plt.scatter(X_pca[:,0], X_pca[:,1],
                c=labels, cmap='Set1', s=50, alpha=0.6)

    # Centroids
    plt.scatter(centroids_pca[:,0], centroids_pca[:,1],
                marker='X', s=120, c='black', label='Centroids')

    # Label centroids
    for i, (x, y) in enumerate(centroids_pca):
        plt.text(x+0.05, y+0.06, str(i), 
                 fontsize=8, fontweight='bold', ha='left', va='bottom')

    # ---- NEW PART (minimal addition) ----
    # Compute center of PCA space
    center = X_pca.mean(axis=0)

    # Distance of each province to center
    distances = np.linalg.norm(X_pca - center, axis=1)

    # Threshold by percentile
    threshold = np.percentile(distances, label_percentile)

    # Label only far points
    for i, prov in enumerate(province_names):
        if distances[i] > threshold:
            plt.text(X_pca[i,0]+0.02, 
                     X_pca[i,1]+0.02, 
                     prov, fontsize=7,
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    # -------------------------------------

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.show()

#--------------------------------------------------------------------------
# 2.3 Generic function: scale, cluster, PCA, plot
#--------------------------------------------------------------------------
def cluster_and_plot(X_numeric, province_names, title_prefix="", label_percentile=60):
    """
    Perform robust scaling, KMeans clustering, PCA, and PCA scatter plotting.
    Returns silhouette score, best n_clusters, labels, KMeans model.
    """
    # Fill missing values
    X_numeric = X_numeric.fillna(0)

    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    # Find best KMeans
    n_range = range(2,10)
    best_n, best_score, labels, kmeans_model = best_kmeans(X_scaled, n_range)

    # PCA transformation
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(kmeans_model.cluster_centers_)

    # Plot PCA
    plot_pca(
        X_pca,
        labels,
        centroids_pca,
        province_names,
        title=f"{title_prefix} PCA",
        label_percentile=label_percentile
    )

    return best_score, best_n, labels, kmeans_model

#--------------------------------------------------------------------------
# 2.4 Absolute Levels Clustering
#--------------------------------------------------------------------------
# Select representative provinces automatically (top 5 total offences)
total_offences_per_province = df_data_numeric.sum(axis=1)
top_indices = total_offences_per_province.sort_values(ascending=False).index[:5]
#representative_provinces = provinces[top_indices]

score_abs, n_abs, labels_abs, kmeans_abs = cluster_and_plot(
    df_data_numeric,
    provinces,
    title_prefix="Absolute Crime Levels",
    label_percentile=90
)

#--------------------------------------------------------------------------
# 2.5 Proportional Structure Clustering
#--------------------------------------------------------------------------
X_prop = df_data_numeric.div(df_data_numeric.sum(axis=1), axis=0).fillna(0)
score_prop, n_prop, labels_prop, kmeans_prop = cluster_and_plot(
    X_prop,
    provinces,
    title_prefix="Proportional Structure",
    label_percentile=97
)

#--------------------------------------------------------------------------
# 2.6 Growth Patterns Clustering
#--------------------------------------------------------------------------
X_growth_list = []
offence_base_names = sorted({col.rsplit('_',1)[0] for col in df_data_numeric.columns})

for offence in offence_base_names:
    cols = sorted([c for c in df_data_numeric.columns if c.startswith(offence)])
    for i in range(1, len(cols)):
        prev = df_data_numeric[cols[i-1]]
        curr = df_data_numeric[cols[i]]
        growth = (curr - prev) / prev.replace(0, np.nan)
        X_growth_list.append(growth.fillna(0))

X_growth_df = pd.concat(X_growth_list, axis=1).fillna(0)

score_growth, n_growth, labels_growth, kmeans_growth = cluster_and_plot(
    X_growth_df,
    provinces,
    title_prefix="Crime Growth Patterns",
    label_percentile=70
)

#--------------------------------------------------------------------------
# 2.7 Summary of Silhouette Scores
#--------------------------------------------------------------------------
print("-" * LINE_SIZE)
print("\nSilhouette Scores Summary:")
print(f"Absolute Levels: {round(score_abs,3)} (n_clusters={n_abs})")
print(f"Proportional Structure: {round(score_prop,3)} (n_clusters={n_prop})")
print(f"Growth Patterns: {round(score_growth,3)} (n_clusters={n_growth})")













