#--------------------------------------------------------------------------
# Capstone Project:
# This project analyzes aggregated crime data for all localities in the
# Atlantic Canadian provinces: Nova Scotia, New Brunswick, Newfoundland
# and Labrador, and Prince Edward Island over the period 2020–2024.
# The main goal is to identify patterns in crime data and understand
# structural similarities and differences in criminal activity across
# these regions.
#
# Student ID: DA21106
# Name: Silvana Paredes
#--------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    "Codiac Regional",
    "Headquarters",
    "offshore"
]

# Define the base-level columns (recommended for analysis)
columns_for_analysis = [
    "Total violent Criminal Code violations",
    "Total property crime violations",
    "Total other Criminal Code violations",
    "Total Criminal Code traffic violations",
    "Total Drug violations",
    "Total Cannabis Act",
    "Youth Criminal Justice Act",
    "Total other Federal Statutes"
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
print("Shape before cleaning:", df.shape)

# Remove not needed row on the header
df = df.drop(index=2).reset_index(drop=True)

# Drop total columns (from 1 to 7) with totals
df = df.drop(df.columns[1:8], axis=1)

# Replace null values
df = df.fillna(0)

#--------------------------------------------------------------------------
# 1.1.2 Drop rows containing provincial totals (redundant data)
#--------------------------------------------------------------------------
# Rename column 0 to 'Locations'
df.rename(columns={df.columns[0]: "Locations"}, inplace=True)

# Drop rows that start with a province name
mask = df["Locations"].astype(str).str.strip().str.startswith(tuple(province_mapping.keys()))
df = df[~mask]
df.reset_index(drop=True, inplace=True)

#--------------------------------------------------------------------------
# 1.1.3 Drop rows with values that do not correspond to cities
#--------------------------------------------------------------------------

# Remove rows where 'Locations' starts with any pattern in remove_patterns
df = df[~df['Locations'].astype(str).str.strip().str.startswith(tuple(remove_patterns))]

# Remove rows where 'Locations' contains any pattern in remove_patterns (case-insensitive)
pattern_regex = '|'.join([re.escape(pat) for pat in remove_patterns])
df = df[~df['Locations'].str.contains(pattern_regex, case=False, na=False)].reset_index(drop=True)

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
# Select only data columns (ignore the first column 'Locations')
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
# 1.1.6 Dataset Overview
#--------------------------------------------------------------------------

row_locations = df.shape[0]

print("\nShape after cleaning:", df.shape)
print("\nColumns selected for analysis:")
print(columns_for_analysis)


#--------------------------------------------------------------------------
# 1.2 Data manipulation
#--------------------------------------------------------------------------
# 1.2.1 Replace province names with their abbreviations
#--------------------------------------------------------------------------

# Replace full_name with abbreviation, discard everything after it
for full_name, abbr in province_mapping.items():
    df["Locations"] = df["Locations"].str.replace(rf"{full_name}.*", abbr, regex=True)

# Strip leading/trailing whitespace
df["Locations"] = df["Locations"].str.strip()


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
    else:
        year = str(year_raw).strip()  # keep as string (e.g., "Unnamed")

    # Update current crime if column name is not "Unnamed"
    if not str(col_name).startswith("Unnamed"):
        current_crime = col_name.strip()

    # Combine crime name with year
    new_columns.append(f"{current_crime}_{year}")

df.columns = new_columns


#--------------------------------------------------------------------------
# 1.2.3 Select recommended offence columns for analysis
#--------------------------------------------------------------------------

# Save Locations aligned with numeric data
locations = df['Locations'].iloc[1:].values  # exclude header row

# Keep only the recommended offence columns (all years)
columns_for_analysis_full = [
    col for col in df.columns
    if any(col.startswith(base) for base in columns_for_analysis)
]

# Create df_numeric with only numeric data for analysis
df_numeric = df[columns_for_analysis_full].iloc[1:].apply(pd.to_numeric, errors='coerce')

# Optional: inspect
print("\nOffences dataset:", df_numeric.head())

print("\nLocations:")
for i, loc in enumerate(df.iloc[1:, 0], start=1):
    print(f"{i}: {loc}")


#--------------------------------------------------------------------------
# 2. Machine Learning: Clustering and PCA
#--------------------------------------------------------------------------
print("-" * LINE_SIZE)
print("2. Machine Learning")
print("-" * LINE_SIZE)

#--------------------------------------------------------------------------
# 2.1 Generic function: show High Crime locations
#--------------------------------------------------------------------------

def show_high_crime_locations(locations_array, labels_array, cluster_names={0: 'Low Crime', 1: 'High Crime'}, title="High Crime Locations"):
    """
    Prints locations classified as High Crime along with their total count.
    """
    cluster_df = pd.DataFrame({
        'Location': locations_array,
        'ClusterLabel': labels_array
    })
    
    # Map numeric clusters to descriptive labels
    cluster_df['ClusterLabel'] = cluster_df['ClusterLabel'].map(cluster_names)
    
    # Select High Crime rows
    high_crime_locations = cluster_df[cluster_df['ClusterLabel'] == 'High Crime']
    
    # Print title, count and the full list
    print(f"\n{title} (count={len(high_crime_locations)}):")
    print(high_crime_locations)
    print("-" * LINE_SIZE)
    
    return high_crime_locations

#--------------------------------------------------------------------------
# 2.2 Generic function: cluster, PCA, plot, return labels
#--------------------------------------------------------------------------

def cluster_and_plot(X_numeric, province_names, title_prefix="", label_percentile=60, analysis_type="absolute"):
    """
    Scale, KMeans clustering, PCA and scatter plotting.
    Uses descriptive cluster labels and a neutral color palette.
    """
    # Fill missing values
    X_numeric = X_numeric.fillna(0)

    # Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    # Range of clusters
    n_range = range(2, 10)
    best_score, best_n, best_labels, best_model = -1, None, None, None
    scores = []

    for n in n_range:
        kmeans = KMeans(n_clusters=n, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
        if score > best_score:
            best_score, best_n, best_labels, best_model = score, n, labels, kmeans

    # Silhouette plot
    plt.figure(figsize=(8,4))
    sns.lineplot(x=list(n_range), y=scores, marker='o', color='grey')
    plt.axvline(best_n, color='black', linestyle='--', label=f'Best n={best_n}')
    plt.xlabel("n_clusters")
    plt.ylabel("Silhouette Score")
    plt.title(f"{title_prefix} - Silhouette Scores")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(best_model.cluster_centers_)

    # PCA variance explained
    var_ratio = pca.explained_variance_ratio_
    print(f"\n{title_prefix} - PCA Variance Explained:")
    print(f"PC1: {var_ratio[0]:.3f}")
    print(f"PC2: {var_ratio[1]:.3f}")
    print(f"Total (2 PCs): {var_ratio.sum():.3f}")

    # Assign descriptive labels
    cluster_summary = pd.DataFrame(X_numeric)
    cluster_summary["cluster"] = best_labels
    cluster_means = cluster_summary.groupby("cluster").mean().mean(axis=1)
    sorted_clusters = cluster_means.sort_values().index

    # Label names depending on analysis type
    if analysis_type == "absolute":
        if best_n == 2:
            label_names = ["Low Crime", "High Crime"]
        elif best_n == 3:
            label_names = ["Low Crime", "Medium Crime", "High Crime"]
        else:
            label_names = [f"Cluster {i}" for i in range(best_n)]

    elif analysis_type == "proportional":
        label_names = [f"Pattern {i+1}" for i in range(best_n)]

    elif analysis_type == "growth":
        label_names = [f"Growth Group {i+1}" for i in range(best_n)]

    else:
        label_names = [f"Cluster {i}" for i in range(best_n)]

    # Print clusters
    cluster_labels_map = {sorted_clusters[i]: label_names[i] for i in range(best_n)}
    cluster_summary["cluster_label"] = cluster_summary["cluster"].map(cluster_labels_map)

    print(f"\n{title_prefix} - Cluster Size (Descriptive Labels):")
    print(cluster_summary["cluster_label"].value_counts().sort_index())
    print(f"\n{title_prefix} - Cluster Feature Means:")
    print(cluster_summary.groupby("cluster_label").mean())

    # PCA DataFrame
    center = X_pca.mean(axis=0)
    distances = np.linalg.norm(X_pca - center, axis=1)
    label_threshold = np.percentile(distances, label_percentile)

    pca_df = pd.DataFrame({
        'PC1': X_pca[:,0],
        'PC2': X_pca[:,1],
        'Cluster': best_labels,
        'Province': province_names,
        'Distance': distances
    })
    pca_df['Cluster'] = pca_df['Cluster'].map(cluster_labels_map)
    pca_df['Label'] = pca_df.apply(lambda row: row['Province'] if row['Distance'] > label_threshold else '', axis=1)

    # Scatter plot with neutral palette
    neutral_palette = sns.color_palette("tab10", best_n)
    plt.figure(figsize=(14,10))
    sns.scatterplot(
        data=pca_df, x='PC1', y='PC2',
        hue='Cluster', palette=neutral_palette,
        s=120, alpha=0.7, edgecolor='k'
    )

    # Plot centroids only if cluster has more than 1 point
    cluster_counts = pd.Series(best_labels).value_counts()

    for i, (x, y) in enumerate(centroids_pca):
        if cluster_counts[i] > 1:  # only plot if cluster has more than 1 observation
            plt.scatter(
                x, y,
                marker='X',
                s=250,
                facecolors='none',   # hollow marker (does not hide points)
                edgecolor='black',
                linewidth=2,
                zorder=3
            )

    # Annotate province labels
    for i, row in pca_df.iterrows():
        if row['Label']:
            x_range = X_pca[:,0].max() - X_pca[:,0].min()
            x_offset = 0.02 * x_range  

            plt.text(
                row['PC1']+x_offset,
                row['PC2'],
                row['Label'],
                fontsize=9,
                bbox=dict(facecolor='white', alpha=1, edgecolor='none'))

    plt.title(f"{title_prefix} - PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return best_score, best_n, best_labels, best_model

#--------------------------------------------------------------------------
# 2.3 Absolute Levels Clustering
#--------------------------------------------------------------------------

score_abs, n_abs, labels_abs, kmeans_abs = cluster_and_plot(
    df_numeric,
    locations,
    title_prefix="Absolute Crime Levels",
    label_percentile=90,
    analysis_type="absolute"
)

show_high_crime_locations(locations, labels_abs, title="Absolute Levels - High Crime Locations")

#--------------------------------------------------------------------------
# 2.4 Proportional Structure Clustering
#--------------------------------------------------------------------------

X_prop = df_numeric.div(df_numeric.sum(axis=1), axis=0).fillna(0)
score_prop, n_prop, labels_prop, kmeans_prop = cluster_and_plot(
    X_prop,
    locations,
    title_prefix="Proportional Structure",
    label_percentile=90,
    analysis_type="proportional"
)

show_high_crime_locations(locations, labels_prop, title="Proportional Structure - High Crime Locations")

#--------------------------------------------------------------------------
# 2.5 Growth Patterns Clustering
#--------------------------------------------------------------------------

X_growth_list = []
offence_base_names = sorted({col.rsplit('_',1)[0] for col in df_numeric.columns})

for offence in offence_base_names:
    cols = sorted([c for c in df_numeric.columns if c.startswith(offence)])
    for i in range(1, len(cols)):
        prev = df_numeric[cols[i-1]]
        curr = df_numeric[cols[i]]
        growth = (curr - prev) / prev.replace(0, np.nan)
        X_growth_list.append(growth.fillna(0))

X_growth_df = pd.concat(X_growth_list, axis=1).fillna(0)

score_growth, n_growth, labels_growth, kmeans_growth = cluster_and_plot(
    X_growth_df,
    locations,
    title_prefix="Crime Growth Patterns",
    label_percentile=70,
    analysis_type="growth"
)

show_high_crime_locations(locations, labels_growth, title="Growth Patterns - High Crime Locations")

#--------------------------------------------------------------------------
# 2.6 Summary of Silhouette Scores
#--------------------------------------------------------------------------

print("\nSilhouette Scores Summary:")
print(f"Absolute Levels: {round(score_abs,3)} (n_clusters={n_abs})")
print(f"Proportional Structure: {round(score_prop,3)} (n_clusters={n_prop})")
print(f"Growth Patterns: {round(score_growth,3)} (n_clusters={n_growth})")
