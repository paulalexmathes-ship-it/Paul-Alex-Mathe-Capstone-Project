
"""
=============================================================================
MODULE 2 — ANALYTICS PIPELINE (Part A)
Capstone Project: Zepto Data & AI Platform
=============================================================================
01_eda.py — Profiling, Cleaning, and the Data Story
- Task 1: Load & profile the Titanic dataset
- Task 2: Missing-value handling (threshold-based)
- Task 3: Univariate analysis (histograms, box plots, IQR outliers)
- Task 4: Bivariate analysis (survival rates, correlation heatmap)
- Task 5: Multivariate data story (4+ charts with interpretation)
- Task 6: Exploratory standardization check (z-score)
=============================================================================
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Set output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# ============================================================================
# TASK 1: LOAD & PROFILE THE DATASET
# ============================================================================

print("=" * 70)
print("TASK 1: LOAD & PROFILE THE TITANIC DATASET")
print("=" * 70)

# Load dataset ONCE from seaborn (requires internet first time, cached after)
try:
    df = sns.load_dataset('titanic')
    print("\n✅ Dataset loaded successfully from seaborn (network/cache).")
except Exception as e:
    print(f"⚠️  Network load failed ({e}). Loading from local CSV fallback...")
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "titanic.csv"))

# Save the committed offline fallback immediately after loading
df.to_csv(os.path.join(OUTPUT_DIR, "titanic.csv"), index=False)
print(f"✅ Saved offline fallback: {os.path.join(OUTPUT_DIR, 'titanic.csv')}")

# --- Profiling ---
print("\n--- df.shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- df.info() ---")
print(df.info())

print("\n--- df.describe() ---")
print(df.describe().to_string())

print("\n--- df.head(10) ---")
print(df.head(10).to_string())

# --- Missing Values Report ---
print("\n--- Missing Values Report ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({
    'missing_count': missing,
    'missing_pct': missing_pct
})
missing_report = missing_report[missing_report['missing_count'] > 0].sort_values(
    'missing_pct', ascending=False
)
print(missing_report.to_string())

print(f"\nColumns with missing values: {len(missing_report)}")


# ============================================================================
# TASK 2: MISSING-VALUE HANDLING (Threshold Rule)
# ============================================================================

print("\n\n" + "=" * 70)
print("TASK 2: MISSING-VALUE HANDLING (Threshold Rule)")
print("=" * 70)

print("""
Threshold Rule Applied:
  • Under 5% missing   → DROP those rows
  • 5% – 30% missing   → IMPUTE
  • Over 30% missing    → DROP column or encode 'missing' as own category

Measured Missing Percentages:
""")

# Print strategy for each column
for col in missing_report.index:
    pct = missing_report.loc[col, 'missing_pct']
    count = missing_report.loc[col, 'missing_count']
    
    if pct < 5:
        strategy = "DROP ROWS (< 5%)"
    elif pct <= 30:
        strategy = "IMPUTE (5%–30%)"
    else:
        strategy = "DROP COLUMN or ENCODE MISSING (> 30%)"
    
    print(f"  {col:15s} → {pct:6.2f}% missing ({count} values) → Strategy: {strategy}")

# --- Apply Strategies ---
print("\n\nApplying strategies:")

# 'deck' column: ~77% missing → DROP the column
# Justification: With 77% of values missing, imputation would be unreliable 
# and introduce too much noise. No meaningful pattern can be preserved.
deck_pct = missing_report.loc['deck', 'missing_pct']
print(f"\n  1. 'deck' ({deck_pct:.2f}% missing) → DROPPING COLUMN")
print("     Justification: With ~77% missing, imputation would be unreliable.")
print("     The overwhelming majority of data is absent, so any fill strategy")
print("     would introduce artificial patterns rather than recover real signal.")
df.drop(columns=['deck'], inplace=True)

# 'age' column: ~19.9% missing → IMPUTE with median (robust to outliers)
age_pct = missing_report.loc['age', 'missing_pct']
age_median = df['age'].median()
print(f"\n  2. 'age' ({age_pct:.2f}% missing) → IMPUTING with median = {age_median}")
print("     Justification: 19.9% falls in the 5-30% range. Median is chosen over")
print("     mean because age has outliers that would skew the mean.")
df['age'].fillna(age_median, inplace=True)

# 'embarked' column: ~0.22% missing → DROP those rows
embarked_pct = missing_report.loc['embarked', 'missing_pct']
print(f"\n  3. 'embarked' ({embarked_pct:.2f}% missing) → DROPPING ROWS")
print("     Justification: Only 0.22% (2 rows) — safe to drop without data loss.")
df.dropna(subset=['embarked'], inplace=True)

# 'embark_town' column: same rows as 'embarked' — already handled
print(f"\n  4. 'embark_town' → Already resolved (same 2 rows as 'embarked')")

# Check for 'age' in the 'alive' column or other columns if applicable
remaining_nulls = df.isnull().sum()
remaining_nulls = remaining_nulls[remaining_nulls > 0]
if len(remaining_nulls) > 0:
    print(f"\n  Remaining nulls after cleaning:")
    print(remaining_nulls.to_string())
    # Drop any remaining nulls
    df.dropna(inplace=True)

print(f"\n✅ Cleaning complete!")
print(f"  Final dataset shape: {df.shape}")
print(f"  Remaining null values: {df.isnull().sum().sum()}")

# Reset index after row drops
df.reset_index(drop=True, inplace=True)


# ============================================================================
# TASK 3: UNIVARIATE ANALYSIS
# ============================================================================

print("\n\n" + "=" * 70)
print("TASK 3: UNIVARIATE ANALYSIS (Age & Fare)")
print("=" * 70)

# --- Histograms and Box Plots ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age histogram
axes[0, 0].hist(df['age'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribution of Age (Histogram)', fontsize=12)
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(df['age'].mean(), color='red', linestyle='--', label=f"Mean: {df['age'].mean():.1f}")
axes[0, 0].axvline(df['age'].median(), color='green', linestyle='-', label=f"Median: {df['age'].median():.1f}")
axes[0, 0].legend()

# Age box plot
axes[0, 1].boxplot(df['age'], vert=True)
axes[0, 1].set_title('Box Plot of Age', fontsize=12)
axes[0, 1].set_ylabel('Age')

# Fare histogram
axes[1, 0].hist(df['fare'], bins=30, color='coral', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Distribution of Fare (Histogram)', fontsize=12)
axes[1, 0].set_xlabel('Fare (£)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(df['fare'].mean(), color='red', linestyle='--', label=f"Mean: {df['fare'].mean():.1f}")
axes[1, 0].axvline(df['fare'].median(), color='green', linestyle='-', label=f"Median: {df['fare'].median():.1f}")
axes[1, 0].legend()

# Fare box plot
axes[1, 1].boxplot(df['fare'], vert=True)
axes[1, 1].set_title('Box Plot of Fare', fontsize=12)
axes[1, 1].set_ylabel('Fare (£)')

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "task3_histograms_boxplots.png"), dpi=150, bbox_inches='tight')
plt.show()
print(f"\n✅ Chart saved: charts/task3_histograms_boxplots.png")

# --- IQR Outlier Detection ---
print("\n--- IQR-Based Outlier Detection ---")


def detect_outliers_iqr(series, col_name):
    """Detect outliers using IQR rule: outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    
    print(f"\n  {col_name}:")
    print(f"    Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
    print(f"    Lower bound = {lower_bound:.2f}, Upper bound = {upper_bound:.2f}")
    print(f"    Number of outliers: {len(outliers)}")
    print(f"    Outlier percentage: {len(outliers)/len(series)*100:.2f}%")
    
    return outliers


age_outliers = detect_outliers_iqr(df['age'], 'Age')
fare_outliers = detect_outliers_iqr(df['fare'], 'Fare')

# --- Mean, Median, Mode for Fare + Skewness ---
print("\n\n--- Fare: Mean, Median, Mode & Skewness ---")
fare_mean = df['fare'].mean()
fare_median = df['fare'].median()
fare_mode = df['fare'].mode()[0]

print(f"  Mean:   £{fare_mean:.2f}")
print(f"  Median: £{fare_median:.2f}")
print(f"  Mode:   £{fare_mode:.2f}")
print(f"\n  Ordering: Mode ({fare_mode:.2f}) < Median ({fare_median:.2f}) < Mean ({fare_mean:.2f})")
print(f"\n  CONCLUSION: The fare distribution is RIGHT-SKEWED (positively skewed).")
print(f"  Evidence: Mean > Median > Mode. This is the classic signature of a")
print(f"  right-skewed distribution — a long tail of high-fare passengers (likely")
print(f"  first-class luxury cabins) pulls the mean above the median.")


# ============================================================================
# TASK 4: BIVARIATE ANALYSIS
# ============================================================================

print("\n\n" + "=" * 70)
print("TASK 4: BIVARIATE ANALYSIS")
print("=" * 70)

# --- 4a. Survival Rate by Sex (Boolean Masking) ---
print("\n--- 4a. Survival Rate by Sex ---")

male_survived = df[(df['sex'] == 'male') & (df['survived'] == 1)]
male_total = df[df['sex'] == 'male']
female_survived = df[(df['sex'] == 'female') & (df['survived'] == 1)]
female_total = df[df['sex'] == 'female']

male_survival_rate = len(male_survived) / len(male_total) * 100
female_survival_rate = len(female_survived) / len(female_total) * 100

print(f"  Male survival rate:   {male_survival_rate:.2f}% ({len(male_survived)}/{len(male_total)})")
print(f"  Female survival rate: {female_survival_rate:.2f}% ({len(female_survived)}/{len(female_total)})")

# --- 4b. Survival Rate by Pclass (Boolean Masking) ---
print("\n--- 4b. Survival Rate by Pclass ---")

for pclass in sorted(df['pclass'].unique()):
    class_survived = df[(df['pclass'] == pclass) & (df['survived'] == 1)]
    class_total = df[df['pclass'] == pclass]
    rate = len(class_survived) / len(class_total) * 100
    print(f"  Class {pclass} survival rate: {rate:.2f}% ({len(class_survived)}/{len(class_total)})")

# --- 4c. Survival Rate by Sex AND Pclass (Boolean Masking) ---
print("\n--- 4c. Survival Rate by Sex AND Pclass ---")

for sex in ['male', 'female']:
    for pclass in sorted(df['pclass'].unique()):
        group_survived = df[(df['sex'] == sex) & (df['pclass'] == pclass) & (df['survived'] == 1)]
        group_total = df[(df['sex'] == sex) & (df['pclass'] == pclass)]
        rate = len(group_survived) / len(group_total) * 100
        print(f"  {sex.capitalize()}, Class {pclass}: {rate:.2f}% ({len(group_survived)}/{len(group_total)})")

# --- 4d. Correlation Matrix (6 specified columns) ---
print("\n--- 4d. Correlation Matrix ---")
print("Columns: survived, pclass, age, sibsp, parch, fare")
print("(Excluding boolean columns: adult_male, alone — derived/redundant flags)")

corr_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
corr_matrix = df[corr_cols].corr()

print("\nCorrelation Matrix:")
print(corr_matrix.round(3).to_string())

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Correlation Matrix: survived, pclass, age, sibsp, parch, fare', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "task4_correlation_heatmap.png"), dpi=150, bbox_inches='tight')
plt.show()
print(f"\n✅ Chart saved: charts/task4_correlation_heatmap.png")

# --- Find Top 2 Strongest Correlations ---
print("\n--- Top 2 Strongest Off-Diagonal Correlations ---")

# Extract upper triangle (excluding diagonal)
corr_pairs = []
for i in range(len(corr_cols)):
    for j in range(i + 1, len(corr_cols)):
        corr_pairs.append({
            'Feature 1': corr_cols[i],
            'Feature 2': corr_cols[j],
            'Correlation': corr_matrix.iloc[i, j],
            'Abs Correlation': abs(corr_matrix.iloc[i, j])
        })

corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('Abs Correlation', ascending=False)
print(corr_pairs_df.head(5).to_string(index=False))

top1 = corr_pairs_df.iloc[0]
top2 = corr_pairs_df.iloc[1]

print(f"""
INTERPRETATION OF TWO STRONGEST CORRELATIONS:

1. {top1['Feature 1']} & {top1['Feature 2']} (r = {top1['Correlation']:.3f}):
   This is the strongest correlation. {'A negative value indicates an inverse relationship — ' if top1['Correlation'] < 0 else 'A positive value indicates a direct relationship — '}
   {'as pclass number increases (lower class), survival decreases significantly.' if 'pclass' in [top1['Feature 1'], top1['Feature 2']] and 'survived' in [top1['Feature 1'], top1['Feature 2']] else 'these features move together.'}
   This reflects the well-documented "women and children first" / class-based priority 
   in lifeboat access during the Titanic disaster.

2. {top2['Feature 1']} & {top2['Feature 2']} (r = {top2['Correlation']:.3f}):
   This is the second strongest correlation. {'Higher fare is associated with higher survival — ' if 'fare' in [top2['Feature 1'], top2['Feature 2']] and 'survived' in [top2['Feature 1'], top2['Feature 2']] else ''}
   {'Wealthier passengers (who paid higher fares) had better access to lifeboats, ' if top2['Correlation'] > 0 else ''}
   likely due to their cabins being closer to the boat deck and receiving 
   priority during evacuation.
""")


# ============================================================================
# TASK 5: MULTIVARIATE DATA STORY (4+ Charts)
# ============================================================================

print("\n\n" + "=" * 70)
print("TASK 5: MULTIVARIATE DATA STORY")
print("=" * 70)

# --- Chart 1: Survival Rate by Sex and Pclass (Grouped Bar) ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

survival_by_sex_class = df.groupby(['pclass', 'sex'])['survived'].mean().unstack()
survival_by_sex_class.plot(kind='bar', ax=axes[0, 0], color=['coral', 'steelblue'])
axes[0, 0].set_title('Chart 1: Survival Rate by Pclass and Sex', fontsize=12)
axes[0, 0].set_xlabel('Passenger Class')
axes[0, 0].set_ylabel('Survival Rate')
axes[0, 0].set_xticklabels(['Class 1', 'Class 2', 'Class 3'], rotation=0)
axes[0, 0].legend(title='Sex')
axes[0, 0].set_ylim(0, 1)

# --- Chart 2: Age Distribution by Survival Status (Box Plot) ---
sns.boxplot(data=df, x='survived', y='age', hue='sex', ax=axes[0, 1],
            palette='Set2')
axes[0, 1].set_title('Chart 2: Age Distribution by Survival & Sex', fontsize=12)
axes[0, 1].set_xlabel('Survived (0=No, 1=Yes)')
axes[0, 1].set_ylabel('Age')

# --- Chart 3: Fare vs Age colored by Survival (Scatter) ---
survived_mask = df['survived'] == 1
not_survived_mask = df['survived'] == 0
axes[1, 0].scatter(df.loc[not_survived_mask, 'age'], df.loc[not_survived_mask, 'fare'],
                   alpha=0.4, c='red', label='Did not survive', s=20)
axes[1, 0].scatter(df.loc[survived_mask, 'age'], df.loc[survived_mask, 'fare'],
                   alpha=0.4, c='green', label='Survived', s=20)
axes[1, 0].set_title('Chart 3: Fare vs Age by Survival Status', fontsize=12)
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Fare (£)')
axes[1, 0].legend()

# --- Chart 4: Survival Count by Embarkation Port and Class (Stacked Bar) ---
embark_surv = df.groupby(['embarked', 'pclass'])['survived'].sum().unstack()
embark_surv.plot(kind='bar', stacked=True, ax=axes[1, 1], 
                 colormap='viridis')
axes[1, 1].set_title('Chart 4: Survivors by Embarkation Port & Class', fontsize=12)
axes[1, 1].set_xlabel('Embarkation Port')
axes[1, 1].set_ylabel('Number of Survivors')
axes[1, 1].set_xticklabels(['Cherbourg', 'Queenstown', 'Southampton'], rotation=0)
axes[1, 1].legend(title='Pclass')

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "task5_multivariate_story.png"), dpi=150, bbox_inches='tight')
plt.show()
print(f"\n✅ Chart saved: charts/task5_multivariate_story.png")

print("""
CHART INTERPRETATIONS:

Chart 1 — Survival Rate by Pclass and Sex:
  Female passengers had dramatically higher survival rates across ALL classes.
  First-class females had ~97% survival vs first-class males at ~37%. The gender
  gap narrows slightly in third class but remains stark, confirming that "women
  and children first" was the dominant evacuation protocol.

Chart 2 — Age Distribution by Survival & Sex:
  Among survivors, the age distribution is slightly younger, suggesting children
  received priority. Male survivors tend to be younger than male non-survivors.
  Female survival was high regardless of age, reinforcing gender as the primary
  survival predictor over age.

Chart 3 — Fare vs Age by Survival Status:
  High-fare passengers (top of Y-axis) are predominantly green (survived),
  indicating wealth strongly correlated with survival. The cluster of red
  (non-survivors) at low fares reflects third-class passengers who had
  limited access to lifeboats due to their lower-deck cabin locations.

Chart 4 — Survivors by Embarkation Port & Class:
  Southampton contributed the most passengers overall but also the most deaths.
  Cherbourg had a higher proportion of first-class passengers, which explains
  its relatively better survival outcomes. Queenstown had mostly third-class
  passengers, reflected in fewer survivors.
""")


# ============================================================================
# TASK 6: EXPLORATORY STANDARDIZATION CHECK (z-score)
# ============================================================================

print("\n\n" + "=" * 70)
print("TASK 6: EXPLORATORY STANDARDIZATION (z-score check)")
print("=" * 70)

print("\nNote: This is purely an EDA-stage sanity check. It does NOT feed into")
print("the modeling pipeline, which performs its own train-only scaling.\n")

# Before standardization
print("--- BEFORE Standardization ---")
print(f"  Age  → Mean: {df['age'].mean():.4f}, Std: {df['age'].std():.4f}")
print(f"  Fare → Mean: {df['fare'].mean():.4f}, Std: {df['fare'].std():.4f}")

# Apply z-score: z = (x - mean) / std
df['age_zscore'] = (df['age'] - df['age'].mean()) / df['age'].std()
df['fare_zscore'] = (df['fare'] - df['fare'].mean()) / df['fare'].std()

# After standardization
print("\n--- AFTER Standardization (z-score) ---")
print(f"  Age (z-scored)  → Mean: {df['age_zscore'].mean():.6f}, Std: {df['age_zscore'].std():.6f}")
print(f"  Fare (z-scored) → Mean: {df['fare_zscore'].mean():.6f}, Std: {df['fare_zscore'].std():.6f}")

print("\n  ✅ Confirmed: Both transformed columns have approximately mean=0 and std=1.")

# Before/After comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].hist(df['age'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Age — Before Standardization')
axes[0, 0].set_xlabel('Age (years)')

axes[0, 1].hist(df['age_zscore'], bins=30, color='darkblue', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Age — After Z-score Standardization')
axes[0, 1].set_xlabel('Z-score')

axes[1, 0].hist(df['fare'], bins=30, color='coral', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Fare — Before Standardization')
axes[1, 0].set_xlabel('Fare (£)')

axes[1, 1].hist(df['fare_zscore'], bins=30, color='darkred', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Fare — After Z-score Standardization')
axes[1, 1].set_xlabel('Z-score')

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "task6_standardization_comparison.png"), dpi=150, bbox_inches='tight')
plt.show()
print(f"\n✅ Chart saved: charts/task6_standardization_comparison.png")

# Drop z-score columns (EDA only — not for modeling)
df.drop(columns=['age_zscore', 'fare_zscore'], inplace=True)

# Save the final cleaned data (this is what 02_modeling.py will read)
df.to_csv(os.path.join(OUTPUT_DIR, "titanic.csv"), index=False)
print(f"\n✅ Final cleaned dataset saved: titanic.csv ({df.shape[0]} rows × {df.shape[1]} cols)")

print("\n\n" + "=" * 70)
print("PART A COMPLETE ✅ — Proceed to 02_modeling.py")
print("=" * 70)

