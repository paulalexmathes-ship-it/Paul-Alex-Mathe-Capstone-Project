
import os, warnings, numpy as np, pandas as pd, joblib, shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)
print("=" * 60)
print("SHAP EXPLAINABILITY ANALYSIS")
print("=" * 60)

print("\n[1/8] Loading saved pipeline...")
pipeline = joblib.load("best_pipeline.joblib")
print("      Pipeline steps:", [name for name, _ in pipeline.steps])

print("\n[2/8] Loading and splitting data...")
df = pd.read_csv("titanic.csv")
feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target_col = 'survived'
X = df[feature_cols].copy()
y = df[target_col].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"      X_train: {X_train.shape}  |  X_test: {X_test.shape}")

print("\n[3/8] Extracting pipeline components...")
preprocessor = pipeline[0]
classifier = pipeline[-1]
print(f"      Preprocessor: {type(preprocessor).__name__}")
print(f"      Classifier:   {type(classifier).__name__}")
print(f"      n_estimators:  {classifier.n_estimators}")
print(f"      max_depth:     {classifier.max_depth}")

print("\n[4/8] Transforming X_test through preprocessor...")
X_test_transformed = preprocessor.transform(X_test)
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()
print(f"      Transformed shape: {X_test_transformed.shape}")

print("\n[5/8] Extracting transformed feature names...")
try:
    feature_names = preprocessor.get_feature_names_out()
except AttributeError:
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            if transformer == "passthrough":
                feature_names.extend(columns)
            continue
        if hasattr(transformer, "get_feature_names_out"):
            feature_names.extend(transformer.get_feature_names_out(columns))
        else:
            feature_names.extend(columns)
    feature_names = np.array(feature_names)
clean_names = [n.split("__")[-1] if "__" in n else n for n in feature_names]
print(f"      {len(clean_names)} features: {clean_names}")
X_test_df = pd.DataFrame(X_test_transformed, columns=clean_names)

print("\n[6/8] Computing SHAP values (may take 15-30 seconds)...")
explainer = shap.TreeExplainer(classifier)
shap_values_raw = explainer.shap_values(X_test_df)
print(f"      Return type: {type(shap_values_raw)}")
if isinstance(shap_values_raw, list):
    print(f"      List of {len(shap_values_raw)} arrays, each shape: {shap_values_raw[0].shape}")
else:
    print(f"      Array shape: {shap_values_raw.shape}, ndim: {shap_values_raw.ndim}")

# HANDLE EVERY POSSIBLE RETURN TYPE
if isinstance(shap_values_raw, list):
    sv_survived = shap_values_raw[1]
elif isinstance(shap_values_raw, np.ndarray):
    if shap_values_raw.ndim == 3:
        sv_survived = shap_values_raw[:, :, 1]
    else:
        sv_survived = shap_values_raw
else:
    sv_survived = np.array(shap_values_raw)

print(f"      sv_survived shape: {sv_survived.shape}")
print(f"      Expected value: {explainer.expected_value}")

if isinstance(explainer.expected_value, (list, np.ndarray)):
    base_val = float(explainer.expected_value[1])
else:
    base_val = float(explainer.expected_value)

print("\n[7/8] Generating SHAP visualizations...")

print("      -> Chart 1: Bee swarm plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(sv_survived, X_test_df, feature_names=clean_names, show=False, plot_size=(12, 8))
plt.title("SHAP Summary - Feature Impact on Survival Prediction", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "shap_summary_beeswarm.png"), dpi=150, bbox_inches="tight")
plt.close()
print("        Saved: charts/shap_summary_beeswarm.png")

print("      -> Chart 2: Feature importance bar...")
plt.figure(figsize=(10, 6))
shap.summary_plot(sv_survived, X_test_df, feature_names=clean_names, plot_type="bar", show=False, plot_size=(10, 6))
plt.title("SHAP Global Feature Importance", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "shap_feature_importance_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("        Saved: charts/shap_feature_importance_bar.png")

print("      -> Chart 3: Waterfall plot...")
survived_idx = np.where(y_test.values == 1)[0]
si = survived_idx[0] if len(survived_idx) > 0 else 0
expl = shap.Explanation(values=sv_survived[si], base_values=base_val, data=X_test_df.iloc[si].values, feature_names=clean_names)
plt.figure(figsize=(10, 6))
shap.waterfall_plot(expl, show=False)
plt.title(f"SHAP Waterfall - Passenger #{si} (Actual: Survived)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "shap_waterfall_single.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"        Saved: charts/shap_waterfall_single.png")

print("      -> Chart 4: Dependence plot...")
mean_abs = np.abs(sv_survived).mean(axis=0)
ti = np.argmax(mean_abs)
tn = clean_names[ti]
plt.figure(figsize=(10, 6))
shap.dependence_plot(ti, sv_survived, X_test_df, feature_names=clean_names, show=False)
plt.title(f"SHAP Dependence - {tn} (Top Predictor)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "shap_dependence_top_feature.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"        Saved: charts/shap_dependence_top_feature.png")

print("\n[8/8] SHAP Interpretation Summary")
print("=" * 60)
fi = pd.DataFrame({"Feature": clean_names, "Mean |SHAP|": np.abs(sv_survived).mean(axis=0)})
fi = fi.sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
print("\nTop Features by Mean |SHAP Value|:\n")
print(fi.to_string(index=False))
t1, t2, t3 = fi.iloc[0]["Feature"], fi.iloc[1]["Feature"], fi.iloc[2]["Feature"]
print(f"\nKEY FINDINGS:")
print(f"1. '{t1}' is the STRONGEST predictor of survival.")
print(f"2. '{t2}' is the 2nd most influential feature.")
print(f"3. '{t3}' is the 3rd most influential feature.")
print(f"\n4 SHAP charts saved to: {os.path.abspath(CHARTS_DIR)}/")
print("=" * 60)
print("SHAP explainability analysis complete.")


