import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack

# ---- Load dataset ----
df = pd.read_csv("/content/influenzaA_virus.csv")
df = df.dropna(subset=["Reverse_Translated_genome", "Protein_Sequence", "Human", "Species", "Family", "Host_agg"])

X_dna = df["Reverse_Translated_genome"]
X_protein = df["Protein_Sequence"]
y = df["Human"].astype(int)

# ---- Stratification key ----
df["strat_key"] = df["Species"].astype(str) + "_" + df["Family"].astype(str) + "_" + df["Host_agg"].astype(str)

# ---- K-mer function ----
def get_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence)-k+1)]

# ---- Feature extraction ----
# DNA: k=6
X_dna_kmers = [" ".join(get_kmers(seq, 6)) for seq in X_dna]
vectorizer_dna = CountVectorizer(max_features=5000)
X_dna_features = vectorizer_dna.fit_transform(X_dna_kmers)

# Protein: k=3
X_protein_kmers = [" ".join(get_kmers(seq, 3)) for seq in X_protein]
vectorizer_protein = CountVectorizer(max_features=5000)
X_protein_features = vectorizer_protein.fit_transform(X_protein_kmers)

# ---- Combine DNA and Protein features ----
X_features = hstack([X_dna_features, X_protein_features])

# ---- Feature selection ----
selector = SelectKBest(chi2, k=5000)
X_selected = selector.fit_transform(X_features, y)

# ---- Stratified Train-Test Split ----
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_selected, y, df.index, test_size=0.2, random_state=42, stratify=df["strat_key"]
)

# ---- SVM with Grid Search ----
param_grid = {'C':[0.1, 1, 10], 'gamma':['scale','auto'], 'kernel':['rbf','linear','poly']}
grid = GridSearchCV(
    SVC(probability=True, class_weight='balanced', random_state=42),
    param_grid, cv=5, scoring='roc_auc', n_jobs=-1
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"Best SVM Params: {grid.best_params_}, Best CV ROC-AUC: {grid.best_score_:.4f}")

# ---- Evaluation ----
y_prob = best_model.predict_proba(X_test)[:,1]
y_pred = (y_prob >= 0.5).astype(int)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# ---- Confusion Matrix ----
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\nTN: {cm[0,0]}, FP: {cm[0,1]}\nFN: {cm[1,0]}, TP: {cm[1,1]}")

# ---- Save results ----
test_df = df.loc[idx_test].copy()
test_df["Predicted_Human"] = y_pred
test_df["Spillover_Score"] = y_prob
test_df.to_csv("/content/result_svm_dna6_protein3.csv", index=False)
print("\nResults saved to result_svm_dna6_protein3.csv")
