import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2

# Load dataset
df = pd.read_csv("./influenzaA_virus.csv")
df = df.dropna(subset=["Reverse_Translated_genome", "Human", "Species", "Family", "Host_agg"])

X_dna = df["Reverse_Translated_genome"].astype(str)
y = df["Human"].astype(int)

# Stratification key
df["strat_key"] = df["Species"].astype(str) + "_" + df["Family"].astype(str) + "_" + df["Host_agg"].astype(str)

# k-mer function
def get_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence)-k+1)]

# Train-test split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_dna, y, df.index, test_size=0.2, random_state=42, stratify=df["strat_key"]
)

# Create k-mer "documents" for DNA
X_train_docs = [" ".join(get_kmers(seq, 3)) for seq in X_train] # change k-mer size as requirement
X_test_docs  = [" ".join(get_kmers(seq, 3)) for seq in X_test] # change k-mer size as requirement

# Vectorize
vectorizer = CountVectorizer(max_features=5000)
X_train_features = vectorizer.fit_transform(X_train_docs)
X_test_features  = vectorizer.transform(X_test_docs)

# Feature selection
k = min(5000, X_train_features.shape[1])
selector = SelectKBest(chi2, k=k)
X_train_sel = selector.fit_transform(X_train_features, y_train)
X_test_sel  = selector.transform(X_test_features)

# SVM with GridSearchCV
param_grid = {'C':[0.1, 1, 10], 'gamma':['scale','auto'], 'kernel':['rbf','linear','poly']}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(SVC(probability=True, class_weight='balanced', random_state=42),
                    param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train_sel, y_train)

best_model = grid.best_estimator_
print(f"DNA Only - Best Params: {grid.best_params_}, Best CV ROC-AUC: {grid.best_score_:.4f}")

# Evaluation
y_prob = best_model.predict_proba(X_test_sel)[:,1]
y_pred = (y_prob >= 0.5).astype(int)
print("\nDNA Only - Classification Report:\n", classification_report(y_test, y_pred))
print("DNA Only - ROC-AUC Score:", roc_auc_score(y_test, y_prob))
cm = confusion_matrix(y_test, y_pred)
print(f"DNA Only - Confusion Matrix:\nTN: {cm[0,0]}, FP: {cm[0,1]}\nFN: {cm[1,0]}, TP: {cm[1,1]}")
