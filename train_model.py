import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# =====================================
# MEMBACA DATASET
# =====================================

data = pd.read_excel("segmentasi intensitas workout.xlsx")

data = data.rename(columns={
    'Session_Duration (hours)': 'Duration',
    'Calories_Burned': 'Calories'
})

# =====================================
# FITUR UNTUK CLUSTERING
# =====================================

cluster_features = ['Duration', 'Calories', 'Avg_BPM', 'Max_BPM']
features = data[cluster_features]

# =====================================
# SCALING UNTUK K-MEANS
# =====================================

scaler_kmeans = StandardScaler()
features_scaled = scaler_kmeans.fit_transform(features)

# =====================================
# K-MEANS CLUSTERING
# =====================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(features_scaled)
data['Cluster'] = kmeans.labels_

# =====================================
# MELIHAT RATA-RATA TIAP CLUSTER
# =====================================

cluster_info = data.groupby('Cluster')[cluster_features].mean()
print("\nRata-rata tiap cluster:\n")
print(cluster_info)

# =====================================
# MAPPING CLUSTER KE LABEL (BERDASARKAN CALORIES)
# =====================================
# Calories Burned adalah indikator utama intensitas workout.
# Semakin banyak kalori terbakar = semakin tinggi intensitas.
# Kita rank cluster berdasarkan rata-rata Calories saja
# agar mapping lebih tepat secara domain.

calories_mean = cluster_info['Calories']
rank = calories_mean.rank()  # 1=terendah, 3=tertinggi

label_map = {}
for cluster_id, r in rank.items():
    if r == 3:
        label_map[cluster_id] = 'Tinggi'
    elif r == 2:
        label_map[cluster_id] = 'Sedang'
    else:
        label_map[cluster_id] = 'Rendah'

print("\nMapping cluster ke label (berdasarkan Calories):")
for cid, label in label_map.items():
    print(f"  Cluster {cid} → {label} (avg Calories: {calories_mean[cid]:.1f})")

data['Intensity'] = data['Cluster'].map(label_map)

print("\nDistribusi label intensitas:")
print(data['Intensity'].value_counts())

print("\nVerifikasi rata-rata per label:")
print(data.groupby('Intensity')[cluster_features].mean().round(2))

# =====================================
# FITUR UNTUK NAIVE BAYES
# =====================================

nb_features = ['Duration', 'Calories', 'Avg_BPM', 'Max_BPM']
X = data[nb_features]
y = data['Intensity']

# =====================================
# SCALING UNTUK NAIVE BAYES
# =====================================

scaler_nb = StandardScaler()
X_scaled = scaler_nb.fit_transform(X)

# =====================================
# SPLIT DATA
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# MEMBUAT MODEL
# =====================================

model = GaussianNB()
model.fit(X_train, y_train)

# =====================================
# EVALUASI MODEL
# =====================================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Model (test set): {accuracy:.4f} ({accuracy*100:.2f}%)")

cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
print(f"Akurasi Cross-Validation (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=['Tinggi', 'Sedang', 'Rendah'])
cm_df = pd.DataFrame(
    cm,
    index=['Aktual Tinggi', 'Aktual Sedang', 'Aktual Rendah'],
    columns=['Prediksi Tinggi', 'Prediksi Sedang', 'Prediksi Rendah']
)
print(cm_df)

# =====================================
# MENYIMPAN MODEL & SCALER
# =====================================

joblib.dump(model,         'workout_model.pkl')
joblib.dump(scaler_nb,     'scaler_nb.pkl')
joblib.dump(scaler_kmeans, 'scaler_kmeans.pkl')
joblib.dump(kmeans,        'kmeans_model.pkl')
joblib.dump(label_map,     'label_map.pkl')

print("\nSemua file berhasil disimpan!")

# =====================================
# SANITY CHECK PREDIKSI
# =====================================

print("\n--- Sanity Check Prediksi ---")
test_cases = [
    {"Duration": 1.0, "Calories": 800,  "Avg_BPM": 170, "Max_BPM": 200, "expected": "Sedang"},
    {"Duration": 0.5, "Calories": 350,  "Avg_BPM": 125, "Max_BPM": 165, "expected": "Rendah"},
    {"Duration": 2.0, "Calories": 1400, "Avg_BPM": 160, "Max_BPM": 195, "expected": "Tinggi"},
]

for tc in test_cases:
    expected = tc.pop("expected")
    raw    = np.array([[tc["Duration"], tc["Calories"], tc["Avg_BPM"], tc["Max_BPM"]]])
    scaled = scaler_nb.transform(raw)
    pred   = model.predict(scaled)[0]
    proba  = model.predict_proba(scaled)[0]
    prob_dict = dict(zip(model.classes_, proba.round(3)))
    status = "✓" if pred == expected else "✗"
    print(f"{status} Input: {tc} → Prediksi: {pred} (expected: {expected}) | Proba: {prob_dict}")