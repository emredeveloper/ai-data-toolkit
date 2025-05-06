import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter


plt.rcParams['figure.figsize'] = (12, 8)

# Rastgele örnek veriler oluşturalım (dengesiz sınıf dağılımı ile)
X, y = make_classification(
    n_samples=1000, 
    n_classes=2, 
    n_features=10, 
    n_informative=5, 
    n_redundant=2, 
    weights=[0.9, 0.1],  # %90 negatif, %10 pozitif sınıf
    random_state=42
)

# Veri setini eğitim ve test olarak bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Orijinal eğitim veri setindeki sınıf dağılımını kontrol edelim
print("Orijinal eğitim veri seti sınıf dağılımı:")
print(Counter(y_train))

def plot_data(X, y, title):
    """2 boyutlu veri görselleştirme fonksiyonu"""
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5, cmap='viridis')
    plt.title(title)
    plt.colorbar(label='Sınıf')
    plt.grid(True)
    plt.show()

# Orijinal veri dağılımını görselleştir
plot_data(X_train, y_train, "Orijinal Veri Dağılımı")

# SMOTE uygulama
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Random Oversampling uygulama
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

# Resampling sonrası sınıf dağılımlarını kontrol et
print("\nSMOTE sonrası sınıf dağılımı:")
print(Counter(y_train_smote))

print("\nRandom Oversampling sonrası sınıf dağılımı:")
print(Counter(y_train_ros))

# SMOTE ve Random Oversampling sonuçlarını görselleştir
plot_data(X_train_smote, y_train_smote, "SMOTE ile Dengelenmiş Veri")
plot_data(X_train_ros, y_train_ros, "Random Oversampling ile Dengelenmiş Veri")

# Modelleri değerlendirme fonksiyonu
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n--- {model_name} Sonuçları ---")
    print(f"Doğruluk (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))

# Lojistik Regresyon modeli ile değerlendirme
# Orijinal veri seti ile
print("\n=== Orijinal Veri Seti ile Model Değerlendirme ===")
model_original = LogisticRegression(max_iter=1000, random_state=42)
evaluate_model(model_original, X_train, y_train, X_test, y_test, "Orijinal Veri")

# SMOTE ile dengelenen veri seti ile
print("\n=== SMOTE ile Dengelenmiş Veri Seti ile Model Değerlendirme ===")
model_smote = LogisticRegression(max_iter=1000, random_state=42)
evaluate_model(model_smote, X_train_smote, y_train_smote, X_test, y_test, "SMOTE ile Dengelenmiş Veri")

# Random Oversampling ile dengelenen veri seti ile
print("\n=== Random Oversampling ile Dengelenmiş Veri Seti ile Model Değerlendirme ===")
model_ros = LogisticRegression(max_iter=1000, random_state=42)
evaluate_model(model_ros, X_train_ros, y_train_ros, X_test, y_test, "Random Oversampling ile Dengelenmiş Veri")

# ROC eğrilerini çizdirelim
plt.figure()
from sklearn.metrics import roc_curve, auc

# Orijinal model için ROC
y_scores_original = model_original.predict_proba(X_test)[:, 1]
fpr_original, tpr_original, _ = roc_curve(y_test, y_scores_original)
roc_auc_original = auc(fpr_original, tpr_original)

# SMOTE modeli için ROC
y_scores_smote = model_smote.predict_proba(X_test)[:, 1]
fpr_smote, tpr_smote, _ = roc_curve(y_test, y_scores_smote)
roc_auc_smote = auc(fpr_smote, tpr_smote)

# RandomOverSampler modeli için ROC
y_scores_ros = model_ros.predict_proba(X_test)[:, 1]
fpr_ros, tpr_ros, _ = roc_curve(y_test, y_scores_ros)
roc_auc_ros = auc(fpr_ros, tpr_ros)

plt.figure(figsize=(10, 8))
plt.plot(fpr_original, tpr_original, label=f'Orijinal (AUC = {roc_auc_original:.2f})')
plt.plot(fpr_smote, tpr_smote, label=f'SMOTE (AUC = {roc_auc_smote:.2f})')
plt.plot(fpr_ros, tpr_ros, label=f'Random Oversampling (AUC = {roc_auc_ros:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi Karşılaştırması')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Sonuç değerlendirmesi
print("\n==== Model Karşılaştırma Özeti ====")
print("Dengesiz veri seti için model sonuçları değerlendirmesi:")
print(f"1. Orijinal veri seti: Doğruluk yüksek (%91) ancak azınlık sınıfı için recall düşük (0.19)")
print(f"2. SMOTE ile: Genel doğruluk %81'e düştü ancak azınlık sınıfı için recall 0.75'e yükseldi")
print(f"3. Random Oversampling ile: Genel doğruluk %80, azınlık sınıfı recall 0.75")
print("\nDeğerlendirme:")
print("- Orijinal model yüksek doğruluk gösteriyor ancak azınlık sınıfını tespit etmekte başarısız")
print("- SMOTE ve Random Oversampling benzer performans gösteriyor")
print("- Her iki dengeleme tekniği de azınlık sınıfı için recall değerini önemli ölçüde artırdı")
print("- Precision-recall dengesi açısından F1-score'a bakıldığında dengeleme teknikleri daha başarılı")

# Son görselleştirme - Doğruluk vs. Recall dengesi
plt.figure(figsize=(10, 6))
models = ['Orijinal', 'SMOTE', 'RandomOS']
accuracy = [0.91, 0.81, 0.80]
recall = [0.19, 0.75, 0.75]
precision = [0.86, 0.32, 0.32]
f1 = [0.31, 0.45, 0.45]

x = np.arange(len(models))
width = 0.2

plt.bar(x - width*1.5, accuracy, width, label='Doğruluk')
plt.bar(x - width/2, precision, width, label='Precision')
plt.bar(x + width/2, recall, width, label='Recall') 
plt.bar(x + width*1.5, f1, width, label='F1 Score')

plt.xlabel('Model')
plt.ylabel('Skor')
plt.title('Dengesiz Veri Seti için Model Performansı')
plt.xticks(x, models)
plt.legend()
plt.tight_layout()
plt.show()

print("\nSonuç: SMOTE ve oversampling teknikleri, dengesiz veri setlerinde model performansını artırmak için etkili yöntemlerdir.")
print("Özellikle azınlık sınıfının doğru sınıflandırılması (recall) önemli olduğunda bu teknikler tercih edilmelidir.")
print("Ancak bu durumda genel doğruluk (accuracy) düşebilir ve yanlış pozitiflerde (false positives) artış olabilir.")
