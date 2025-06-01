from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, classification_report
import numpy as np
import matplotlib.pyplot as plt

# Iris veri setini yükle
iris = load_iris()
X = iris.data
# İkili sınıflandırma için sadece iki sınıfı alalım (setosa ve versicolor)
y = (iris.target != 0) * 1  # setosa=0, diğerleri=1

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Basit bir lojistik regresyon modeli eğitelim
model = LogisticRegression(max_iter=200,solver="liblinear")
model.fit(X_train, y_train)

# Test seti üzerinde tahmin olasılıklarını al
y_scores = model.predict_proba(X_test)[:, 1]

# Precision-Recall eğrisi için gerekli metrikleri hesapla
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# F1 skorlarını hesapla (threshold'lar için)
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)  # Sıfıra bölünmeyi önlemek için küçük bir değer ekledik
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

# Sonuçları yazdır
print(f"Optimal Eşik Değeri: {optimal_threshold:.4f}")
print(f"Bu eşik değeri için F1 Skoru: {f1_scores[optimal_idx]:.4f}")

# Eğriyi görselleştir
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
plt.plot(thresholds, f1_scores, 'r-', label='F1 Score')
plt.axvline(x=optimal_threshold, color='k', linestyle='--', label=f'Optimal Eşik ({optimal_threshold:.2f})')
plt.xlabel('Eşik Değeri')
plt.title('Precision, Recall ve F1 Skorlarının Eşik Değerine Göre Değişimi')
plt.legend()
plt.grid(True)
plt.show()

# Modelin performansını değerlendir
y_pred = (y_scores >= optimal_threshold).astype(int)
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))