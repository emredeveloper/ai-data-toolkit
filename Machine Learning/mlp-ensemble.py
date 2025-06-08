from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# MNIST veri setini indir (bu işlem ilk seferde uzun sürebilir)
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Daha hızlı çalışmak için (isteğe bağlı) örnekleri azaltabilirsin, ör: X = X[:20000], y = y[:20000]

# Eğitim ve test verisi olarak ayır (ör: %80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tekli MLP
single_mlp = MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=20, random_state=42, verbose=True)
single_mlp.fit(X_train, y_train)
y_pred_single = single_mlp.predict(X_test)
single_acc = accuracy_score(y_test, y_pred_single)
print(f"Tekli MLP (Test Seti, MNIST) Doğruluk: {single_acc:.4f}")

# Ensemble modeli
mlp1 = MLPClassifier(hidden_layer_sizes=(64,), max_iter=20, random_state=1, verbose=False)
mlp2 = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, random_state=2, verbose=False)
mlp3 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=20, random_state=3, verbose=False)
ensemble = VotingClassifier(estimators=[
    ('mlp1', mlp1),
    ('mlp2', mlp2),
    ('mlp3', mlp3)
], voting='soft')
ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)
ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
print(f"MLP Ensemble (Test Seti, MNIST) Doğruluk: {ensemble_acc:.4f}")

if ensemble_acc > single_acc:
    print("Ensemble modeli daha iyi sonuç verdi.")
elif ensemble_acc < single_acc:
    print("Tekli MLP modeli daha iyi sonuç verdi.")
else:
    print("Her iki modelin doğruluğu eşit.")