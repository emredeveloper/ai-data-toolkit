import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from annoy import AnnoyIndex
import time

# 1. Adım: Gerçek bir veri seti yükleyelim
# MNIST veri setini kullanacağız - 70,000 adet 28x28=784 boyutlu el yazısı rakam görüntüsü
print("MNIST veri seti yükleniyor...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
data = mnist.data.astype('float32')
labels = mnist.target.astype(int)  # Rakam etiketlerini de alalım (0-9)

# Veri setini küçültelim (daha hızlı test için ilk 10,000 örneği alalım)
data = data[:10000]
labels = labels[:10000]
num_vectors, vector_dim = data.shape

# Veriyi normalize edelim (0-1 arasına getir)
scaler = StandardScaler()
data = scaler.fit_transform(data).astype('float32')

print(f"Veri seti yüklendi: {num_vectors} adet {vector_dim} boyutlu vektör")

# Aratacağımız sorgu vektörünü veri setinden rastgele seçelim
query_idx = np.random.randint(0, num_vectors)
query_vector = data[query_idx:query_idx+1]
query_label = labels[query_idx]
print(f"Sorgu vektörü seçildi: {query_idx}. indeksteki vektör (Rakam: {query_label})")
# Bulmak istediğimiz komşu sayısı
k = 10

# -------------------------------------------------------------------------
# BÖLÜM 1: k-NN ile KESİN ARAMA (Tüm Mavi Kareleri Taramak)
# -------------------------------------------------------------------------
print("--- k-NN (Exact Search) Başladı ---")

# scikit-learn'ün NearestNeighbors modelini kuralım.
# 'brute' algoritması, tüm noktaları kontrol eden kaba kuvvet yöntemidir.
knn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')

# Modeli veri setimiz ile eğitelim (aslında sadece veriyi hafızaya alır)
start_time = time.time()
knn.fit(data)
fit_time = time.time() - start_time

# En yakın komşuları bulalım
start_time = time.time()
distances_knn, indices_knn = knn.kneighbors(query_vector)
search_time_knn = time.time() - start_time

print(f"k-NN: Veriyi hazırlama süresi: {fit_time:.6f} saniye")
print(f"k-NN: Arama süresi: {search_time_knn:.6f} saniye")
print(f"k-NN ile bulunan en yakın {k} komşunun indeksleri:\n{indices_knn[0]}\n")


# -------------------------------------------------------------------------
# BÖLÜM 2: ANN ile YAKLAŞIK ARAMA (Sadece Turuncu Kareleri Taramak)
# -------------------------------------------------------------------------
print("\n--- ANN (Approximate Search) Başladı ---")

# Annoy index'ini oluşturalım.
# 'euclidean' mesafe metriğini kullanıyoruz.
ann = AnnoyIndex(vector_dim, 'euclidean')

# Veri setindeki her bir vektörü index'e ekleyelim
for i in range(num_vectors):
    ann.add_item(i, data[i])

# Index'i "inşa edelim". Bu adımda Annoy, veriyi hızlı arama için
# ağaç yapılarında organize eder. Ağaç sayısı ne kadar fazlaysa
# doğruluk o kadar artar ama index oluşturma süresi uzar.
start_time = time.time()
ann.build(10) # 10 adet ağaç oluştur
build_time_ann = time.time() - start_time

# En yakın komşuları arayalım
start_time = time.time()
indices_ann, distances_ann = ann.get_nns_by_vector(query_vector[0], k, include_distances=True)
search_time_ann = time.time() - start_time

# Annoy sonuçlarını sıralı bir şekilde almak için
ann_results = sorted(zip(indices_ann, distances_ann), key=lambda x: x[1])
indices_ann_sorted = [x[0] for x in ann_results]


print(f"ANN: Index oluşturma süresi: {build_time_ann:.6f} saniye")
print(f"ANN: Arama süresi: {search_time_ann:.6f} saniye")
print(f"ANN ile bulunan en yakın {k} komşunun indeksleri:\n{indices_ann_sorted}\n")


# -------------------------------------------------------------------------
# SONUÇLARI KARŞILAŞTIRMA
# -------------------------------------------------------------------------
print("\n--- Karşılaştırma ---")
print(f"k-NN Arama Süresi: {search_time_knn:.6f} saniye")
print(f"ANN Arama Süresi:  {search_time_ann:.6f} saniye (Yaklaşık {(search_time_knn/search_time_ann):.1f} kat daha hızlı!)")

# İki listenin ne kadar benzediğini bulalım (kesişim)
common_neighbors = set(indices_knn[0]).intersection(set(indices_ann_sorted))
accuracy = len(common_neighbors) / k
print(f"Doğruluk (Recall): {accuracy * 100:.1f}% ({k} sonuçtan {len(common_neighbors)} tanesi aynı)")

# MNIST verisinde sorgu vektörü ve bulunan komşuları gösterelim
print(f"\nSorgu vektörü (indeks {query_idx}) - Rakam: {query_label}")
print("k-NN ile bulunan en yakın komşular:")
for i, (idx, dist) in enumerate(zip(indices_knn[0], distances_knn[0])):
    neighbor_label = labels[idx]
    match_status = "✓" if neighbor_label == query_label else "✗"
    print(f"  {i+1}. sıra: İndeks {idx}, Rakam: {neighbor_label} {match_status}, Mesafe: {dist:.4f}")

print("\nANN ile bulunan en yakın komşular:")
for i, (idx, dist) in enumerate(zip(indices_ann_sorted, [x[1] for x in ann_results])):
    neighbor_label = labels[idx]
    match_status = "✓" if neighbor_label == query_label else "✗"
    print(f"  {i+1}. sıra: İndeks {idx}, Rakam: {neighbor_label} {match_status}, Mesafe: {dist:.4f}")

# Etiket bazında doğruluk analizi
knn_correct = sum(1 for idx in indices_knn[0] if labels[idx] == query_label)
ann_correct = sum(1 for idx in indices_ann_sorted if labels[idx] == query_label)

print(f"\n--- ETİKET BAZINDA DOĞRULUK ANALİZİ ---")
print(f"Sorgu rakamı: {query_label}")
print(f"k-NN: {k} komşudan {knn_correct} tanesi aynı rakam ({knn_correct/k*100:.1f}%)")
print(f"ANN:  {k} komşudan {ann_correct} tanesi aynı rakam ({ann_correct/k*100:.1f}%)")

# Performans özeti
print(f"\n--- PERFORMANS ÖZETİ ---")
print(f"Veri Seti: MNIST ({num_vectors} adet {vector_dim} boyutlu vektör)")
print(f"Sorgu: Rakam {query_label}")
print(f"k-NN Toplam Süre: {(fit_time + search_time_knn):.6f} saniye")
print(f"ANN Toplam Süre:  {(build_time_ann + search_time_ann):.6f} saniye")
print(f"Hız Avantajı: {((fit_time + search_time_knn)/(build_time_ann + search_time_ann)):.1f}x daha hızlı")
print(f"İndeks Bazında Doğruluk: {accuracy * 100:.1f}% (Aynı indeksler)")
print(f"Etiket Bazında k-NN Doğruluk: {knn_correct/k*100:.1f}% (Aynı rakamlar)")
print(f"Etiket Bazında ANN Doğruluk: {ann_correct/k*100:.1f}% (Aynı rakamlar)")