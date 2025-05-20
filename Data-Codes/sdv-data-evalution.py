from sdv.datasets.demo import download_demo
from sdv.evaluation.single_table import evaluate_quality
from sdv.single_table import GaussianCopulaSynthesizer

def main():
    # Veri yükle
    print("Veri yükleniyor...")
    real_data, metadata = download_demo(modality='single_table', dataset_name='fake_hotel_guests')
    
    # Sentetik veri oluştur
    print("Sentetik veri oluşturuluyor...")
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(real_data)
    synthetic_data = synthesizer.sample(num_rows=len(real_data))
    
    # Bilgileri göster
    print("\n=== VERİ BİLGİLERİ ===")
    print(f"Gerçek Veri Boyutu: {real_data.shape}")
    print(f"Sentetik Veri Boyutu: {synthetic_data.shape}")
    print("\nSütunlar:", ", ".join(real_data.columns))
    
    # Veri kalitesini değerlendir
    print("\n=== KALİTE DEĞERLENDİRMESİ ===")
    quality_report = evaluate_quality(
        real_data,
        synthetic_data,
        metadata,
        verbose=True
    )
    
    print("\nİşlem tamamlandı!")

if __name__ == "__main__":
    main()