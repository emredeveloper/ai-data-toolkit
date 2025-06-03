import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

# GPU kontrolü
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan cihaz: {device}")

# 1. Veri Hazırlama
# Daha büyük örnek veri seti (gerçek projede CSV'den okuyacaksınız)
sample_data = {
    'text': [
        # Pozitif yorumlar (label: 2)
        "Bu ürün harika! Çok memnun kaldım, herkese tavsiye ederim.",
        "Mükemmel kalite, hızlı kargo. 5 yıldız hak ediyor.",
        "Paramın karşılığını aldım, memnunum genel olarak.",
        "Süper hızlı teslimat, ürün de gayet güzel geldi.",
        "Çok kaliteli bir ürün, tekrar alırım kesinlikle.",
        "Harika bir deneyimdi, çok beğendim bu ürünü.",
        "Mükemmel ambalaj ve hızlı kargo, ürün de süper.",
        "Bu fiyata bu kalite çok iyi, çok memnunum.",
        "Aldığıma çok sevindim, harika bir alışveriş oldu.",
        "Perfect! Tam istediğim gibi, çok güzel ürün.",
        
        # Negatif yorumlar (label: 0)
        "Kalitesi berbat, paramın hakkını alamadım. Asla almam.",
        "Çok kötü bir deneyimdi, müşteri hizmetleri de ilgisiz.",
        "Berbat kalite, çöpe attım resmen. Kesinlikle almayın!",
        "Hiç beğenmedim, resimde göründüğü gibi değil.",
        "Para kaybı, çok kötü kalite. Tavsiye etmem.",
        "Aldığıma çok pişman oldum, hiç iyi değil.",
        "Kargo çok geç geldi, ürün de bozuk geldi.",
        "Çok pahalı ve kalitesi de kötü, almayın.",
        "Tam bir hayal kırıklığı, hiç memnun kalmadım.",
        "İade ettim, çok kötü kaliteydi gerçekten.",
        
        # Nötr yorumlar (label: 1)
        "Ürün fena değil ama beklediğim kadar iyi de değil.",
        "Fiyatına göre idare eder, çok beklenti yapmamak lazım.",
        "Ortalama bir ürün, ne çok iyi ne çok kötü.",
        "Normal kalite, beklentimi karşıladı sayılır.",
        "İdare eder, çok şey beklememek lazım bu fiyata.",
        "Orta kalite, daha iyisi olabilirdi ama fena değil.",
        "Beklediğim kadar iyi değildi ama kötü de sayılmaz.",
        "Sıradan bir ürün, özel bir özelliği yok.",
        "Fiyat performans açısından normal seviyede.",
        "Ne çok memnunum ne de çok şikayetçiyim."
    ],
    'label': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  # Pozitif
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Negatif  
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Nötr
}

# DataFrame oluşturma
df = pd.DataFrame(sample_data)
print("Veri dağılımı:")
print(df['label'].value_counts())

# 2. Model ve Tokenizer Yükleme
model_name = "dbmdz/bert-base-turkish-cased"  # Türkçe BERT modeli
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=3,  # 3 sınıf: negatif, nötr, pozitif
    id2label={0: "Negatif", 1: "Nötr", 2: "Pozitif"},
    label2id={"Negatif": 0, "Nötr": 1, "Pozitif": 2}
)

# 3. Dataset Sınıfı
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 4. Veriyi Bölme
# Test size'ı artırıyoruz çünkü daha fazla veri var
X_train, X_test, y_train, y_test = train_test_split(
    df['text'].tolist(), 
    df['label'].tolist(), 
    test_size=0.3,  # %30'unu test için ayırıyoruz
    random_state=42,
    stratify=df['label']
)

print(f"Eğitim veri sayısı: {len(X_train)}")
print(f"Test veri sayısı: {len(X_test)}")
print("Eğitim veri dağılımı:", pd.Series(y_train).value_counts().sort_index())
print("Test veri dağılımı:", pd.Series(y_test).value_counts().sort_index())

# Dataset objelerini oluşturma
train_dataset = ReviewDataset(X_train, y_train, tokenizer)
test_dataset = ReviewDataset(X_test, y_test, tokenizer)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 5. Eğitim Parametreleri
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,  # Küçük veri seti için epoch sayısını azaltıyoruz
    per_device_train_batch_size=4,  # Batch size'ı küçültüyoruz
    per_device_eval_batch_size=4,
    warmup_steps=10,  # Warmup steps'i azaltıyoruz
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=5,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_accuracy',
    greater_is_better=True,
    report_to=None,  # WandB entegrasyonunu kapatır
    dataloader_drop_last=False,  # Son batch'i düşürme
)

# 6. Değerlendirme Fonksiyonu
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

# 7. Trainer Oluşturma
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 8. Fine-tuning Başlatma
print("Fine-tuning başlıyor...")
trainer.train()

# 9. Model Değerlendirme
print("\nModel değerlendiriliyor...")
eval_results = trainer.evaluate()
print(f"Test Accuracy: {eval_results['eval_accuracy']:.4f}")

# 10. Modeli Kaydetme
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
print("Model kaydedildi: ./fine_tuned_model")

# 11. Tahmin Fonksiyonu
def predict_sentiment(text, model, tokenizer, device='cpu'):
    model.eval()
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in encoding.items()}
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    labels = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}
    return labels[predicted_class], confidence

# 12. Test Tahminleri
test_texts = [
    "Bu ürün gerçekten çok güzel, çok memnunum!",
    "Hiç beğenmedim, çok kötü kalite.",
    "Fena değil ama harika da sayılmaz."
]

print("\n=== ÖRNEK TAHMİNLER ===")
for text in test_texts:
    sentiment, confidence = predict_sentiment(text, model, tokenizer, device)
    print(f"Metin: {text}")
    print(f"Tahmin: {sentiment} (Güven: {confidence:.2f})")
    print("-" * 50)

# 13. Detaylı Performans Raporu
print("\n=== DETAYLI PERFORMANS RAPORU ===")
# Test seti üzerinde tahminler
test_predictions = []
test_labels = []

for i in range(len(test_dataset)):
    sample = test_dataset[i]
    text = X_test[i]
    true_label = y_test[i]
    
    pred_sentiment, confidence = predict_sentiment(text, model, tokenizer, device)
    pred_label = {"Negatif": 0, "Nötr": 1, "Pozitif": 2}[pred_sentiment]
    
    test_predictions.append(pred_label)
    test_labels.append(true_label)

# Classification report
label_names = ["Negatif", "Nötr", "Pozitif"]
print(classification_report(test_labels, test_predictions, target_names=label_names))

print(f"\nFine-tuning tamamlandı!")
print(f"Model dosyaları './fine_tuned_model' klasöründe kaydedildi.")
print(f"Bu modeli daha sonra şu şekilde yükleyebilirsiniz:")
print("model = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_model')")
print("tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')")