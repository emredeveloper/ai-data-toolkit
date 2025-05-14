# Web Summarizer Chrome Extension

Web Summarizer, ziyaret ettiğiniz web sayfalarının içeriğini tek tıkla özetlemenizi sağlayan, modern ve kullanıcı dostu bir Chrome uzantısıdır. Özetleme işlemi, bilgisayarınızda çalışan bir yerel LLM (Large Language Model) sunucusu üzerinden yapılır. Eklenti, hızlı ve gizliliğe duyarlı bir şekilde özet sunar.

## Özellikler

- **Tek Tıkla Özet:** Herhangi bir web sayfasının içeriğini kolayca özetleyin.
- **Modern ve Şık Arayüz:** Kart tasarımı, logo, renkli bildirimler, yükleniyor animasyonu ve kopyala butonu.
- **Satır Sonu ve Format Koruma:** Model çıktısı, satır sonları ve boşluklarıyla okunaklı biçimde gösterilir.
- **Kopyala Butonu:** Özet kutusundaki metni panoya hızlıca kopyalayın.
- **Başarılı/Hatalı Bildirimler:** İşlemlerin durumu için renkli uyarılar.
- **Yükleniyor Animasyonu:** Özet çıkarılırken kullanıcıya görsel bildirim.
- **Tamamen Yerel:** Verileriniz hiçbir şekilde dışarı çıkmaz, özetleme işlemi tamamen kendi bilgisayarınızda gerçekleşir.

---

## Klasör Yapısı

```
Qwen-Agent/
├── backend/                # Yerel LLM sunucu kodları (FastAPI, Python)
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
│
└── extension/              # Chrome uzantısı dosyaları
    ├── manifest.json
    ├── popup.html
    ├── popup.js
    ├── background.js
    ├── content.js
    └── icon.png            # (Varsa) Eklenti logosu
```

---

## Kurulum ve Kullanım

### 1. Backend (Yerel Sunucu) Kurulumu

1. `backend` klasöründe gerekli Python bağımlılıklarını yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
2. Sunucuyu başlatın:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 7864
   ```
   Sunucu başarıyla başlatıldığında, özetleme API’si aktif olur.

### 2. Chrome Uzantısı Kurulumu

1. Chrome’da `chrome://extensions` adresine gidin.
2. Sağ üstten "Geliştirici Modu"nu açın.
3. "Paketlenmemiş uzantı yükle" butonuna tıklayın ve `extension` klasörünü seçin.
4. Uzantı simgesine tıklayarak popup arayüzünü açın.

### 3. Kullanım

- Herhangi bir web sayfasında uzantı simgesine tıklayın.
- "Summarize" butonuna basın.
- Modelin cevabı yükleniyor animasyonu ile birlikte akış halinde ekrana gelir.
- İsterseniz özet kutusunun sağ üstündeki kopyala butonuna tıklayarak sonucu panoya alabilirsiniz.

---

## Teknik Detaylar

- **Backend:** FastAPI ile yazılmıştır. `/summarize_stream_status` endpoint’i, POST edilen metni özetleyip streaming olarak geri döner.
- **Frontend:** Manifest V3 ile uyumlu, modern HTML/CSS/JS ile hazırlanmış popup arayüzü.
- **Arayüz:** Responsive, kart tasarımlı, bildirimli ve kullanıcı dostu.
- **Güvenlik:** Tüm işlemler lokal makinede gerçekleşir, veri dışarı çıkmaz.

---

## Özelleştirme ve Geliştirme

- Arayüzdeki renkler, ikon veya yazı tipleri kolayca değiştirilebilir.
- Backend tarafında farklı modeller veya özetleme algoritmaları entegre edilebilir.
- Uzantıya yeni özellikler (ör. farklı dillerde özet, özet uzunluğu ayarı) eklenebilir.

---

## Sıkça Sorulan Sorular (SSS)

**Q: Model internetten mi çalışıyor?**  
Hayır, özetleme tamamen kendi bilgisayarınızda çalışan LLM ile yapılır.

**Q: Uzantı hangi sayfalarda çalışır?**  
Tüm web sayfalarında çalışacak şekilde tasarlanmıştır.

**Q: Gizlilik konusunda endişem olmalı mı?**  
Hayır, hiçbir veri dışarıya gönderilmez.

---

## Lisans

MIT Lisansı

---

Her türlü öneri, katkı veya hata bildirimi için issue/pull request açabilirsiniz. İyi özetlemeler!
