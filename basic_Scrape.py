import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import warnings
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from rich.panel import Panel
from rich.progress import track
from rich.box import ROUNDED

# Uyarıları kapat
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=SyntaxWarning)

# Örnek bir Wikipedia sayfası (Dünyanın en büyük şirketleri)
URL = "https://en.wikipedia.org/wiki/List_of_largest_companies_by_revenue"

# Çıktı klasörü oluştur
output_dir = "analiz_sonuclari"
os.makedirs(output_dir, exist_ok=True)

def para_formatindan_sayiya(deger):
    """
    Farklı formatlardaki sayısal değerleri float'a çevirir
    Örnek: "$1,234.56" -> 1234.56, "1.234,56" -> 1234.56
    """
    if pd.isna(deger) or deger == '':
        return np.nan
        
    if isinstance(deger, (int, float)):
        return float(deger)
        
    # Stringe çevir
    deger = str(deger).strip()
    
    # Milyar (B) ve Milyon (M) işaretlerini kontrol et
    carpim = 1
    if 'B' in deger:
        carpim = 1_000_000_000
        deger = deger.replace('B', '')
    elif 'M' in deger:
        carpim = 1_000_000
        deger = deger.replace('M', '')
    
    # Para birimi işaretlerini ve gereksiz karakterleri temizle
    deger = re.sub(r'[^\d.,-]', '', deger)
    
    # Binlik ayraçlarını kaldır
    if ',' in deger and '.' in deger:
        # 1,234.56 formatındaysa
        if deger.find(',') < deger.find('.'):
            deger = deger.replace(',', '')
        # 1.234,56 formatındaysa
        else:
            deger = deger.replace('.', '').replace(',', '.')
    elif ',' in deger:
        # 1,234 veya 1,23 formatındaysa
        deger = deger.replace(',', '.')
    
    try:
        sonuc = float(deger) * carpim
        return sonuc if not np.isnan(sonuc) else np.nan
    except (ValueError, TypeError):
        return np.nan

def veriyi_temizle(df):
    """
    Veri çerçevesini temizler ve düzenler
    """
    console = Console()
    
    with console.status("[cyan]Sütunlar temizleniyor...") as status:
        # Sütun isimlerini düzenle
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # Sayısal olması muhtemel sütunlar için desenler
        para_birimleri = ['$', '€', '£', '¥', '₹', '₺']
        para_birimi_deseni = '|'.join(map(re.escape, para_birimleri))
        
        for col in track(df.columns, description="[cyan]Sütunlar işleniyor..."):
            status.update(f"[cyan]{col} sütunu işleniyor...")
            
            # Sütunun ilk 5 değerini kontrol et
            ornek_veriler = df[col].dropna().head().astype(str).str.strip()
            
            if len(ornek_veriler) == 0:
                continue
                
            # Sayısal veri içeriyor mu kontrol et
            sayisal_mi = any(ornek_veriler.str.contains(r'[0-9]', regex=True)) and any(
                ornek_veriler.str.contains(r'[0-9.,]', regex=True)
            )
            
            # Para birimi içeriyor mu kontrol et
            para_birimi_var = any(ornek_veriler.str.contains(para_birimi_deseni, regex=True))
            
            if sayisal_mi or para_birimi_var:
                try:
                    # Önce temizleme işlemi uygula
                    df[col] = df[col].apply(para_formatindan_sayiya)
                    
                    # Eğer tüm değerler NaN değilse, float'a çevir
                    if not df[col].isna().all():
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    console.print(f"[green]✓[/green] {col} sütunu sayısala dönüştürüldü")
                except Exception as e:
                    console.print(f"[yellow]Uyarı:[/yellow] {col} sütunu dönüştürülemedi: {str(e)}")
    
    console.print("[bold green]✓ Veri temizleme tamamlandı![/]")
    return df

def temel_analiz_yap(df, tablo_adi=''):
    """
    Rich kütüphanesi kullanarak temel veri analizi yapar
    """
    console = Console()
    
    # Başlık paneli
    console.print(Panel.fit(
        f"[bold blue]{tablo_adi} - Temel Analiz",
        border_style="blue",
        padding=(1, 2)
    ))
    
    # Genel bilgiler tablosu
    table = Table(show_header=True, header_style="bold magenta", box=ROUNDED)
    table.add_column("Özellik", style="cyan")
    table.add_column("Değer", justify="right")
    
    table.add_row("Toplam Kayıt Sayısı", f"[green]{df.shape[0]}")
    table.add_row("Toplam Sütun Sayısı", f"[green]{df.shape[1]}")
    table.add_row("Toplam Eksik Değer", f"[red]{df.isnull().sum().sum()}")
    
    console.print(table)
    
    # Sütun tipleri tablosu
    type_table = Table(title="Sütun Tipleri", show_header=True, header_style="bold magenta", box=ROUNDED)
    type_table.add_column("Sütun Adı", style="cyan")
    type_table.add_column("Veri Tipi", justify="right")
    
    for col, dtype in df.dtypes.items():
        type_table.add_row(col, str(dtype))
    
    console.print(type_table)
    
    # Eksik değerler tablosu
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_table = Table(title="Eksik Değerler", show_header=True, header_style="bold magenta", box=ROUNDED)
        missing_table.add_column("Sütun", style="cyan")
        missing_table.add_column("Eksik Değer Sayısı", justify="right")
        missing_table.add_column("Yüzde", justify="right")
        
        for col, count in missing.items():
            if count > 0:
                percent = (count / len(df)) * 100
                missing_table.add_row(
                    col, 
                    f"[red]{count}", 
                    f"[yellow]{percent:.1f}%"
                )
        
        console.print(missing_table)
    
    # Sayısal sütunlar için istatistikler
    sayisal_kolonlar = df.select_dtypes(include=[np.number]).columns
    if len(sayisal_kolonlar) > 0:
        stats = df[sayisal_kolonlar].describe()
        
        for col in stats.columns:
            stat_table = Table(title=f"{col} İstatistikleri", show_header=True, header_style="bold green", box=ROUNDED)
            stat_table.add_column("İstatistik", style="cyan")
            stat_table.add_column("Değer", justify="right")
            
            for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                if stat in stats[col]:
                    value = stats[col][stat]
                    if isinstance(value, (int, float)):
                        value = f"{value:,.2f}" if stat != 'count' else f"{int(value):,}"
                    stat_table.add_row(stat, str(value))
            
            console.print(stat_table)
    
    return sayisal_kolonlar

def main():
    console = Console()
    try:
        with console.status("[bold green]Tablolar çekiliyor...") as status:
            console.print(f"[cyan]Hedef:[/cyan] {URL}")
            tables = pd.read_html(URL, match='Rank')
            
            if not tables:
                console.print("[red]Hata:[/red] Hiç tablo bulunamadı.")
                return
                
            console.print(f"[green]✓[/green] Toplam {len(tables)} tablo bulundu.")
            
            # Sadece ilk tabloyu işle (ana tablo)
            ana_tablo = tables[0].copy()
            
            # Veriyi temizle
            console.print("\n[bold]Veri temizleniyor...[/]")
            temiz_veri = veriyi_temizle(ana_tablo)
            
            # Tarih bilgisi ile dosya adı oluştur
            tarih = datetime.now().strftime("%Y%m%d_%H%M%S")
            cikti_dosyasi = os.path.join(output_dir, f"sirket_analiz_{tarih}.xlsx")
            
            # Excel'e yazma işlemi
            console.print("\n[bold]Excel dosyası oluşturuluyor...[/]")
            try:
                # Excel dosyasına yaz
                with pd.ExcelWriter(cikti_dosyasi, engine='openpyxl') as writer:
                    # Ham veri
                    ana_tablo.to_excel(writer, sheet_name='Ham_Veri', index=False)
                    
                    # Temizlenmiş veri
                    temiz_veri.to_excel(writer, sheet_name='Temiz_Veri', index=False)
                    
                    # Sayısal sütunlar için analiz
                    sayisal_kolonlar = [col for col in temiz_veri.select_dtypes(include=[np.number]).columns 
                                      if temiz_veri[col].notna().any()]
                    
                    if len(sayisal_kolonlar) > 1:  # En az iki sayısal sütun olmalı
                        # Korelasyon matrisi
                        korelasyon = temiz_veri[sayisal_kolonlar].corr()
                        korelasyon.to_excel(writer, sheet_name='Korelasyon')
                        
                        # En yüksek korelasyonlar
                        korelasyon_ust_ucgen = korelasyon.where(
                            np.triu(np.ones(korelasyon.shape), k=1).astype(bool)
                        )
                        en_yuksek_korelasyon = korelasyon_ust_ucgen.unstack().sort_values(ascending=False).dropna()
                        
                        # En yüksek korelasyonları DataFrame'e çevir
                        en_yuksek_df = pd.DataFrame({
                            'Değişken 1': en_yuksek_korelasyon.index.get_level_values(0),
                            'Değişken 2': en_yuksek_korelasyon.index.get_level_values(1),
                            'Korelasyon': en_yuksek_korelasyon.values
                        })
                        
                        en_yuksek_df.head(10).to_excel(
                            writer, 
                            sheet_name='En_Yuksek_Korelasyon', 
                            index=False
                        )
                        
                        console.print("\n[green]✓[/green] Korelasyon analizi tamamlandı.")
                    else:
                        console.print("\n[yellow]Uyarı:[/yellow] Korelasyon analizi için yeterli sayısal sütun bulunamadı.")
                    
                    # Temel analiz yap
                    console.print("\n[bold]Temel analiz yapılıyor...[/]")
                    sayisal_kolonlar = temel_analiz_yap(temiz_veri, "Temizlenmiş Veri")
                    
                    # Örnek veri gösterimi
                    console.print("\n[bold cyan]İlk 5 Kayıt:[/]")
                    console.print(temiz_veri.head().to_string())
                    
                    console.print(f"\n[bold green]✓[/bold green] Analiz sonuçları başarıyla kaydedildi: [cyan]{cikti_dosyasi}[/]")
                    
            except Exception as e:
                console.print(f"\n[red]Hata:[/red] Excel yazma hatası: {str(e)}")
                # Excel yazılamazsa CSV olarak kaydet
                try:
                    csv_dosyasi = cikti_dosyasi.replace('.xlsx', '.csv')
                    temiz_veri.to_csv(csv_dosyasi, index=False, encoding='utf-8-sig')
                    console.print(f"[green]✓[/green] Veri CSV olarak kaydedildi: [cyan]{csv_dosyasi}[/]")
                except Exception as e2:
                    console.print(f"[red]Hata:[/red] CSV kaydetme hatası: {str(e2)}")
    
    except Exception as e:
        console.print(f"\n[red]Beklenmeyen bir hata oluştu:[/red] {str(e)}")
        import traceback
        console.print(traceback.format_exc(), style="red")
        
        # Web sayfasındaki tüm tabloları al
        tables = pd.read_html(URL, match='Rank')
        
        if not tables:
            print("Hiç tablo bulunamadı.")
            return
            
        print(f"\nToplam {len(tables)} tablo bulundu.")
        
        # Sadece ilk tabloyu işle (ana tablo)
        ana_tablo = tables[0].copy()
        
        # Veriyi temizle
        print("\nVeri temizleniyor...")
        temiz_veri = veriyi_temizle(ana_tablo)
        
        # Tarih bilgisi ile dosya adı oluştur
        tarih = datetime.now().strftime("%Y%m%d_%H%M%S")
        cikti_dosyasi = os.path.join(output_dir, f"sirket_analiz_{tarih}.xlsx")
        
        # Excel'e yazma işlemi
        print("\nExcel dosyası oluşturuluyor...")
        try:
            # Excel dosyasına yaz
            with pd.ExcelWriter(cikti_dosyasi, engine='openpyxl') as writer:
                # Ham veri
                ana_tablo.to_excel(writer, sheet_name='Ham_Veri', index=False)
                
                # Temizlenmiş veri
                temiz_veri.to_excel(writer, sheet_name='Temiz_Veri', index=False)
                
                # Sayısal sütunlar için analiz
                sayisal_kolonlar = [col for col in temiz_veri.select_dtypes(include=[np.number]).columns 
                                  if temiz_veri[col].notna().any()]
                
                if len(sayisal_kolonlar) > 1:  # En az iki sayısal sütun olmalı
                    # Korelasyon matrisi
                    korelasyon = temiz_veri[sayisal_kolonlar].corr()
                    korelasyon.to_excel(writer, sheet_name='Korelasyon')
                    
                    # En yüksek korelasyonlar
                    korelasyon_ust_ucgen = korelasyon.where(
                        np.triu(np.ones(korelasyon.shape), k=1).astype(bool)
                    )
                    en_yuksek_korelasyon = korelasyon_ust_ucgen.unstack().sort_values(ascending=False).dropna()
                    
                    # En yüksek korelasyonları DataFrame'e çevir
                    en_yuksek_df = pd.DataFrame({
                        'Değişken 1': en_yuksek_korelasyon.index.get_level_values(0),
                        'Değişken 2': en_yuksek_korelasyon.index.get_level_values(1),
                        'Korelasyon': en_yuksek_korelasyon.values
                    })
                    
                    en_yuksek_df.head(10).to_excel(
                        writer, 
                        sheet_name='En_Yuksek_Korelasyon', 
                        index=False
                    )
                    
                    print("\nKorelasyon analizi tamamlandı.")
                else:
                    print("\nUyarı: Korelasyon analizi için yeterli sayısal sütun bulunamadı.")
                
                # Temel analiz yap
                print("\nTemel analiz yapılıyor...")
                sayisal_kolonlar = temel_analiz_yap(temiz_veri, "Temizlenmiş Veri")
                
                # Örnek veri gösterimi
                print("\nİlk 5 Kayıt:")
                print(temiz_veri.head().to_string())
                
                print(f"\n\nAnaliz sonuçları başarıyla kaydedildi: {cikti_dosyasi}")
                
        except Exception as e:
            print(f"\nExcel yazma hatası: {str(e)}")
            # Excel yazılamazsa CSV olarak kaydet
            try:
                csv_dosyasi = cikti_dosyasi.replace('.xlsx', '.csv')
                temiz_veri.to_csv(csv_dosyasi, index=False, encoding='utf-8-sig')
                print(f"\nVeri CSV olarak kaydedildi: {csv_dosyasi}")
            except Exception as e2:
                print(f"\nCSV kaydetme hatası: {str(e2)}")
    
    except Exception as e:
        print(f"\nBeklenmeyen bir hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()