import gradio as gr
import collections

def text_analyzer(text, analysis_type):
    """Metni çeşitli yöntemlerle analiz eder.
    
    Args:
        text: Analiz edilecek metin
        analysis_type: Analiz türü (kelime sayısı, harf sayısı, en sık harf)
        
    Returns:
        Analiz sonucu (sayı veya metin)
    """
    text = text.lower()
    
    if analysis_type == "kelime sayısı":
        return len(text.split())
    
    elif analysis_type == "harf sayısı":
        return sum(c.isalpha() for c in text)
    
    elif analysis_type == "en sık harf":
        # Sadece harfleri filtrele
        only_letters = [c for c in text if c.isalpha()]
        if not only_letters:
            return "Metinde harf bulunamadı."
        
        # En sık tekrar eden harfi bul
        letter_counts = collections.Counter(only_letters)
        most_common_letter, count = letter_counts.most_common(1)[0]
        return f"'{most_common_letter}' ({count} kez)"
    
    elif analysis_type == "karakter dağılımı":
        # Metindeki tüm karakterlerin dağılımını hesapla
        char_counts = collections.Counter(text)
        # En yaygın 5 karakteri al
        common_chars = char_counts.most_common(5)
        result = ", ".join([f"'{char}': {count}" for char, count in common_chars])
        return result
    
    return "Geçersiz analiz türü"

demo = gr.Interface(
    fn=text_analyzer,
    inputs=[
        gr.Textbox(lines=5, placeholder="Analiz edilecek metni buraya girin..."),
        gr.Radio(["kelime sayısı", "harf sayısı", "en sık harf", "karakter dağılımı"], 
                label="Analiz Türü")
    ],
    outputs="text",
    title="Metin Analiz Aracı",
    description="Metninizi çeşitli yöntemlerle analiz edin"
)

demo.launch(mcp_server=True)