import requests
import json
import os
from rapidfuzz import fuzz
import time
from collections import OrderedDict

OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Ollama sunucunuzun adresi
CACHE_FILE = "cache.json"
CACHE_LIMIT = 100  # Maksimum cache kaydı
STATS = {"hit": 0, "miss": 0}

# LRU cache için OrderedDict kullan
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        raw_cache = json.load(f)
        # Eski cache anahtarlarını yeni formata dönüştür
        cache = OrderedDict()
        for key, value in raw_cache.items():
            if "|" in key:
                cache[key] = value
            else:
                # Varsayılan model ile eski anahtarı birleştir
                cache[f"granite3.3:8b|{key}"] = value
        # Eğer dönüştürme olduysa cache dosyasını güncelle
        if len(raw_cache) != len(cache):
            with open(CACHE_FILE, "w", encoding="utf-8") as f2:
                json.dump(cache, f2, ensure_ascii=False, indent=2)
else:
    cache = OrderedDict()

def normalize_prompt(prompt):
    return prompt.strip().lower()

def jaccard_similarity(a, b):
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def find_fuzzy_match(norm_prompt, model):
    for key in cache.keys():
        k_model, k_prompt = key.split("|", 1)
        if k_model == model:
            fuzzy_score = fuzz.ratio(norm_prompt, k_prompt)
            jaccard_score = jaccard_similarity(norm_prompt, k_prompt) * 100
            if fuzzy_score >= 80 or jaccard_score >= 70:
                return key
    return None

def ollama_generate(prompt, model="granite3.3:8b"):
    norm_prompt = normalize_prompt(prompt)
    cache_key = f"{model}|{norm_prompt}"
    # Fuzzy match
    match_key = find_fuzzy_match(norm_prompt, model)
    start = time.time()
    if match_key:
        STATS["hit"] += 1
        print(f"Önbellekten yanıt dönülüyor. (Fuzzy match: {match_key})")
        result = cache[match_key]
        cache.move_to_end(match_key)  # LRU güncelle
        print(f"Yanıt süresi: {time.time()-start:.2f} sn | Hit: {STATS['hit']} Miss: {STATS['miss']}")
        return result
    STATS["miss"] += 1
    payload = {
        "model": model,
        "prompt": prompt
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=60)
        response.raise_for_status()
        result = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    result += data["response"]
    except Exception as e:
        print(f"[HATA] Ollama API hatası: {e}")
        return "[HATA] Ollama API hatası: " + str(e)
    # LRU: Sınırı aşarsa en eskiyi sil
    if len(cache) >= CACHE_LIMIT:
        cache.popitem(last=False)
    cache[cache_key] = result
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"Yanıt süresi: {time.time()-start:.2f} sn | Hit: {STATS['hit']} Miss: {STATS['miss']}")
    return result

if __name__ == "__main__":
    while True:
        prompt = input("Prompt girin (çıkmak için 'q'): ")
        if prompt.lower() == 'q':
            break
        yanit = ollama_generate(prompt)
        print("Yanıt:", yanit)