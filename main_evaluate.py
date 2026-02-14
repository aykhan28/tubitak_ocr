import json
import sys
import os
import subprocess
from difflib import SequenceMatcher

def normalize_ocr_text(text: str):
    """OCR hatalarını düzelt: Türkçede olmayan karakterleri benzer Türkçe karakterlere çevir"""
    if not text:
        return ""
    
    # Karakter dönüşümleri
    char_map = {
        'ä': 'ö', 'à': 'a', 'á': 'a', 'â': 'ö', 'ã': 'ö', 'å': 'a', 'ā': 'ö', 'ă': 'ö', 'ą': 'a',
        'ë': 'e', 'è': 'e', 'é': 'e', 'ê': 'e', 'ē': 'e', 'ĕ': 'e', 'ę': 'ç', 'ė': 'e',
        'ï': 'i', 'ì': 'i', 'í': 'i', 'î': 'i', 'ĩ': 'i', 'ī': 'i', 'ĭ': 'i', 'į': 'i',
        'ò': 'ö', 'ó': 'ö', 'ô': 'ö', 'õ': 'ö', 'ø': 'o', 'ō': 'ö', 'ŏ': 'ö', 'ő': 'ö',
        'ù': 'ü', 'ú': 'ü', 'û': 'ü', 'ũ': 'ü', 'ū': 'ü', 'ŭ': 'ü', 'ů': 'ü', 'ű': 'ü',
        'ÿ': 'y', 'ý': 'y', 'ŷ': 'g',
        'ć': 'c', 'ĉ': 'c', 'ċ': 'c', 'č': 'c',
        'ġ': 'ğ', 'ģ': 'ğ',
        'ś': 's', 'ŝ': 's','š': 's',
        'ž': 'z', 'ź': 'z', 'ż': 'z',
        'ñ': 'n', 'ń': 'n', 'ņ': 'n', 'ň': 'n', 'ŉ': 'n',
    }
    
    # Karakterleri dönüştür
    normalized = []
    for c in text:
        if c in char_map:
            normalized.append(char_map[c])
        else:
            normalized.append(c)
    
    return ''.join(normalized)

def normalize_text(text: str):
    """Metni normalize et: küçük harf + boşlukları temizle + OCR karakter düzeltmeleri"""
    if not text:
        return ""
    
    # Önce OCR karakterlerini düzelt
    text = normalize_ocr_text(text)
    
    # Sonra normal normalize işlemleri
    text = text.lower()
    text = text.strip()
    text = ' '.join(text.split())  # Birden fazla boşluğu teke indir
    
    return text

def string_similarity(s1: str, s2: str):
    """İki string arasındaki benzerlik (0-100)"""
    s1_norm = normalize_text(s1)
    s2_norm = normalize_text(s2)
    
    if not s1_norm or not s2_norm:
        return 0
    
    # Tam eşleşme
    if s1_norm == s2_norm:
        return 100
    
    # Kelime kelime karşılaştır
    words1 = s1_norm.split()
    words2 = s2_norm.split()
    
    # Kısa cevaplar için karakter benzerliği
    char_similarity = SequenceMatcher(None, s1_norm, s2_norm).ratio() * 100
    
    # Kelime eşleşme oranı
    if words1 and words2:
        matching_words = sum(1 for w in words1 if any(
            SequenceMatcher(None, w, w2).ratio() > 0.75 for w2 in words2
        ))
        word_similarity = (matching_words / max(len(words1), len(words2))) * 100
        
        # İkisinin ortalamasını al
        return (char_similarity + word_similarity) / 2
    
    return char_similarity

def run_ollama(prompt: str, model: str = "gemma3:270m"):
    """Ollama modelini çalıştır"""
    try:
        env = os.environ.copy()
        env['OLLAMA_NUM_GPU'] = '0'
        
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            env=env,
            timeout=30
        )
        return result.stdout.strip()
    except Exception as e:
        return ""

def score_to_points(score: int):
    """Benzerlik skorunu puana çevir"""
    if score >= 90:  # Aynı anlam
        return 1.0
    elif score >= 70:  # Çok yakın
        return 0.75
    elif score >= 50:  # Kısmen doğru
        return 0.5
    elif score >= 30:  # Az benzer
        return 0.25
    else:  # Farklı
        return 0.0

def evaluate_answer(student_answer: str, correct_answer: str):
    """Öğrenci cevabını değerlendir (alternatif cevapları da kontrol et)"""
    
    if not student_answer or student_answer.strip() == "":
        return {"puan_katsayi": 0.0, "durum": "Boş", "yontem": "Boş", "eslesen_cevap": ""}
    
    # Alternatif cevapları ayır (/ ile)
    alternative_answers = [ans.strip() for ans in correct_answer.split('/')]
    
    best_score = 0
    best_method = ""
    best_answer = alternative_answers[0]
    best_str_sim = 0
    best_llm_sim = 0
    
    # Her alternatif için kontrol et
    for alt_answer in alternative_answers:
        # 1. String benzerliği hesapla (OCR düzeltmeli)
        str_similarity = string_similarity(student_answer, alt_answer)
        
        # Yüksek string benzerliği varsa LLM'e gerek yok
        if str_similarity >= 85:
            if str_similarity > best_score:
                best_score = str_similarity
                best_method = "String"
                best_answer = alt_answer
                best_str_sim = str_similarity
                best_llm_sim = 0
            continue
        
        # 2. LLM ile anlam benzerliği kontrol et (OCR düzeltmeli)
        # Öğrenci cevabını da normalize et
        norm_student = normalize_text(student_answer)
        norm_correct = normalize_text(alt_answer)
        
        prompt = f"""İki cevap aynı anlamda mı? OCR hataları olabilir (ä->a, ö->o, ü->u, ß->ss gibi dönüşümler yapıldı).

Doğru (normalize): {norm_correct}
Öğrenci (normalize): {norm_student}

Büyük/küçük harf önemsiz. Yazım hataları tolere et.

Benzerlik puanı ver (0-100):
90-100: Aynı anlam
70-89: Çok yakın
50-69: Kısmen doğru
30-49: Az benzer
0-29: Farklı

Sadece sayı yaz:"""

        response = run_ollama(prompt)
        
        try:
            llm_score = int(''.join(filter(str.isdigit, response[:10])))
            llm_score = max(0, min(100, llm_score))
        except:
            llm_score = 0
        
        # String ve LLM skorunun en yükseğini al
        final_score = max(str_similarity, llm_score)
        
        if final_score > best_score:
            best_score = final_score
            best_method = "String" if str_similarity > llm_score else "LLM"
            best_answer = alt_answer
            best_str_sim = str_similarity
            best_llm_sim = llm_score
    
    # Puan katsayısını hesapla
    puan_katsayi = score_to_points(best_score)
    
    # Durumu belirle
    if puan_katsayi >= 0.9:
        durum = "Tam Doğru"
    elif puan_katsayi >= 0.7:
        durum = "Çok Yakın"
    elif puan_katsayi >= 0.5:
        durum = "Kısmen Doğru"
    elif puan_katsayi >= 0.25:
        durum = "Az Benzer"
    else:
        durum = "Yanlış"
    
    return {
        "puan_katsayi": puan_katsayi,
        "durum": durum,
        "yontem": best_method,
        "eslesen_cevap": best_answer,
        "string_benzerlik": round(best_str_sim, 1),
        "llm_benzerlik": best_llm_sim
    }

def load_json(file_path: str):
    """JSON dosyasını yükle"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    if len(sys.argv) < 3:
        print("Kullanım: python evaluate.py <ocr_sonuc.json> <dogru_cevaplar.json>")
        return
    
    ocr_file = sys.argv[1]
    correct_file = sys.argv[2]
    
    # Dosyaları yükle
    ocr_data = load_json(ocr_file)
    correct_answers = load_json(correct_file)
    
    student_answers = ocr_data.get("answers", {})
    
    # Değerlendirme
    results = {}
    toplam_katsayi = 0
    dogru = 0
    yanlis = 0
    bos = 0
    
    print(f"\n🔍 Değerlendiriliyor (OCR karakter düzeltmeleri aktif)...\n")
    
    for q_num, correct_ans in correct_answers.items():
        student_ans = student_answers.get(str(q_num), "")
        
        eval_result = evaluate_answer(student_ans, correct_ans)
        
        results[q_num] = {
            "ogrenci_cevabi": student_ans,
            "ogrenci_cevabi_normalized": normalize_text(student_ans),
            "dogru_cevap": correct_ans,
            **eval_result
        }
        
        # İstatistik
        if eval_result["puan_katsayi"] == 0 and not student_ans:
            bos += 1
        elif eval_result["puan_katsayi"] >= 0.7:
            dogru += 1
        else:
            yanlis += 1
        
        toplam_katsayi += eval_result["puan_katsayi"]
        
        # İlerleme göster
        status = "✓" if eval_result["puan_katsayi"] >= 0.7 else "✗"
        print(f"{status} Soru {q_num}: {eval_result['puan_katsayi']*100:.0f}/100 - {eval_result['durum']} ({eval_result['yontem']})")
    
    # Toplamı 100'e ölçekle
    max_katsayi = len(correct_answers)
    yuzdelik_puan = (toplam_katsayi / max_katsayi * 100) if max_katsayi > 0 else 0

    final_result = {
        "ogrenci_no": ocr_data.get("student_id", ""),
        "ogrenci_adi": ocr_data.get("student_name", ""),
        "sorular": results,
        "ozet": {
            "toplam_puan": round(yuzdelik_puan, 2),
            "max_puan": 100,
            "dogru": dogru,
            "yanlis": yanlis,
            "bos": bos,
            "toplam_soru": len(correct_answers)
        }
    }
    
    # Kaydet
    output_file = f"output_yonetim/{os.path.splitext(os.path.basename(ocr_file))[0]}_evaluation.json"
    os.makedirs("output_yonetim", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"📊 SONUÇ: {yuzdelik_puan:.1f}/100 ({dogru} doğru, {yanlis} yanlış, {bos} boş)")
    print(f"💾 Kaydedildi: {output_file}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()