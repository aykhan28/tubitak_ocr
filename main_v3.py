# v3.py

import cv2
import json
import sys
import os
import re
from paddleocr import PaddleOCR
import time

def preprocess_image(image_path: str):
    # Görüntüyü Otsu thresholding ile önişlemeden geçirir
    print(f"Görüntü önişleme başlatılıyor: {image_path}")
    
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Hata: {image_path} okunamadı!")
        return None
    
    # Gri Tonlama
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Otsu Thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Önişlenmiş görüntüyü kaydet
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    preprocessed_path = f"output/{base_name}_preprocessed.jpg"
    cv2.imwrite(preprocessed_path, thresh)
    
    print(f"Önişlenmiş görüntü kaydedildi: {preprocessed_path}")
    
    return preprocessed_path

def run_ocr_on_image(image_path: str):
    #Resim üzerinde PaddleOCR çalıştırır ve sonuçları JSON olarak kaydeder
    print(f"OCR çalıştırılıyor: {image_path}")
    
    ocr = PaddleOCR(
        use_doc_orientation_classify=False, 
        use_doc_unwarping=False, 
        use_textline_orientation=True,
        lang = 'tr'
    )
    
    result = ocr.predict(image_path)
    
    # OCR sonuçlarını JSON olarak kaydet
    for res in result:
        res.save_to_json("output")
    
    return result

def correct_common_ocr_errors(text: str) -> str:
    # OCR hatalarını düzelt
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
        'ñ': 'n', 'ń': 'n', 'ņ': 'n', 'ň': 'n', 'ŉ': 'n'
    }
    
    text = text.translate(str.maketrans(char_map))

    # sayısal OCR düzeltmeleri
    text = text.replace('u', '4').replace('U', '4')
    text = text.replace('o', '0').replace('O', '0')
    text = text.replace('l', '1').replace('I', '1')
    
    return text

def process_ocr_json(json_file_path: str, original_image_path: str):
    # JSON dosyasındaki rec_texts'i işler ve düzenlenmiş çıktı üretir
    print(f"\nJSON işleniyor: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)
    
    rec_texts = ocr_data.get('rec_texts', [])
    rec_scores = ocr_data.get('rec_scores', [])  # Skorları al
    
    if not rec_texts:
        print("⚠️ Uyarı: rec_texts boş!")
        return None
    
    # Verileri düzenle
    student_name = None
    student_id = None
    answers = {}
    answer_warnings = {}  # Düşük güvenilirlik uyarıları için
    
    for i, text in enumerate(rec_texts):
        text = text.strip()
        
        # Ad Soyad bul
        if "Ad Soyad" in text or "ad soyad" in text.lower() or "Ad soyad" in text:
            parts = text.split(":")
            if len(parts) > 1:
                student_name = parts[1].strip()
        
        # Öğrenci Numarası bul
        elif "Ogrenci No" in text or "Öğrenci No" in text or "ogrenci no" in text.lower():
            parts = text.split(":")
            if len(parts) > 1:
                # OCR hatalarını düzelt
                raw_id = parts[1].strip()
                student_id = correct_common_ocr_errors(raw_id)
                # Sadece rakamları al
                student_id = ''.join(filter(str.isdigit, student_id))
        
        # Soru cevaplarını bul
        elif "Soru" in text or "soru" in text.lower():
            # Soru numarasını çıkar
            match = re.search(r'[Ss]oru\s*(\d+)', text)
            if match:
                question_num = int(match.group(1))
                answer = None
                low_confidence_indices = []  # Düşük güvenilirlik olan indeksler
                
                # Cevabı çıkar
                if ":" in text:
                    parts = text.split(":", 1)
                    if len(parts) > 1:
                        answer = parts[1].strip()
                        # İlk satırın skorunu kontrol et
                        if i < len(rec_scores) and rec_scores[i] < 0.85:
                            low_confidence_indices.append(i)
                
                # Alt satırlara kayan cevaplar için kontrol - sonraki "Soru" görülene kadar devam et
                j = i + 1
                while j < len(rec_texts):
                    next_text = rec_texts[j].strip()
                    # Sonraki satır yeni bir soru mu kontrol et
                    if re.search(r'[Ss]oru\s*\d+', next_text):
                        break
                    # Soru değilse ve boş değilse, cevaba ekle
                    if next_text:
                        # Skorunu kontrol et
                        if j < len(rec_scores) and rec_scores[j] < 0.85:
                            low_confidence_indices.append(j)
                        
                        if answer:
                            answer += " " + next_text
                        else:
                            answer = next_text
                    j += 1
                
                # Boş cevap kontrolü
                if not answer or answer == "" or answer.lower() == "boş" or answer.lower() == "bos":
                    answers[question_num] = "Boş"
                else:
                    answers[question_num] = answer
                    # Düşük güvenilirlik uyarısı ekle
                    if low_confidence_indices:
                        answer_warnings[question_num] = {
                            "warning": "Düşük OCR güvenilirliği",
                            "low_confidence_scores": [rec_scores[idx] for idx in low_confidence_indices if idx < len(rec_scores)]
                        }
    

    # Düzenlenmiş çıktıyı oluştur
    formatted_output = []
    formatted_output.append(f"Ad Soyad: {student_name if student_name else 'Bulunamadı'}")
    formatted_output.append(f"Öğrenci Numarası: {student_id if student_id else 'Bulunamadı'}")
    formatted_output.append("\nCevaplar:")
    
    if answers:
        for q_num in sorted(answers.keys()):
            formatted_output.append(f"{q_num}. {answers[q_num]}")
    else:
        formatted_output.append("Cevap bulunamadı")
    
    # Sonucu JSON olarak kaydet
    result_data = {
        "student_name": student_name,
        "student_id": student_id,
        "image_path": original_image_path,
        "answers": answers,
        "answer_warnings": answer_warnings,  # Uyarıları ekle
        "total_questions": len(answers),
        "answered_questions": len([a for a in answers.values() if a != "Boş"]),
        "empty_questions": len([a for a in answers.values() if a == "Boş"])
    }

    base_name = os.path.splitext(os.path.basename(original_image_path))[0]
    processed_json = f"output/{base_name}_processed.json"
    
    with open(processed_json, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"İşlenmiş JSON kaydedildi: {processed_json}")
    
    return result_data

def main():
    #başlangıç zamanı
    start_time = time.time()
    
    if len(sys.argv) < 2:
        print("Kullanım: python v3.py \"resim_yolu\"")
        print("\nÖrnek:")
        print('python v3.py "examm.jpg"')
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Hata: {image_path} dosyası bulunamadı!")
        return
    
    # Output klasörünü oluştur
    if not os.path.exists("output"):
        os.makedirs("output")
    
    # 1. Görüntü önişleme
    preprocessed_path = preprocess_image(image_path)
    
    if preprocessed_path is None:
        return
    
    # 2. OCR işlemi (önişlenmiş görüntü üzerinde)
    ocr_result = run_ocr_on_image(preprocessed_path)
    print("OCR tamamlandı!")
    
    # JSON dosyası konumu
    base_name = os.path.splitext(os.path.basename(preprocessed_path))[0]
    json_file = f"output/{base_name}_res.json"
    
    if os.path.exists(json_file):
        
        # 3. JSON'ı işle ve düzenle
        process_ocr_json(json_file, image_path)
    else:
        print(f"⚠️ Uyarı: {json_file} dosyası oluşturulamadı!")
    
    print("\nİşlem tamamlandı!")
    print("=" * 50)
    #bitiş zamanı
    end_time = time.time()

    print(f"Toplam işlem süresi: {end_time - start_time:.2f} saniye")

if __name__ == "__main__":
    main()