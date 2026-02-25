import cv2
import json
import sys
import os
import re
from paddleocr import PaddleOCR
import time

folder_path = "output(puan)"

def correct_ocr_errors(text: str) -> str:
    replacements = {
        's': '3', 'S': '3',
        'f': 'p', 'F': 'p',
        'l': '1', 'I': '1',
        'o': '0', 'O': '0',
        'ğ': '9', 'Ğ': '9',
        'ş': '6', 'Ş': '6',
        'ı': '1', 'İ': '1',
        'ρ': 'p', 'P': 'p'  # rho karakteri de p'ye dönüşsün
    }
    
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    
    return text

def extract_scores_from_text(text: str, all_scores: dict):
    corrected_text = correct_ocr_errors(text)
    original_text = text
    
    # Format: 1p=7, 5p=7, 6p=7 (p harfi yerine f, ρ de olabilir)
    pattern1 = r'(\d+)\s*[pPρ]\s*=\s*(\d+)'
    matches = re.findall(pattern1, corrected_text)
    for match in matches:
        if len(match) == 2:
            q_num = int(match[0])
            score = int(match[1])
            all_scores[q_num] = score
    
    # Format: p=7, p= 7 (boşluklu)
    pattern2 = r'[pPρ]\s*=\s*(\d+)'
    matches = re.findall(pattern2, corrected_text)
    
    if matches:
        numbers_in_text = re.findall(r'(\d+)', original_text)
        
        for score_match in matches:
            score = int(score_match)
            
            # Önce aynı satırda soru numarası ara (örn: "7. ... p=7")
            question_match = re.search(r'(\d+)\.', original_text)
            if question_match:
                q_num = int(question_match.group(1))
                if q_num not in all_scores and 1 <= q_num <= 50:
                    all_scores[q_num] = score
                    continue
            
            # Yoksa diğer sayıları dene
            for num in numbers_in_text:
                num_int = int(num)
                if num_int not in all_scores and 1 <= num_int <= 50:
                    if f"{num}p" in corrected_text.lower() or f"{num} p" in corrected_text.lower():
                        all_scores[num_int] = score
                        break

def process_ocr_json(json_file_path: str, original_image_path: str):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)
    
    rec_texts = ocr_data.get('rec_texts', [])
    
    if not rec_texts:
        print("Uyarı: rec_texts boş!")
        return None
    
    all_scores = {}
    
    for text in rec_texts:
        text = text.strip()
        if text:
            extract_scores_from_text(text, all_scores)
    
    print("\nBULUNAN NOTLAR:")
    print("-" * 30)
    
    if all_scores:
        for q_num in sorted(all_scores.keys()):
            print(f"Soru {q_num:2d} = {all_scores[q_num]:2d} puan")
    else:
        print("Not bulunamadı!")
    
    print("-" * 30)
    total = sum(all_scores.values()) if all_scores else 0
    print(f"Toplam Puan: {total}")
    
    result_data = {
        "image_path": original_image_path,
        "scores": all_scores,
        "total_score": total,
        "question_count": len(all_scores)
    }
    
    base_name = os.path.splitext(os.path.basename(original_image_path))[0]
    processed_json = f"{folder_path}/{base_name}_scores.json"
    
    with open(processed_json, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    return result_data

def run_ocr_on_image(image_path: str):
    ocr = PaddleOCR(
        use_doc_orientation_classify=False, 
        use_doc_unwarping=False, 
        use_textline_orientation=True,
        lang='tr',
    )
    
    result = ocr.predict(image_path)
    
    for res in result:
        res.save_to_json(folder_path)
    
    return result

def main():
    start_time = time.time()
    
    if len(sys.argv) < 2:
        print("Kullanım: python main_puan.py \"resim_yolu\"")
        print('Örnek: python main_puan.py "p1.jpeg"')
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Hata: {image_path} dosyası bulunamadı!")
        return
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    print(f"Dosya: {image_path}")
    
    run_ocr_on_image(image_path)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_file = f"{folder_path}/{base_name}_res.json"
    
    if os.path.exists(json_file):
        process_ocr_json(json_file, image_path)
    else:
        print(f"Hata: {json_file} oluşturulamadı!")
    
    end_time = time.time()
    print(f"İşlem süresi: {end_time - start_time:.2f} saniye")

if __name__ == "__main__":
    main()