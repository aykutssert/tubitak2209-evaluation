import json
import requests
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI



# HTTP bağlantılarını yeniden kullanmak için Session oluştur
session = requests.Session()





client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

import time

def ask_openai(prompt: str,model, retries=3, delay=5,) -> str:
    print(f" {model} ile iletişim kuruyor...")
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Hata: {e}")
            if attempt < retries - 1:
                print(f"{delay} saniye bekleniyor ve tekrar deneniyor...")
                time.sleep(delay)
            else:
                return f"Hata: OpenAI isteği başarısız oldu - {str(e)}"


def ask_ollama(prompt, model="gemma3:4b"):
    print(f"Ollama {model} ile iletişim kuruyor...")
  
    try:
        response = session.post("http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False
        })
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.RequestException as e:
        return f"Hata: Ollama isteği başarısız oldu - {str(e)}"





def build_prompt_for_section(section_name, content, questions):
    """Bölüm için LLM prompt'u oluşturur."""
    numbered_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    return f"""
Aşağıdaki {section_name} bölümünü değerlendirmeni istiyorum.

Metin:
\"\"\"
{content}
\"\"\"

Sorular:
{numbered_questions}

Her soruya şu formatta cevap ver:

1. Cevap: Evet / Hayır / Kısmen  
   Gerekçe: Açıklayıcı bir gerekçe verin.
2. ...
"""

def parse_response_by_section(response, questions):
    """LLM yanıtını sorulara göre ayrıştırır."""
    result = {}
    lines = response.strip().split("\n")
    current_lines = []
    current_idx = 1

    for line in lines:
        if line.strip().startswith(f"{current_idx}."):
            if current_lines and current_idx - 2 < len(questions):
                result[questions[current_idx - 2]] = "\n".join(current_lines).strip()
            current_lines = [line]
            current_idx += 1
        else:
            current_lines.append(line)

    if current_lines and current_idx - 2 < len(questions):
        result[questions[current_idx - 2]] = "\n".join(current_lines).strip()

    # Eksik sorular için varsayılan değer
    for i, q in enumerate(questions):
        if q not in result:
            result[q] = "Cevap: Kısmen\nGerekçe: Yanıt alınamadı."
    return result

def score_answer(answer):
    """Cevaba göre puan hesaplar."""
    lowered = answer.lower()
    if "evet" in lowered and "hayır" not in lowered:
        return 5
    elif "kısmen" in lowered:
        return 3
    elif "hayır" in lowered:
        return 0
    else:
        return 0  # Varsayılan olarak 0 puan

def evaluate_summary_manually(content_dict):
    """Özet bölümünü manuel olarak değerlendirir."""
    result = {}
    ozet_bolumu = content_dict.get("ÖZET", {})
    ozet_text = ozet_bolumu.get("Özet", "").strip()
    anahtarlar = ozet_bolumu.get("Anahtar Kelimeler", "").strip()

    word_count = len(ozet_text.split())
    if 25 <= word_count <= 450:
        result["Projenin özeti 25 - 450 kelime arasında mı?"] = f"Cevap: Evet\nGerekçe: Özet {word_count} kelime içeriyor."
    else:
        result["Projenin özeti 25 - 450 kelime arasında mı?"] = f"Cevap: Hayır\nGerekçe: Özet {word_count} kelime içeriyor."

    keywords = [k.strip() for k in anahtarlar.split(",") if k.strip()]
    count = len(keywords)
    if 3 <= count <= 5:
        result["Anahtar kelime sayısı 3 ile 5 arasında mı?"] = f"Cevap: Evet\nGerekçe: {count} anahtar kelime bulundu."
    else:
        result["Anahtar kelime sayısı 3 ile 5 arasında mı?"] = f"Cevap: Hayır\nGerekçe: {count} anahtar kelime bulundu."

    return result

def evaluate(data, rubric, engine="api", model="gpt-4o-mini-2024-07-18"):

    """Verilen veriyi ve rubriği değerlendirir."""
    print("Evaluating data...")
    full_result = {}
    section_scores = {}
    lock = threading.Lock()

    def evaluate_section(section, questions):
        nonlocal full_result, section_scores
        content = None
        if section in data:
            content = data[section] if isinstance(data[section], str) else data[section]
        else:
            for major in data:
                if isinstance(data[major], dict) and section in data[major]:
                    content = data[major][section]
                    break

        if not content:
            with lock:
                full_result[section] = {q: "❌ İçerik bulunamadı" for q in questions}
                section_scores[section] = 0
            return

        manual_answers = {}
        if section == "ÖZET":
            manual_answers = evaluate_summary_manually(data)

        llm_questions = [q for q in questions if q not in manual_answers]
        llm_result = {}
        if llm_questions:
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
            prompt = build_prompt_for_section(section, text, llm_questions)
            if engine == "api":
                response = ask_openai(prompt,model=model)
            else:
                response = ask_ollama(prompt, model=model)
          
            llm_result = parse_response_by_section(response, llm_questions)

        combined_result = {**manual_answers, **llm_result}
        with lock:
            full_result[section] = combined_result
            section_scores[section] = sum(score_answer(ans) for ans in combined_result.values())

    # Thread pool ile paralel çalıştırma
    with ThreadPoolExecutor(max_workers=10) as executor:  # max_workers sistem kaynaklarına göre ayarlanabilir
        executor.map(lambda args: evaluate_section(*args), [(section, questions) for section, questions in rubric.items()])

    return {
        "answers": full_result,
        "section_scores": section_scores
    }