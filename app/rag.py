# Enhanced rag_system.py for production
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from typing import Dict, List, Any
from openai import OpenAI
from dotenv import load_dotenv
import re
load_dotenv()

class TurkishRAGSystem:
    def __init__(self):
        # Türkçe için optimize edilmiş model
        self.model=None
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Başlık öncelik mapping'i
        self.title_priority = {
            "başlık": 10,
            "amaç": 9,
            "hedef": 8,
            "önemi": 7,
            "özgün": 6,
            "hipotez": 5,
            "özet": 4,
            "yöntem": 3,
            "bütçe": 2,
            "etki": 1
        }
    def load_model(self):
        if self.model is None:
            try:
                # Türkçe için optimize edilmiş model
                self.model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
            except Exception as e:
                print(f"Model yükleme hatası: {e}")
                raise RuntimeError("Model yüklenemedi. Lütfen modeli kontrol edin.")

    def embed_sections_with_local_model(self, data: Dict) -> List[Dict]:
        """2209-A proje yapısındaki section'ları vektörleştir - enhanced version"""
        self.load_model()
        chunks = []
        
        # Ana section'ları işle
        for section_name, content in data.items():
            # String content'ler
            if isinstance(content, str) and content.strip():
                priority = self._calculate_priority(section_name)
                chunks.append({
                    "section": section_name,
                    "text": content.strip(),
                    "type": "text",
                    "priority": priority
                })
            
            # Dictionary content'ler (alt section'lar)
            elif isinstance(content, dict):
                for subkey, subval in content.items():
                    if isinstance(subval, str) and subval.strip():
                        priority = self._calculate_priority(f"{section_name} > {subkey}")
                        chunks.append({
                            "section": f"{section_name} > {subkey}",
                            "text": subval.strip(),
                            "type": "subsection",
                            "priority": priority
                        })
                    elif isinstance(subval, dict):
                        # Nested dict'ler için
                        for nested_key, nested_val in subval.items():
                            if isinstance(nested_val, str) and nested_val.strip():
                                chunks.append({
                                    "section": f"{section_name} > {subkey} > {nested_key}",
                                    "text": f"Soru: {nested_key}\nCevap: {nested_val}",
                                    "type": "qa_pair",
                                    "priority": 3
                                })
            
            # List content'ler (tablolar)
            elif isinstance(content, list) and content:
                table_text = self._format_table(content)
                if table_text:
                    chunks.append({
                        "section": f"{section_name} (Tablo)",
                        "text": table_text,
                        "type": "table",
                        "priority": 2
                    })

        # Metinleri vektörleştir
        if chunks:
            texts = [chunk["text"] for chunk in chunks]
            try:
                vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                for i, chunk in enumerate(chunks):
                    chunk["vector"] = vectors[i]
            except Exception as e:
                print(f"Embedding error: {e}")
                return []
        
        return chunks

    def _calculate_priority(self, section_name: str) -> int:
        """Section'un önem derecesini hesapla"""
        section_lower = section_name.lower()
        max_priority = 0
        
        for keyword, priority in self.title_priority.items():
            if keyword in section_lower:
                max_priority = max(max_priority, priority)
        
        return max_priority if max_priority > 0 else 1

    def _format_table(self, table_data: List) -> str:
        """Tablo verisini okunabilir metne çevir"""
        try:
            if not table_data:
                return ""
            
            formatted_rows = []
            for row in table_data:
                if isinstance(row, dict):
                    row_text = " | ".join([f"{k}: {v}" for k, v in row.items() if v])
                elif isinstance(row, list):
                    row_text = " | ".join([str(item) for item in row if item])
                else:
                    row_text = str(row)
                
                if row_text.strip():
                    formatted_rows.append(row_text)
            
            return "\n".join(formatted_rows)
        except:
            return json.dumps(table_data, ensure_ascii=False, indent=2)

    def find_best_matches_enhanced(self, question: str, embedded_chunks: List[Dict], top_k: int = 3) -> List[Dict]:
        """Enhanced matching with title priority and semantic + keyword search"""
        if not embedded_chunks:
            return []

        try:
            # Soruyu vektörleştir
            q_vec = self.model.encode([question], convert_to_numpy=True)[0]
        except Exception as e:
            print(f"Question embedding error: {e}")
            return []
        
        # Soru kelimelerini al
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        
        # Similarity hesapla
        similarities = []
        for chunk in embedded_chunks:
            try:
                # Semantic similarity
                semantic_sim = cosine_similarity([q_vec], [chunk["vector"]])[0][0]
                
                # Keyword similarity
                chunk_words = set(re.findall(r'\b\w+\b', chunk["text"].lower()))
                common_words = question_words.intersection(chunk_words)
                keyword_sim = len(common_words) / len(question_words) if question_words else 0
                
                # Priority boost
                priority_boost = chunk.get("priority", 1) * 0.1
                
                # Combined score
                combined_score = (
                    semantic_sim * 0.6 +  # Semantic ağırlık
                    keyword_sim * 0.3 +   # Keyword ağırlık  
                    priority_boost * 0.1  # Priority boost
                )
                
                similarities.append({
                    "section": chunk["section"],
                    "text": chunk["text"],
                    "type": chunk["type"],
                    "similarity": float(semantic_sim),
                    "combined_score": float(combined_score),
                    "keyword_match": len(common_words)
                })
            except Exception as e:
                print(f"Similarity calculation error: {e}")
                continue
        
        # Combined score'a göre sırala
        similarities.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return similarities[:top_k]

    def find_best_matches(self, question: str, embedded_chunks: List[Dict], top_k: int = 3) -> List[Dict]:
        """Wrapper for backward compatibility"""
        return self.find_best_matches_enhanced(question, embedded_chunks, top_k)

    def ask_openai(self, question: str, context_chunks: List[Dict], model: str = "gpt-4o-mini") -> str:
        """Enhanced OpenAI call with better prompt engineering"""
        
        # Context'leri önceliğe göre sırala ve birleştir
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            keyword_info = f" (Anahtar kelime eşleşmesi: {chunk.get('keyword_match', 0)})" if chunk.get('keyword_match', 0) > 0 else ""
            context_parts.append(f"Bölüm {i} - {chunk['section']}{keyword_info}:\n{chunk['text']}")
        
        combined_context = "\n\n" + "="*50 + "\n\n".join(context_parts)
        
        # Enhanced Turkish prompt
        prompt = f"""Sen 2209-A projelerini değerlendiren bir uzmansın. Aşağıdaki proje dokümanı bölümlerini analiz ederek soruyu yanıtla.

📋 PROJE DOKÜMANI BÖLÜMLERİ:
{combined_context}

❓ KULLANICI SORUSU: {question}

📝 YANIT KURALLARI:
• Sadece verilen doküman bölümlerindeki bilgileri kullan
• Türkçe, net ve anlaşılır yanıt ver
• Hangi bölümden bilgi aldığını belirt
• Eğer yeterli bilgi yoksa, bunu belirt ve alternatif soru öner
• Maddeler halinde düzenle (gerekirse)

💡 YANIT:"""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Sen deneyimli bir proje değerlendirme uzmanısın. Verilen doküman bölümlerini analiz ederek net, profesyonel yanıtlar veriyorsun."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.2  # Daha tutarlı yanıtlar için düşük temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"OpenAI API hatası: {str(e)}")
            return f"Üzgünüm, AI modeli ile iletişimde bir sorun oluştu. Lütfen daha sonra tekrar deneyin."

# Global instance
rag_system = TurkishRAGSystem()

# Backward compatibility functions
def embed_sections_with_local_model(data):
    return rag_system.embed_sections_with_local_model(data)

def find_best_match(question, embedded_chunks):
    matches = rag_system.find_best_matches(question, embedded_chunks, top_k=1)
    if matches:
        return {
            "section": matches[0]["section"],
            "context": matches[0]["text"],
            "similarity": matches[0]["similarity"]
        }
    return {
        "section": "Bulunamadı",
        "context": "İlgili bölüm bulunamadı",
        "similarity": 0.0
    }