from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import json
from extract import extract_full_2209a_structure
from evaluation import evaluate
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_limiter.errors import RateLimitExceeded
import firebase_admin
from firebase_admin import credentials, auth, firestore
from functools import wraps
from rag import TurkishRAGSystem, embed_sections_with_local_model


app = Flask(__name__)
CORS(app)

limiter = Limiter(
    get_remote_address,
    app=app
)

# Firebase Admin SDK initialization from env variable
firebase_cred_json = os.environ.get("FIREBASE_CREDENTIALS")
if not firebase_cred_json:
    raise ValueError("FIREBASE_CREDENTIALS environment variable not set")

cred_dict = json.loads(firebase_cred_json)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUBRIC_PATH = os.path.join(BASE_DIR, "newRubric.json")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

rag_system = TurkishRAGSystem()

def verify_firebase_token(f):
    """Firebase token doğrulama decorator'ı"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token bulunamadı'}), 401
        
        try:
            # Bearer prefix'ini kaldır
            if token.startswith('Bearer '):
                token = token[7:]
            
            # Token'ı doğrula
            decoded_token = auth.verify_id_token(token)
            request.user_id = decoded_token['uid']
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'error': 'Geçersiz token'}), 401
    
    return decorated_function

@app.errorhandler(RateLimitExceeded)
def ratelimit_handler(e):
    return jsonify({"error": "Günlük istek limitine ulaştınız. Lütfen yarın tekrar deneyin."}), 429

@app.route("/projects")
def projects():
    return render_template("projects.html")

@app.route("/rag-history")
def rags():
    return render_template("rag-history.html")
@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/")
def index():
    return render_template("index.html") 

@app.route("/settings", methods=["GET"])
def settings():
    return render_template("settings.html") 

@app.route("/upload", methods=["POST"])
@verify_firebase_token
def upload():

    user_id = request.user_id
    if 'file' not in request.files:
        return jsonify({"error": "Dosya yüklenemedi"}), 400

    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.docx'):
        return jsonify({"error": "Sadece .docx dosyaları kabul edilir"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
   
        data = extract_full_2209a_structure(filepath)

        with open(RUBRIC_PATH, "r", encoding="utf-8") as f:
            rubric = json.load(f)

        engine = request.form.get("engine", "api")  
        model = request.form.get("model", "gpt-4o-mini-2024-07-18")

        result = evaluate(data, rubric, engine=engine, model=model)

        def temizle_cevap(yanit: str) -> str:
            if not yanit or not isinstance(yanit, str):
                return ""
            lines = yanit.strip().split("\n")
            if not lines:
                return ""

            cevap = lines[0].split(":")[-1].strip()
            gerekce = " ".join(
                line.replace("Gerekçe:", "").strip()
                for line in lines[1:]
                if line.strip()
            )

            if gerekce:
                return f"{cevap}, {gerekce}"
            return cevap

        for section, questions in result.get("answers", {}).items():
            for question, yanit in questions.items():
                result["answers"][section][question] = temizle_cevap(yanit)

        answers = result.get("answers", {})
        scores = result.get("section_scores", {})

        total_max_score = 0
        for section, questions in answers.items():
            question_count = len(questions)
            total_max_score += question_count * 5

        project_data = {
            'fileName': file.filename,
            'mode': 'evaluate',
            'model': model,
            'answers': answers,
            'scores': scores,
            'totalScore': sum(scores.values()),
            'totalMaxScore': total_max_score,  
            'completed': True,
            'createdAt': firestore.SERVER_TIMESTAMP
        }
        
        db.collection('users').document(user_id).collection('projects').add(project_data)
        os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"Hata oluştu: {str(e)}"}), 500

@app.route("/ask", methods=["POST"])
@verify_firebase_token
def ask_with_context():
    user_id = request.user_id
    
    if 'file' not in request.files or 'question' not in request.form:
        return jsonify({"error": "Dosya ve soru gereklidir."}), 400

    file = request.files['file']
    question = request.form['question'].strip()
    model = request.form.get("model", "gpt-4o-mini")

    if not question:
        return jsonify({"error": "Soru boş olamaz."}), 400
    
    if len(question) < 3:
        return jsonify({"error": "Soru en az 3 karakter olmalıdır."}), 400

    if file.filename == '' or not file.filename.endswith('.docx'):
        return jsonify({"error": "Sadece .docx dosyaları kabul edilir."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        print(f"Processing question: {question[:50]}...")

        data = extract_full_2209a_structure(filepath)
        
        if not data:
            os.remove(filepath)
            return jsonify({"error": "Doküman içeriği okunamadı."}), 400


        embedded_chunks = rag_system.embed_sections_with_local_model(data)
        
        if not embedded_chunks:
            os.remove(filepath)
            return jsonify({"error": "Dokümanda işlenebilir içerik bulunamadı."}), 400

        print(f"Found {len(embedded_chunks)} chunks")

        best_matches = rag_system.find_best_matches_enhanced(question, embedded_chunks, top_k=5)

        if best_matches:
            print(f"Best matches:")
            for i, match in enumerate(best_matches[:3]):
                print(f"  {i+1}. {match['section']}: similarity={match['similarity']:.3f}, combined={match.get('combined_score', 0):.3f}")

        answer = ""
        best_section = "N/A"
        similarity = 0.0
        
        if any(word in question.lower() for word in ['nedir', 'ne', 'nasıl', 'hangi', 'kim', 'neden']):
            threshold = 0.15 
        else:
            threshold = 0.25  
        
        if not best_matches or best_matches[0]["combined_score"] < threshold:
            if best_matches:
                available_sections = [match['section'] for match in best_matches[:3]]
                answer = f"""Bu soruya doküman içeriğinde tam olarak uygun bilgi bulunamadı. 

📋 Dokümanda bulunan ana bölümler:
• {chr(10).join(['• ' + section for section in available_sections])}

💡 Öneriler:
• Daha spesifik sorular sorun (örn: "Projenin amacı nedir?" veya "Hangi yöntemler kullanılacak?")
• Yukarıdaki bölümlerle ilgili sorular sorun"""
            else:
                answer = "Doküman analiz edildi ancak sorunuzla ilgili bilgi bulunamadı. Lütfen farklı bir soru deneyin."
                
            best_section = "Bilgi bulunamadı"
            similarity = best_matches[0]["similarity"] if best_matches else 0.0
        else:
            try:
                print("Calling OpenAI...")
                answer = rag_system.ask_openai(question, best_matches[:3], model=model)
                best_section = best_matches[0]['section']
                similarity = best_matches[0]["similarity"]
                print("OpenAI response received")
            except Exception as openai_error:
                print(f"OpenAI Error: {openai_error}")
                answer = f"""OpenAI servisi geçici olarak kullanılamıyor. 

📍 En ilgili bölüm: {best_matches[0]['section']}

📝 İçerik:
{best_matches[0]['text'][:500]}{'...' if len(best_matches[0]['text']) > 500 else ''}"""
                best_section = best_matches[0]['section']
                similarity = best_matches[0]["similarity"]

        project_data = {
            'fileName': file.filename,
            'mode': 'rag',
            'model': model,
            'question': question,
            'answer': answer,
            'bestSection': best_section,
            'similarity': round(similarity, 3),
            'matchedSections': len(best_matches),
            'threshold': threshold,
            'completed': True,
            'createdAt': firestore.SERVER_TIMESTAMP
        }
        db.collection('users').document(user_id).collection('rags').add(project_data)

        os.remove(filepath)

        return jsonify({
            "question": question,
            "answer": answer,
            "best_section": best_section,
            "similarity": round(similarity, 3),
            "matched_sections": len(best_matches),
            "success": True
        })

    except Exception as e:
        print(f"General Error: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)

        error_message = "Beklenmeyen bir hata oluştu."
        if "embedding" in str(e).lower():
            error_message = "Doküman analizi sırasında hata oluştu."
        elif "openai" in str(e).lower():
            error_message = "AI modeli ile iletişimde sorun yaşandı."
        elif "firestore" in str(e).lower():
            error_message = "Sonuç kaydedilirken hata oluştu."
            
        return jsonify({"error": error_message, "success": False}), 500
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
