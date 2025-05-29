from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import json
from extract import extract_full_2209a_structure
from evaluation import evaluate
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
CORS(app)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["2 per day"]
)



UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")  # templates klasöründen index.html yüklenir

@app.route("/upload", methods=["POST"])
@limiter.limit("2 per day")
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "Dosya yüklenemedi"}), 400

    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.docx'):
        return jsonify({"error": "Sadece .docx dosyaları kabul edilir"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Veriyi çıkar
        data = extract_full_2209a_structure(filepath)
       

        # Değerlendirme kurallarını yükle
        with open("newRubric.json", "r", encoding="utf-8") as f:
            rubric = json.load(f)


        # Kullanıcıdan gelen model ve engine bilgilerini al
        engine = request.form.get("engine", "api")  # api veya local
        model = request.form.get("model", "gpt-4o-mini-2024-07-18")


        # Değerlendirme sonucunu al
        result = evaluate(data, rubric, engine=engine, model=model)


        # 🔽 Temizleme fonksiyonu
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

        # 🔁 Tüm cevapları temizle
        for section, questions in result.get("answers", {}).items():
            for question, yanit in questions.items():
                result["answers"][section][question] = temizle_cevap(yanit)

        # Dosyayı kaldır
        os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Hata oluştu: {str(e)}"}), 500



# @app.route("/ask", methods=["POST"])
# def ask_with_context():
#     if 'file' not in request.files or 'question' not in request.form:
#         return jsonify({"error": "Dosya ve soru gereklidir."}), 400

#     file = request.files['file']
#     question = request.form['question']
#     model = request.form.get("model", "gpt-4o-mini-2024-07-18")

#     if file.filename == '' or not file.filename.endswith('.docx'):
#         return jsonify({"error": "Sadece .docx dosyaları kabul edilir."}), 400

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(UPLOAD_FOLDER, filename)
#     file.save(filepath)

#     try:
#         # 1. Dosyayı işle
#         data = extract_full_2209a_structure(filepath)

#         # 2. Section'ları embed et
#         embedded = embed_sections_with_local_model(data)

#         # 3. En iyi eşleşen bölümü bul
#         best_match = find_best_match(question, embedded)
#         context = best_match["context"]

#         # 4. GPT'ye bağlamla sor
#         prompt = (
#             "Aşağıdaki metni dikkatlice oku ve ardından kullanıcı sorusunu yanıtla.\n\n"
#             "Bağlam:\n"
#             "\"\"\"\n"
#             f"{context}\n"
#             "\"\"\"\n\n"
#             f"Soru: {question}\n\nCevap:"
#         )

#         answer = ask_openai(prompt, model=model)

#         os.remove(filepath)

#         return jsonify({
#             "question": question,
#             "best_section": best_match["section"],
#             "similarity": best_match["similarity"],
#             "context": context,
#             "answer": answer
#         })

#     except Exception as e:
#        return jsonify({"error": f"Hata oluştu: {str(e)}"}), 500
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
