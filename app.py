import os
import io
import json
import torch
import numpy as np
import subprocess
from joblib import load
from flask import Flask, render_template, request, send_file, session, jsonify
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn.functional import softmax
from datetime import timedelta
import logging

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# ==== Config ====
MODEL_NAME = "microsoft/codebert-base"
MODEL_DIR = "./models"
CODEBERT_DIR = "./codebert_finetuned"
LOGISTIC_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model_python.joblib")

# ==== Load Models ====
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
codebert_model = RobertaForSequenceClassification.from_pretrained(CODEBERT_DIR)
codebert_model.eval()
log_reg = load(LOGISTIC_MODEL_PATH)

# ==== Logging ====
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.before_request
def before_request():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=5)

# ==== Linter Feature Extraction ====
def extract_linter_features(code_list, lang_list):
    features = []
    all_errors, all_warnings, all_conventions = [], [], []

    for i, (code, lang) in enumerate(zip(code_list, lang_list)):
        temp_file = f"temp_{i}.{lang}"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(code)

        counts = {"E": 0, "W": 0, "C": 0}
        error_msgs, warn_msgs, conv_msgs = [], [], []

        try:
            result = subprocess.run(
                ['pylint', temp_file, '--output-format=json', '--score=n'],
                capture_output=True, text=True, timeout=30
            )
            output = result.stdout.strip()
            try:
                data = json.loads(output) if output else []
            except json.JSONDecodeError:
                logger.error(f"JSON decode error for output: {output}")
                data = []

            for item in data:
                msg = item.get("message-id", "")
                msg_text = item.get("message", "")
                line = item.get("line", 1)
                entry = {"line": line, "message": msg_text}

                if msg.startswith("E"):
                    counts["E"] += 1
                    error_msgs.append(entry)
                elif msg.startswith("W"):
                    counts["W"] += 1
                    warn_msgs.append(entry)
                elif msg.startswith("C"):
                    counts["C"] += 1
                    conv_msgs.append(entry)

        except subprocess.TimeoutExpired:
            logger.error(f"⚠️ Linter timed out for file {temp_file}")
        except Exception as e:
            logger.error(f"⚠️ Linter error: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        features.append([counts["E"], counts["W"], counts["C"]])
        all_errors.append(error_msgs)
        all_warnings.append(warn_msgs)
        all_conventions.append(conv_msgs)

    return np.array(features), all_errors, all_warnings, all_conventions

# ==== CodeBERT Prediction ====
def codebert_predict(code_samples):
    inputs = tokenizer(code_samples, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = codebert_model(**inputs)
        logits = outputs.logits
        probs = softmax(logits, dim=1).cpu().numpy()
        predictions = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1) * 100
    return predictions, confidences

# ==== Main Analyze Route ====
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        uploaded_code = request.form.get('code')
        if not uploaded_code and 'file' in request.files:
            file = request.files['file']
            uploaded_code = file.read().decode('utf-8')

        if not uploaded_code:
            return "No code provided", 400

        code_samples = [uploaded_code]

        # Extract Linter Features
        linter_features, error_list, warn_list, conv_list = extract_linter_features(code_samples, ["py"])
        logger.debug(f"Linter features: {linter_features}")

        errors = error_list[0]
        warnings = warn_list[0]
        conventions = conv_list[0]

        # Run CodeBERT
        codebert_pred, confidences = codebert_predict(code_samples)
        logger.debug(f"CodeBERT: {codebert_pred}, Confidence: {confidences}")

        # Run Logistic Regression on linter features
        lint_pred = int(log_reg.predict(linter_features)[0])
        logger.debug(f"Logistic Regression: {lint_pred}")

        # Session Values
        session['classification'] = "Good" if codebert_pred[0] == 1 else "Needs Improvement"
        session['confidence'] = round(float(confidences[0]), 2)
        session['lint_prediction'] = "Warning Detected" if lint_pred == 1 else "No Issues"
        session['errors'] = errors
        session['warnings'] = warnings
        session['conventions'] = conventions
        session['code'] = uploaded_code

        return render_template('result.html',
            classification=session['classification'],
            confidence=session['confidence'],
            lint_prediction=session['lint_prediction'],
            errors=len(errors),
            warnings=len(warnings),
            conventions=len(conventions),
            error_details=errors,
            warning_details=warnings,
            convention_details=conventions,
            code=uploaded_code
        )

    except Exception as e:
        logger.exception("Unhandled error in /analyze route")
        return "Internal Server Error", 500

# ==== Download Review ====
@app.route('/download_code_review')
def download_code_review():
    def format_issues(issues):
        return "\n".join([f"Line {item['line']}: {item['message']}" for item in issues]) or "None"

    review = (
        f"Code Review Report\n\n"
        f"CodeBERT Classification: {session.get('classification', 'Needs Improvement')}\n"
        f"Confidence: {session.get('confidence', 75)}%\n"
        f"Logistic Regression: {session.get('lint_prediction', 'Warning Detected')}\n"
        f"Errors: {len(session.get('errors', []))} | "
        f"Warnings: {len(session.get('warnings', []))} | "
        f"Conventions: {len(session.get('conventions', []))}\n\n"
        f"Errors:\n{format_issues(session.get('errors', []))}\n\n"
        f"Warnings:\n{format_issues(session.get('warnings', []))}\n\n"
        f"Conventions:\n{format_issues(session.get('conventions', []))}\n"
    )

    return send_file(
        io.BytesIO(review.encode()),
        mimetype="text/plain",
        as_attachment=True,
        download_name="code_review.txt"
    )

# ==== ChatGPT Suggestions (Placeholder) ====
@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    code = session.get('code', '')
    if not code:
        return jsonify({'suggestions': "No code found in session."})

    suggestion = (
        f"Suggested improvements for the submitted code:\n"
        f"- Use meaningful variable and function names\n"
        f"- Add comments and docstrings for better readability\n"
        f"- Break down long functions into smaller reusable units\n"
        f"- Follow PEP8 indentation and formatting\n"
        f"- Ensure proper exception handling"
    )
    return jsonify({'suggestions': suggestion})

# ==== Static Routes ====
@app.route('/')
def info():
    return render_template('info.html')

@app.route('/start', methods=['GET', 'POST'])
def start():
    return render_template('index.html')

# ==== Run Server ====
if __name__ == '__main__':
    app.run(debug=True)
