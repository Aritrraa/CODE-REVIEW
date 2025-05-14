from flask import Flask, render_template, request, send_file
import io

app = Flask(__name__)

# Dummy analysis function (replace with actual CodeBERT + Logistic Regression + Pylint logic)
def analyze_code(code):
    return {
        "classification": "Needs Improvement",
        "confidence": 75,
        "lint_prediction": "Warning Detected",
        "errors": 3,
        "warnings": 2,
        "conventions": 5,
        "error_details": [
            {"line": 2, "message": "Indentation error"},
            {"line": 5, "message": "Unused variable"},
        ],
        "warning_details": [
            {"line": 7, "message": "Function name should be snake_case"}
        ],
        "convention_details": [
            {"line": 1, "message": "Missing docstring"},
            {"line": 6, "message": "Avoid using global variables"}
        ],
        "code": code
    }

@app.route('/')
def info():
    return render_template('info.html')  # The HTML you shared above

@app.route('/start', methods=['POST'])
def start():
    return render_template('index.html')  # A page with a code textarea form

@app.route('/analyze', methods=['POST'])
def analyze():
    uploaded_code = request.form['code']
    result = analyze_code(uploaded_code)
    return render_template('results.html', **result)

@app.route('/download_code_review')
def download_code_review():
    review = (
        "Code Review Result\n\n"
        "CodeBERT Classification: Needs Improvement\n"
        "Confidence: 75%\n"
        "Logistic Regression: Warning Detected\n"
        "Errors: 3 | Warnings: 2 | Conventions: 5\n"
        "Errors:\n"
        "Line 2: Indentation error\n"
        "Line 5: Unused variable\n"
        "Warnings:\n"
        "Line 7: Function name should be snake_case\n"
        "Conventions:\n"
        "Line 1: Missing docstring\n"
        "Line 6: Avoid using global variables\n"
    )

    # Serve the text as a downloadable file
    return send_file(
        io.BytesIO(review.encode()),
        mimetype="text/plain",
        as_attachment=True,
        download_name="code_review.txt"
    )

if __name__ == '__main__':
    app.run(debug=True)
