import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
import requests
import openai
import easyocr
import cv2
from bs4 import BeautifulSoup

openai.api_key = os.environ['OPENAI']

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

pytesseract.pytesseract.tesseract_cmd = r'/nix/store/nprhbhaa9j23xm07hvl3fw27mm81nl1z-tesseract-5.3.4/bin/tesseract'

# Initialisation d'EasyOCR
reader = easyocr.Reader(['en'])

# Page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

# Traitement de l'image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files and 'image_url' not in request.form:
        return 'Aucune image ou URL fournie'

    file = request.files.get('file')
    image_url = request.form.get('image_url')

    if file:
        # Enregistrer l'image uploadée
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    elif image_url:
        # Télécharger l'image depuis l'URL
        response = requests.get(image_url)
        filename = secure_filename(image_url.split("/")[-1])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)

    # Prétraitement de l'image pour améliorer l'OCR
    preprocessed_image = preprocess_image(filepath)

    # Extraction de texte via Tesseract et EasyOCR
    tesseract_text = extract_text_from_image_tesseract(preprocessed_image)
    easyocr_text = extract_text_from_image_easyocr(preprocessed_image)

    # Fusion des résultats OCR
    text = f"Tesseract: {tesseract_text}\nEasyOCR: {easyocr_text}"

    # Utiliser GPT-4 pour améliorer la fiabilité
    improved_text = improve_text_with_gpt(text)

    return f'Texte extrait et amélioré: {improved_text}'

# Fonction de prétraitement de l'image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Binarisation avec le seuil d'Otsu
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Filtre de flou pour réduire le bruit
    img = cv2.medianBlur(img, 3)
    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(image_path))
    cv2.imwrite(processed_image_path, img)
    return processed_image_path

# Fonction d'OCR avec Tesseract
def extract_text_from_image_tesseract(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image, lang='eng', config='--psm 6')

# Fonction d'OCR avec EasyOCR
def extract_text_from_image_easyocr(image_path):
    result = reader.readtext(image_path, detail=0)
    return " ".join(result)

# Fonction d'amélioration via GPT-4
def improve_text_with_gpt(text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Tu es un assistant qui aide à corriger et améliorer du texte extrait par OCR."},
            {"role": "user", "content": f"Améliore et corrige ce texte extrait par OCR : {text}"}
        ],
        max_tokens=1000
    )
    return response['choices'][0]['message']['content'].strip()

# Fonction pour scrapper du texte depuis une page web avec BeautifulSoup
def scrape_page_for_text(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    return soup.get_text()

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)