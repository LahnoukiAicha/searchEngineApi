from flask import Flask, request, jsonify, send_from_directory
from googleapiclient.discovery import build
from scholarly import scholarly
from PIL import Image as PILImage
import pytesseract
from rake_nltk import Rake
import nltk
import os
import logging
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# Initialize logging
logging.basicConfig(level=logging.INFO)

# Download required nltk resources
nltk.download('punkt')
nltk.download('stopwords')

# Flask setup
app = Flask(__name__)
CORS(app, origins="http://localhost:3000")

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Load stopwords for keyword extraction
english_stop_words = set(stopwords.words('english'))

# Function to extract text from an image
def extract_text_from_image(image_path):
    try:
        image = PILImage.open(image_path)
        text = pytesseract.image_to_string(image, lang="eng")
        logging.info(f"Processed image: {image_path}")
        return text
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return ""

# Function to extract keywords from an image
def extract_keywords_from_image(image_path):
    text = extract_text_from_image(image_path)
    if text.strip():
        rake = Rake(stopwords=english_stop_words)
        rake.extract_keywords_from_text(text)
        return rake.get_ranked_phrases()
    return []

# Preprocess query to remove stopwords
def preprocess_query(query):
    tokens = word_tokenize(query.lower())  # Tokenize the query
    return [word for word in tokens if word not in english_stop_words]

# Function to filter images based on query
def filter_images_by_query(image_dir, query):
    core_keywords = preprocess_query(query)  # Process query to get important keywords
    filtered_images = {}

    if not os.path.exists(image_dir):
        logging.error(f"Image directory not found: {image_dir}")
        return {"error": f"Image directory {image_dir} not found"}

    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_dir, image_file)
            keywords = extract_keywords_from_image(image_path)

            # Check if any core keyword from the query matches any of the extracted keywords
            if any(core_keyword in keyword.lower() for core_keyword in core_keywords for keyword in keywords):
                filtered_images[image_file] = {
                    "image_path": f"/images/{image_file}",  # URL path to the image
                    "keywords": keywords
                }
    return filtered_images

# Function to search Google Scholar
def search_google_scholar(query):
    try:
        search_query = scholarly.search_pubs(query)
        papers = []
        for _ in range(15):  # Limit to 15 results
            paper = next(search_query, None)
            if not paper:
                break
            papers.append({
                'title': paper.get('bib', {}).get('title', 'N/A'),
                'authors': paper.get('bib', {}).get('author', 'N/A'),
                'link': paper.get('pub_url', 'N/A')
            })
        return papers
    except Exception as e:
        logging.error(f"Error in Google Scholar search: {e}")
        return {"error": f"Error in Google Scholar search: {e}"}

# Function to search YouTube
def search_youtube(query, api_key):
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            videoDuration="short",  # Filter for short videos
            maxResults=5
        )
        response = request.execute()
        videos = []
        for item in response.get("items", []):
            videos.append({
                "title": item["snippet"]["title"],
                "link": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "description": item["snippet"]["description"],
                "thumbnail": item["snippet"]["thumbnails"]["high"]["url"]
            })
        return videos
    except Exception as e:
        logging.error(f"Error in YouTube search: {e}")
        return {"error": f"Error in YouTube search: {e}"}

# Flask route for serving images
@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

# Flask route for favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')

# Flask route for query processing
@app.route('/search', methods=['GET'])
def process_query():
    query = request.args.get('query', '')  # Get the query from the request
    youtube_api_key = request.args.get('youtube_api_key', '')  # Get the API key for YouTube
    image_dir = request.args.get("image_dir", "images/")  # Default to 'images/' folder

    if not query or not youtube_api_key:
        return jsonify({"error": "Please provide both 'query' and 'youtube_api_key'"}), 400

    # Fetch data
    pdfs = search_google_scholar(query)
    videos = search_youtube(query, youtube_api_key)
    image_keywords = filter_images_by_query(image_dir, query)

    # Return results
    return jsonify({
        "pdfs": pdfs,
        "videos": videos,
        "image_keywords": image_keywords
    })

if __name__ == '__main__':
    app.run(debug=True)
