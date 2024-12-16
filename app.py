import os
import cv2
from moviepy import VideoFileClip
import numpy as np
from flask import Flask, flash, request, redirect, url_for, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask import send_from_directory

from src.config import base_dir
from src.predictions.detect_drone_in_image import detect_drone_in_image
from src.predictions.tracking_drone_in_video import tracking_drone_in_video

UPLOAD_FOLDER = os.path.join(base_dir, "uploadfolder")
SAVE_OUTPUTS_FILES = os.path.join(base_dir, "uploadfolder/outputs")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SAVE_OUTPUTS_FILES"] = SAVE_OUTPUTS_FILES
app.config['SECRET_KEY'] = 'oseesoke'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def convert_to_mp4(filepath):
    video = VideoFileClip(filepath)
    mp4_filepath = os.path.splitext(filepath)[0] + '.mp4'
    video.write_videofile(mp4_filepath, codec='libx264')
    return mp4_filepath

def allowed_file(filename):

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload')
def index():
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method="post" enctype="multipart/form-data" action="/detect/upload">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''
@app.route('/api')
def init():
    return "Welcome to my API. Add /upload to test the api"

@app.route('/api/download/<filename>')
def download_file(filename):
    ip_address = request.remote_addr
    return send_from_directory(app.config["SAVE_OUTPUTS_FILES"], filename, as_attachment=True)



@app.route('/detect/upload/image', methods=['POST', 'GET'])
def upload_image():
    try:
        if request.method == 'POST':
            print(request.files)
            if 'file' not in request.files:
                return redirect(request.url)

            file = request.files['file']
            if file.name == '':
                flash('No file selected')
                return redirect(request.url)

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], "outputs", filename)
                file.save(filepath)

                predict, image_detect, nbr = detect_drone_in_image(image_path=filepath)
                cv2.imwrite(filepath2, image_detect)
                result = {
                        'coordinates': predict,
                        'image': url_for('download_file', filename=filename, _external=True),
                        'num_objects': nbr
                        }
                return jsonify(result)
                
        print("Received a not POST request.")
    except Exception as e:
        print(e)
    return "No execution " 




@app.route('/detect/upload/video', methods=['POST', 'GET'])
def upload_video():
    """Traite l'upload et la détection de drones sur des vidéos."""
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return jsonify({"error": "No file part"}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                #file.save(filepath) 
                file_data = file.read()
                print(f"Taille du fichier : {len(file_data)} octets")
                with open(filepath, 'wb') as f:
                    f.write(file_data)

                try:
                    # Conversion et suivi sur la vidéo
                    if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                        #mp4_filepath = convert_to_mp4(filepath)
                        is_running, video_path, csv_file_path = tracking_drone_in_video(filepath)
                        result = {
                            'video': url_for('download_file', filename=os.path.splitext(filename)[0] + '.mp4', _external=True),
                            'coordonnate': url_for('download_file', filename=os.path.splitext(filename)[0] + '.csv', _external=True),
                            'run': is_running
                        }
                        return jsonify(result)
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
        return jsonify({"error": "Invalid request method"}), 405
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Utilisez la variable d'environnement PORT fournie par Render, sinon, utilisez un port par défaut (par exemple 5000).
    port = int(os.getenv("PORT", 5000)) 
    app.run(host="0.0.0.0", port=port)  # Assurez-vous que l'application écoute sur le port approprié. 
    
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
