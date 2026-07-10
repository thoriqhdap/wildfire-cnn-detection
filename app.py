import os
import time
import numpy as np
import keras
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import io

app = Flask(__name__, static_folder='static', template_folder='templates')

MODELS_DIR = "models"
DATASET_DIR = r"dataset/processed/train/Wild_Fire"

MODEL_PATHS = {
    "S1_LR_Besar": os.path.join(MODELS_DIR, "S1_LR_Besar.keras"),
    "S2_LR_Kecil": os.path.join(MODELS_DIR, "S2_LR_Kecil.keras"),
    "S5_Epoch_Tinggi": os.path.join(MODELS_DIR, "S5_Epoch_Tinggi.keras")
}

# Cache loaded models in memory for speed
loaded_models = {}

def get_model(name):
    if name not in loaded_models:
        model_path = MODEL_PATHS.get(name)
        if model_path and os.path.exists(model_path):
            print(f"Loading model into memory cache: {name}...")
            loaded_models[name] = keras.models.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model {name} not found at {model_path}")
    return loaded_models[name]

def preprocess_image(img_bytes, target_size=(224, 224)):
    # Read image from bytes
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    # Resize
    img_resized = img.resize(target_size)
    # Normalize to [0, 1]
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    # Expand dims to (1, 224, 224, 3)
    return np.expand_dims(img_array, axis=0)

def preprocess_image_from_path(filepath, target_size=(224, 224)):
    img = Image.open(filepath).convert('RGB')
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dataset')
def list_dataset():
    try:
        if not os.path.exists(DATASET_DIR):
            return jsonify({"success": False, "error": "Dataset directory not found"}), 404
        
        files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return jsonify({"success": True, "files": sorted(files)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/dataset/image/<filename>')
def get_dataset_image(filename):
    return send_from_directory(DATASET_DIR, filename)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Determine source (upload vs dataset)
        source = request.form.get('source', 'upload')
        img_array = None
        filename = "uploaded_image.png"
        
        if source == 'dataset':
            img_name = request.form.get('filename')
            if not img_name:
                return jsonify({"success": False, "error": "No filename specified"}), 400
            filepath = os.path.join(DATASET_DIR, img_name)
            if not os.path.exists(filepath):
                return jsonify({"success": False, "error": "File not found"}), 404
            img_array = preprocess_image_from_path(filepath)
            filename = img_name
        else:
            if 'image' not in request.files:
                return jsonify({"success": False, "error": "No image file provided"}), 400
            file = request.files['image']
            img_bytes = file.read()
            img_array = preprocess_image(img_bytes)
            filename = file.filename

        results = {}
        for name in MODEL_PATHS.keys():
            try:
                model = get_model(name)
                start_time = time.time()
                preds = model.predict(img_array)
                latency = (time.time() - start_time) * 1000 # ms
                
                score = float(preds[0][0])
                label = "Wildfire Detected" if score >= 0.5 else "Safe / No Fire"
                
                results[name] = {
                    "probability": score,
                    "class": label,
                    "latency_ms": round(latency, 2)
                }
            except Exception as e:
                results[name] = {
                    "error": str(e)
                }
                
        return jsonify({
            "success": True,
            "filename": filename,
            "predictions": results
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Endpoint to trigger batch evaluation
@app.route('/api/evaluate', methods=['POST'])
def trigger_evaluation():
    try:
        # Run evaluate.py script as process
        import subprocess
        result = subprocess.run([r"venv\Scripts\python", "evaluate.py"], capture_output=True, text=True)
        if result.returncode == 0:
            return jsonify({"success": True, "output": result.stdout})
        else:
            return jsonify({"success": False, "error": result.stderr})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Warm up models on startup
    print("Warming up models...")
    for model_name in MODEL_PATHS.keys():
        try:
            get_model(model_name)
        except Exception as e:
            print(f"Failed to preload {model_name}: {e}")
            
    print("Starting Flask web server on http://127.0.0.1:5000...")
    app.run(host='127.0.0.1', port=5000, debug=False)
