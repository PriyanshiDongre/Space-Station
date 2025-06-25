from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Path to store uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your trained YOLOv8 model
model = YOLO("C:/Users/Hp/Downloads/Hackathon_Dataset/HackByte_Dataset/runs/detect/orbis_final_model2/weights/best.pt")

# Route to check API status
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "YOLOv8 API is running!"})

# Route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    filename = secure_filename(image_file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(image_path)

    # Run YOLO prediction
    results = model(image_path)

    # Parse results
    predictions = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[cls_id]
            predictions.append({
                "class": class_name,
                "confidence": round(confidence, 2)
            })

    return jsonify({
        "filename": filename,
        "predictions": predictions
    })

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)