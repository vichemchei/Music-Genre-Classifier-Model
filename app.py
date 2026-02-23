import os
import tempfile
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from feature_extractor import (
    extract_features_from_file,
    extract_features_from_bytes,
    extract_features_from_system_audio,
)


# App setup
app = Flask(__name__)
CORS(app)

# Allowed audio extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'webm', 'mp4', 'm4a', 'aac'}


# Load model artifacts once at startup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'genre_classifier.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'genre_scaler.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'genre_label_encoder.pkl'))

print(f"✓ Model loaded:  {type(model).__name__}")
print(f"✓ Scaler loaded:  {type(scaler).__name__}")
print(f"✓ Labels:  {list(label_encoder.classes_)}")


def _allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _predict(features: np.ndarray) -> dict:
    """
    Scale features -> predict -> build response dict.
    """
    # Scale with the same scaler used during training
    features_scaled = scaler.transform(features)

    
    # If the model supports predict_proba (SVM trained with probability=True)
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(features_scaled)[0]
    elif hasattr(model, 'decision_function'):
        # Convert SVM decision function to pseudo-probabilities via softmax
        decision = model.decision_function(features_scaled)[0]
        exp_d = np.exp(decision - np.max(decision))  # numerical stability
        probas = exp_d / exp_d.sum()
    else:
        # Fallback: just predict, no confidence
        pred_idx = model.predict(features_scaled)[0]
        genre = label_encoder.inverse_transform([pred_idx])[0]
        return {
            'genre': genre,
            'confidence': 1.0,
            'top_genres': [{'genre': genre, 'confidence': 1.0}],
        }

    # Sort by confidence (descending)
    sorted_indices = np.argsort(probas)[::-1]
    top_genres = []
    for idx in sorted_indices:
        top_genres.append({
            'genre': label_encoder.inverse_transform([idx])[0],
            'confidence': round(float(probas[idx]), 4),
        })

    best = top_genres[0]
    return {
        'genre': best['genre'],
        'confidence': best['confidence'],
        'top_genres': top_genres,
    }




@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model': type(model).__name__,
        'genres': list(label_encoder.classes_),
    })


@app.route('/genres', methods=['GET'])
def genres():
    """Return list of supported genres."""
    return jsonify({
        'genres': list(label_encoder.classes_),
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict genre from an uploaded audio file.

    Expects multipart/form-data with a 'file' field.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided. Send a file with key "file".'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            'error': f'Unsupported file type. Allowed: {", ".join(sorted(ALLOWED_EXTENSIONS))}',
        }), 400

    # Save to temp file for librosa to read
    ext = file.filename.rsplit('.', 1)[1].lower()
    tmp = tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False)
    try:
        file.save(tmp.name)
        features = extract_features_from_file(tmp.name)
        result = _predict(features)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Failed to process audio: {str(e)}'}), 500
    finally:
        os.unlink(tmp.name)


@app.route('/predict/record', methods=['POST'])
def predict_record():
    """
    Predict genre from raw audio bytes (e.g. browser MediaRecorder).

    Expects the raw audio in the request body (Content-Type: audio/* or application/octet-stream).
    """
    audio_bytes = request.get_data()

    if not audio_bytes or len(audio_bytes) == 0:
        return jsonify({'error': 'No audio data received.'}), 400

    try:
        features = extract_features_from_bytes(audio_bytes)
        result = _predict(features)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Failed to process audio: {str(e)}'}), 500


if __name__ == '__main__':
   
    app.run(debug=True, host='0.0.0.0', port=5000)
