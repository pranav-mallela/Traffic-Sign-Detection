import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
from inference import TSignDetector, get_prediction

app = Flask(__name__)
CORS(app)

model = TSignDetector()
file = "classifier.pt"
model.load_state_dict(torch.load(file, map_location=torch.device('cpu')))

@app.route('/upload', methods=['POST'])
def upload_frame():
    try:
        # Ensure that the content type is application/json
        if request.is_json:
            data = request.get_json()
            if 'image' in data:
                # Decode the base64 image
                image_data = base64.b64decode(data['image'].split(',')[1])
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    prediction = get_prediction(model, frame)
                    return jsonify({'prediction': prediction}), 200
                return jsonify({'error': 'Failed to process frame'}), 400
            return jsonify({'error': 'No image data provided'}), 400
        else:
            return jsonify({'error': 'Invalid content type, expected application/json'}), 415
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
