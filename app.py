import re
from final import transform, final
from flask import Flask, request, jsonify

import io
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

app.config['DEBUG'] = True
app.config['ENV'] = 'development' 


ALLOWED_EXTENSTIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSTIONS


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename =="":
            return jsonify({'error': 'No file sent'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported'})
        try:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            
            # Apply transforms
            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            img = transform(img).unsqueeze(0)
            caption = final(img)
            return jsonify({'caption': caption})

        except:
            return jsonify({'error': 'error during processing file'})

    return jsonify({'res' : 1})

