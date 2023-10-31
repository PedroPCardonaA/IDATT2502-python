from flask import Flask, request
import packages.models as models
import torch
from torch import nn
import io
import numpy as np
from PIL import Image
from torchvision import transforms
app = Flask(__name__)

# Sample data (in-memory storage)
model = models.garbage_classifier_5L_attention(input_shape=3, hidden_units=64, output_shape=6)
model.load_state_dict(torch.load("models/model5l.pth"))
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# PUT endpoint
@app.route('/data/<key>', methods=['PUT'])
def update_data(key):
    if request.method == 'PUT':
        if key not in data:
            return "Key not found", 404

        # Update the value for the provided key
        data[key] = request.data
        return f"Value updated for key '{key}'"

# Post image
@app.route('/image', methods=['POST'])
def post_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in the request", 400

        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        # Check if the file is an image (you can add more comprehensive checks)
        if file.content_type.split('/')[0] != 'image':
            return "File is not an image", 400

        # Read the image data into a PIL Image
        image = Image.open(io.BytesIO(file.read()))

        # Convert the PIL Image to a NumPy array
        image_array = np.array(image)
        image = Image.fromarray(image_array)
        image_tensor = preprocess(image)
        model.eval()
        with torch.no_grad():
            result = model(image_tensor.unsqueeze(0))
            print(result)
            print(result.shape)
            percent = nn.functional.softmax(result, dim=1)[0] * 100
            return str(result.argmax(1).item()), str(percent[result.argmax(1).item()].item())

        #Convert the NumPy array to a PyTorch Tensor


if __name__ == '__main__':
    app.run(debug=True,host="10.24.6.179", port=8080)
