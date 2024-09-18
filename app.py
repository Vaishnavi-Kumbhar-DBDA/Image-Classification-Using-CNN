from flask import Flask, request, render_template, redirect, url_for
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np


# Load the trained model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 11 * 11, 2),
        )

    def forward(self, x):
        return self.model(x)


model = CNN()
model.load_state_dict(torch.load('/home/vaishnavi/Pycharm/image/model.pth'))
model.eval()

# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# Define Flask app
app = Flask(__name__)

# Define class names
classes = ["Daisy", "Dandelion"]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        image = Image.open(file)
        image_tensor = transform(image)
        img = image_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)
            pred_index = outputs.argmax(1)
            pred_class = classes[pred_index.item()]

        return render_template('result.html', prediction=pred_class)


if __name__ == '__main__':
    app.run(debug=True)
