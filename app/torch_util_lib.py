import io
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
from PIL import Image
import os


def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(3),
                                    transforms.Resize((28,28)),
                                    transforms.ToTensor()])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

num_classes = 10
print(os.getcwd())
PATH = "app/model_full.pt"
model = torch.load(PATH,map_location='cpu')
model.eval()

def get_prediction(image_tensor):
    images = image_tensor
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    return predicted