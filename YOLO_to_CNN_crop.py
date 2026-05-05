from PIL import Image
import torch
from torchvision import transforms

input_size = 64
pad = 0.25  

# ResNet normalization
resnet_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def yolo_to_cnn_crop(image: Image.Image, bbox, input_size=64, pad=0.25):
    W, H = image.size
    x1, y1, x2, y2 = bbox

    # width & height
    w = x2 - x1
    h = y2 - y1

    # padded coordinates
    new_x1 = max(0, x1 - pad * w)
    new_y1 = max(0, y1 - pad * h)
    new_x2 = min(W, x2 + pad * w)
    new_y2 = min(H, y2 + pad * h)

    # crop and resize
    crop = image.crop((new_x1, new_y1, new_x2, new_y2))
    crop = crop.resize((input_size, input_size))

    # convert to tensor + normalize for ResNet
    tensor = resnet_tfms(crop)

    return tensor
