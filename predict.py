import os
import argparse
import json
import torch
from torchvision import models, transforms
from PIL import Image

def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image', type=str, help='Path to image file for prediction.')
    parser.add_argument('--data_directory', type=str, help='Path to directory containing images for prediction.')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file as str.', required=True)
    parser.add_argument('--top_k', type=int, help='Choose top K matches as int.', default=5)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default=False, action="store_true", dest="gpu")

    return parser.parse_args()

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image_path):
    img = Image.open(image_path)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img = transform(img)
    img = img.unsqueeze(0)

    return img

def predict(image_tensor, model, topk=5):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        ps = torch.exp(output)
        top_probs, top_labels = ps.topk(topk, dim=1)

    return top_probs[0].tolist(), top_labels[0].tolist()


def print_probability(probs, labels, class_to_name):
    for i, (prob, label) in enumerate(zip(probs, labels), 1):
        # Adjust the label to match the indexing in cat_to_name
        adjusted_label = str(label + 1)
        flower_name = class_to_name.get(adjusted_label, f'Unknown Label ({adjusted_label})')
        print(f"Rank {i}: Flower: {flower_name}, Likelihood: {prob * 100:.2f}%")

def predict_on_directory(directory, checkpoint, top_k, category_names, gpu):
    model = load_checkpoint(checkpoint)
    model.to(torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu"))
    model.eval()

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(directory, filename)
            predict_on_image(image_path, model, top_k, cat_to_name)

def predict_on_image(image, checkpoint, top_k, category_names, gpu):
    model = load_checkpoint(checkpoint)
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    image_tensor = process_image(image)
    image_tensor = image_tensor.to(device)

    top_probs, top_labels = predict(image_tensor, model, top_k)

    print(f"Predictions for {image}:")
    print_probability(top_probs, top_labels, cat_to_name)

def main():
    args = arg_parser()

    if args.image:
        predict_on_image(args.image, args.checkpoint, args.top_k, args.category_names, args.gpu)
    elif args.data_directory:
        predict_on_directory(args.data_directory, args.checkpoint, args.top_k, args.category_names, args.gpu)
    else:
        print("Please provide either --image or --data_directory for prediction.")

if __name__ == '__main__':
    main()
