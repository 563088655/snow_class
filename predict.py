import os
import json
import shutil
import argparse

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from model import resnet34

data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class_indict = {0: "daisy",
              1: "dandelion",
              2: "roses",
              3: "sunflowers",
              4: "tulips"}

# Visualizerクラスを追加しました
class Visualizer:
    def __init__(self):
        pass

    def visualize_snow_coverage(self, img_tensor, coverage, filename):
        # tensorをnumpy配列に変換して表示
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)  # ピクセル値を0〜1の範囲に収める
        
        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.title(f"Snow Coverage: {coverage*100:.2f}%")
        plt.axis('off')
        plt.savefig(filename)
        plt.show()


def eval(device, img):
    # create model
    model = resnet34(num_classes=5).to(device)

    # load model weights
    weights_path = "./ResNet34-best.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_dataloder(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        predict_cla = int(predict_cla)
    print("class: {:10}   prob: {:.3}".format(class_indict[predict_cla],
                                              predict[predict_cla].numpy()))

    for i in range(len(predict)):
        print("class: {:10} ---  prob: {:.3}".format(class_indict[i], predict[i].numpy()))


def read_img(img_path="../tulip.jpg"):
    # load image
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    return img


def predict(args):
    """
    Make predictions on a single image
    """
    # Load model
    net = resnet34(num_classes=args.num_classes).to(args.device)
    checkpoint = torch.load(args.weights, map_location=args.device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    
    # Load and preprocess image
    img = Image.open(args.img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(args.device)
    
    # Make prediction
    with torch.no_grad():
        category_logits, coverage_pred = net(img_tensor, return_coverage=True)
        category = category_logits.argmax(dim=1).item()
        coverage = coverage_pred.item()
    
    # Print results
    category_names = ['NO_SNOW', '雪<50%', '雪≥50%']
    print(f'類別: {category_names[category]}')
    print(f'雪の覆い値: {coverage:.2%}')
    
    # Visualize results if needed
    if args.visualize:
        visualizer = Visualizer()
        visualizer.visualize_snow_coverage(
            img_tensor[0],
            coverage,
            f'prediction_{os.path.basename(args.img_path)}'
        )

    
def batch_predict(args):
    """
    Perform batch predictions on images in a directory, save the results to a JSON file,
    and move images to their respective class folders.
    """
    # Load model
    net = resnet34(num_classes=args.num_classes).to(args.device)
    checkpoint = torch.load(args.weights, map_location=args.device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Define category names
    category_names = ['无雪', '积雪<50%', '积雪≥50%']

    results = {}
    # Iterate over images in the directory
    for img_name in os.listdir(args.img_dir):
        img_path = os.path.join(args.img_dir, img_name)
        # Check if file is an image based on extension
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            continue

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening {img_name}: {e}")
            continue

        img_tensor = transform(img).unsqueeze(0).to(args.device)

        with torch.no_grad():
            category_logits, coverage_pred = net(img_tensor, return_coverage=True)
            category = category_logits.argmax(dim=1).item()
            coverage = coverage_pred.item()

        predicted_class = category_names[category]
        results[img_name] = {"predicted_class": predicted_class, "coverage": coverage}

        # Create destination directory if not exists
        dest_dir = os.path.join(args.img_dir, predicted_class)
        os.makedirs(dest_dir, exist_ok=True)

        # Move image to destination directory
        dest_path = os.path.join(dest_dir, img_name)
        shutil.move(img_path, dest_path)
        print(f"Moved {img_name} to {dest_dir}")

    # Save results to JSON file
    results_path = os.path.join(args.img_dir, "batch_prediction_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Batch prediction results saved to {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, help='path to input image')
    parser.add_argument('--img-dir', type=str, help='path to directory of images for batch prediction')
    parser.add_argument('--weights', type=str, required=True, help='path to model weights')
    parser.add_argument('--num-classes', type=int, default=9, help='number of classes')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device to use')
    parser.add_argument('--visualize', action='store_true', help='whether to visualize results')
    args = parser.parse_args()
    
    if args.img_dir:
        batch_predict(args)
    elif args.img_path:
        predict(args)
    else:
        print("Please provide either --img-dir for batch prediction or --img-path for single image prediction.")