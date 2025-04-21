import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import argparse
from model import resnet34
from visualization import Visualizer
import numpy as np

# 「RuntimeError：CUDA error。。。For debugging consider passing CUDA_LAUNCH_BLOCKING=1.」
# があるときに、「os.environ['CUDA_LAUNCH_BLOCKING'] = '1'」を加えると動けます。
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Preprocess, data augmentation for training.
# The val need to be CenterCrop to ensure the consistency.
data_transform = {
    "train": transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(192, 192),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# 「ArgumentParser, プログラム実行時にコマンドラインで引数を受け取る処理を簡単に実装できる標準ライブラリです。」
# 紹介はこちらです：https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0。
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    # 最初は３クラスで、後に、クラス数を増やすことができるようにする
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--version', type=int, default=2)
    # you can try this dataset out for supervised learning part
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str, default="data_set/harmo")

    # for this task, you should modify this part 
    parser.add_argument('--weights', type=str, default='checkpoint/resnet34-pre.pth',
                        help='initial weights path')
    # head以外のweightを凍結するかどうか
    # 指定しないと、デフォルトはFalse（トレーニングを行う）
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cpu', help='device id (Expected one of cpu, cuda(for Nvidia), mps(for Apple Silicon)')

    # Semi-supervised learning parameters
    parser.add_argument('--semi-supervised', action='store_true', help='Enable semi-supervised learning')
    parser.add_argument('--unlabeled-data-path', type=str, default="data_set/unlabeled",
                        help='Path to unlabeled data for semi-supervised learning')
    parser.add_argument('--consistency-weight', type=float, default=0.1,
                        help='Weight for consistency loss in semi-supervised learning')
    
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--vis-interval', type=int, default=5, help='Visualization interval (epochs)')
    
    args = parser.parse_args()
    return args


def train_model(net, args, train_loader, unlabeled_loader=None):
    """
    Train the model with semi-supervised learning
    Args:
        net: the neural network model
        args: command line arguments
        train_loader: data loader for labeled data
        unlabeled_loader: data loader for unlabeled data
    """
    net.to(args.device)
    
    # Initialize visualizer if needed
    visualizer = Visualizer() if args.visualize else None
    
    # Define loss functions
    classification_loss = nn.CrossEntropyLoss()
    consistency_loss = nn.MSELoss()
    coverage_loss = nn.MSELoss()
    
    # Construct optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    train_steps = len(train_loader)
    if unlabeled_loader:
        unlabeled_steps = len(unlabeled_loader)
    
    # Initialize lists for visualization
    train_losses = []
    all_features = []
    all_labels = []
    all_predictions = []
    all_coverages = []  # This will store coverage predictions
    
    # Define data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        
        # Training on labeled data
        for step, data in enumerate(train_bar):
            images, labels = data
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            optimizer.zero_grad()
            
            # 1. Supervised learning part
            category_logits, coverage_preds = net(images, return_coverage=True)
            
            # Calculate classification loss
            category_loss = classification_loss(category_logits, labels)
            
            # 2. Semi-supervised learning part
            if args.semi_supervised and unlabeled_loader:
                unlabeled_iter = iter(unlabeled_loader)  # Initialize iterator
                try:
                    unlabeled_images = next(unlabeled_iter)
                    unlabeled_images = unlabeled_images[0].to(args.device)
                    
                    # Loop through the batch and apply ToPILImage to each individual image
                    unlabeled_aug1_list = [transforms.ToPILImage()(img) for img in unlabeled_images]
                    unlabeled_aug2_list = [transforms.ToPILImage()(img) for img in unlabeled_images]
                    
                    # Apply the transformation
                    unlabeled_aug1 = [train_transform(img) for img in unlabeled_aug1_list]
                    unlabeled_aug2 = [train_transform(img) for img in unlabeled_aug2_list]
                    
                    # Get features
                    features1 = net.get_features(torch.stack(unlabeled_aug1).to(args.device))
                    features2 = net.get_features(torch.stack(unlabeled_aug2).to(args.device))
                    
                    # Calculate consistency loss
                    consistency = consistency_loss(features1, features2)
                    
                    # Training with pseudo labels
                    with torch.no_grad():
                        # Predict category and snow coverage
                        pseudo_category_logits, pseudo_coverages = net(torch.stack(unlabeled_aug1).to(args.device), return_coverage=True)
                        pseudo_categories = pseudo_category_logits.argmax(dim=1)
                    
                    # Calculate pseudo-label loss
                    pseudo_category_loss = classification_loss(pseudo_category_logits, pseudo_categories)
                    pseudo_coverage_loss = coverage_loss(
                        net(torch.stack(unlabeled_aug2).to(args.device), return_coverage=True)[1],
                        pseudo_coverages
                    )
                    
                    # Total loss
                    loss = category_loss + \
                           args.consistency_weight * (consistency + pseudo_category_loss + pseudo_coverage_loss)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    loss = category_loss
            else:
                loss = category_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(
                epoch + 1, args.epochs, loss)
            
            # Collect data for visualization
            if args.visualize and step % 10 == 0:
                with torch.no_grad():
                    features = net.get_features(images)
                    all_features.append(features.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    all_predictions.append(category_logits.argmax(dim=1).cpu().numpy())
                    all_coverages.append(coverage_preds.cpu().numpy())  # Collect snow coverage predictions
        
        # Calculate epoch loss
        epoch_loss = running_loss / train_steps
        train_losses.append(epoch_loss)
        print('[epoch %d] train_loss: %.3f' % (epoch + 1, epoch_loss))
        
        # Update learning rate
        scheduler.step(epoch_loss)
        
        # Save checkpoint
        save_path = 'checkpoint/ResNet34-{}-v{}.pth'.format(epoch, args.num_classes)
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, save_path)
        print(f"Checkpoint saved at {save_path}")
        
        # Visualization
        if args.visualize and (epoch + 1) % args.vis_interval == 0:
            # Plot training curve
            visualizer.plot_training_curve(train_losses)
            
            # Visualize feature space
            if all_features:
                features_np = np.vstack(all_features)
                labels_np = np.concatenate(all_labels)
                predictions_np = np.concatenate(all_predictions)
                coverages_np = np.concatenate(all_coverages)  # Corrected: Now we concatenate coverages

                visualizer.visualize_features(features_np, labels_np)
                
                if args.semi_supervised and unlabeled_loader:
                    # Get unlabeled features and predictions
                    unlabeled_features = []
                    unlabeled_predictions = []
                    unlabeled_coverages = []
                    
                    for unlabeled_images, _ in unlabeled_loader:
                        unlabeled_images = unlabeled_images.to(args.device)
                        with torch.no_grad():
                            features = net.get_features(unlabeled_images)
                            predictions, coverages = net(unlabeled_images, return_coverage=True)
                            
                            unlabeled_features.append(features.cpu().numpy())
                            unlabeled_predictions.append(predictions.argmax(dim=1).cpu().numpy())
                            unlabeled_coverages.append(coverages.cpu().numpy())
                    
                    unlabeled_features = np.vstack(unlabeled_features)
                    unlabeled_predictions = np.concatenate(unlabeled_predictions)
                    unlabeled_coverages = np.concatenate(unlabeled_coverages)

                    visualizer.visualize_semi_supervised_results(
                        features_np, unlabeled_features,
                        predictions_np, unlabeled_predictions
                    )
                    
                    # Visualize snow coverage
                    for i in range(min(5, len(unlabeled_images))):
                        visualizer.visualize_snow_coverage(
                            unlabeled_images[i],
                            unlabeled_coverages[i],
                            f'snow_coverage_epoch{epoch}_sample{i}.png'
                        )
            
            # Reset visualization data
            all_features = []
            all_labels = []
            all_predictions = []
            all_coverages = []  # Clear the coverage data
    
    print('Finished Training')


def get_loader(args, is_unlabeled=False):
    """
    Get data loader for labeled or unlabeled data
    Args:
        args: command line arguments
        is_unlabeled: whether to load unlabeled data
    Returns:
        DataLoader object
    """
    print("using {} device.".format(args.device))
    
    if is_unlabeled:
        image_path = args.unlabeled_data_path
        transform = data_transform["train"]  # Use training transform for unlabeled data
    else:
        image_path = args.data_path
        transform = data_transform["train"]
    
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    
    dataset = datasets.ImageFolder(root=image_path, transform=transform)
    num_samples = len(dataset)
    
    if not is_unlabeled:
        # Save class indices for labeled data
        class_list = dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in class_list.items())
        json_str = json.dumps(cla_dict, indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)
    
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=nw
    )
    
    print("using {} images for {}".format(num_samples, "unlabeled" if is_unlabeled else "training"))
    return loader

def load_model(args):
    """
    Load and initialize the model
    If args.weights is provided, load the weights
    """
    # Initialize ResNet model with the correct number of classes
    net = resnet34(num_classes=args.num_classes)

    if args.weights:
        # Modify this line to fix the error with legacy .tar format
        state_dict = torch.load(args.weights, map_location='cpu', weights_only=False)  # Use weights_only=False
        net.load_state_dict(state_dict, strict=False)  # Allow for missing layers

        if args.num_classes == 9:
            # Adjust the classification head to 9 classes if necessary
            in_features = net.category_head[0].in_features  # Get input features for the category head
            net.category_head = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Linear(256, 9)  # Change to 9 classes
            )
            print("Model adjusted for 9 classes for semi-supervised training.")

    if args.freeze_layers:
        for name, param in net.named_parameters():
            # Freeze all layers except the final classification head
            if 'category_head' not in name:
                param.requires_grad = False
                print(f"Freezing layer: {name}")
            else:
                param.requires_grad = True  # Ensure the classification head is trainable

    return net

if __name__ == '__main__':
    args = parse_args()
    train_loader = get_loader(args)
    unlabeled_loader = None
    
    if args.semi_supervised:
        unlabeled_loader = get_loader(args, is_unlabeled=True)
    
    # Load and train model
    net = load_model(args)
    train_model(net=net, args=args, train_loader=train_loader, unlabeled_loader=unlabeled_loader)
