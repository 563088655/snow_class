import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from PIL import Image
import json


class Visualizer:
    def __init__(self, save_dir='visualization'):
        """
        初期化可視化ツール
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 類別のJSON
        with open('class_indices.json', 'r') as f:
            self.class_indices = json.load(f)
        
        # denormalize
        self.denormalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

    def plot_training_curve(self, train_losses, save_name='training_curve.png'):
        """
        draw training curve
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, save_name))
        plt.close()

    def visualize_features(self, features, labels, save_name='feature_space.png'):
        """
        use TSNE to visualize feature space
        Args:
            features: feature tensor
        """
        # downsample feature to 2D（次元削減）
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title('Feature Space Visualization (t-SNE)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.savefig(os.path.join(self.save_dir, save_name))
        plt.close()

    def visualize_predictions(self, images, predictions, true_labels=None, save_name='predictions.png'):
        """
        可視化結果
        Args:
            predictions: 予測結果
            true_labels(option)
        """
        num_images = min(len(images), 16)  # MAX for 16 images
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        axes = axes.ravel()
        
        for idx in range(num_images):
            # denormalize and convert to numpy
            img = self.denormalize(images[idx]).cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            
            # get prediction class
            pred_class = predictions[idx].argmax()
            pred_class_name = self.class_indices[str(pred_class)]
            
            #　show image
            axes[idx].imshow(img)
            if true_labels is not None:
                true_class = true_labels[idx]
                true_class_name = self.class_indices[str(true_class)]
                axes[idx].set_title(f'Pred: {pred_class_name}\nTrue: {true_class_name}')
            else:
                axes[idx].set_title(f'Pred: {pred_class_name}')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name))
        plt.close()

    def plot_confusion_matrix(self, confusion_matrix, save_name='confusion_matrix.png'):
        """
        confusion_matrix
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(os.path.join(self.save_dir, save_name))
        plt.close()

    def visualize_semi_supervised_results(self, labeled_features, unlabeled_features, 
                                        labeled_predictions, unlabeled_predictions,
                                        save_name='semi_supervised_results.png'):
        """
        半教師あり学習結果を可視化
        Args:
            labeled_features: ラベル付きデータの特徴
            unlabeled_features: ラベルなしデータの特徴
            labeled_predictions: ラベル付きデータの予測
            unlabeled_predictions: ラベルなしデータの予測
            save_name: 保存ファイル名
        """
        # PCAで次元削減
        pca = PCA(n_components=2)
        all_features = np.vstack([labeled_features, unlabeled_features])
        all_features_2d = pca.fit_transform(all_features)
        
        # ラベル付きとラベルなしデータの降次結果を分ける
        labeled_2d = all_features_2d[:len(labeled_features)]
        unlabeled_2d = all_features_2d[len(labeled_features):]
        
        plt.figure(figsize=(12, 8))
        
        # ラベル付きデータのプロット
        scatter1 = plt.scatter(labeled_2d[:, 0], labeled_2d[:, 1], 
                             c=labeled_predictions, cmap='viridis', 
                             marker='o', label='Labeled')
        
        # ラベルなしデータのプロット
        scatter2 = plt.scatter(unlabeled_2d[:, 0], unlabeled_2d[:, 1], 
                             c=unlabeled_predictions, cmap='viridis',
                             marker='x', alpha=0.5, label='Unlabeled')
        
        plt.colorbar(scatter1)
        plt.title('Semi-supervised Learning Results')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, save_name))
        plt.close()

    def visualize_snow_coverage(self, image, prediction, save_name='snow_coverage.png'):
        """
        雪の覆い予測を可視化
        Args:
            image: 入力画像
            prediction: 雪の覆い予測値（0〜1の間）
            save_name: 保存ファイル名
        """
        plt.figure(figsize=(10, 5))
        
        # 元画像を表示
        plt.subplot(1, 2, 1)
        img = self.denormalize(image).cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        # 雪の覆い予測を表示
        plt.subplot(1, 2, 2)
        plt.imshow(np.ones_like(img) * prediction, cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(label='Snow Coverage')
        plt.title(f'Predicted Snow Coverage: {prediction:.2%}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name))
        plt.close() 