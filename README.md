# Snow Classification Project

このプロジェクトは、ResNetモデルを使用して積雪画像の分類タスクを行います。半教師あり学習方法を採用しており、まずは教師ありの三分類を行い、その後、各クラス内でより細かな予測を行います。

## Project Structure

- `model.py`: ResNetモデルの定義を含み、複数のResNetバリアント（ResNet18, 34, 50, 101など）をサポートします
- `train.py`: トレーニングスクリプトで、データの読み込み、モデルのトレーニングなどの主要機能を含みます
- `predict.py`: 単一画像予測スクリプト
- `load_weights.py`: モデルの重みをロードするツール
- `data_set/`: データセットディレクトリ
  - `harmo/`: 3つのクラスを含む積雪画像（未公開されましたので、こちらで削除いたしました。）
    - No Snow
    - Snow Coverage <50%
    - Snow Coverage ≥50%

## Environment Requirements

- Python 3.x
- PyTorch
- torchvision
- tqdm
- CUDA（or mps, for acceleration）

## Fast to use it（Example for Apple Silicon）

1. dataset：
   - データセットをクラス別に`data_set/harmo`置いといてください。

2. Supervised learning part：
   ```bash
   python train.py --num_classes 3 --vis-interval 5 --device mps --weights checkpoint/resnet34-pre.pth
   ```

3. Semi-supervised learning part：
    ```bash
    python train.py --num_classes 9 --vis-interval 5 --device mps --weights checkpoint/resnet34-XX.pth
   ```
   Note:
   こちらの`checkpoint/resnet34-XX.pth`は、前の教師あり学習で得られたweightを使った方がいいです。
   例えば
    ```bash
    python train.py --num_classes 9 --vis-interval 5 --device mps --weights checkpoint/ResNet34-6-v3.pth
   ```

4. Prediction：
   - Single image prediction：
     ```bash
     python predict.py --weights [path-to-weight] --img-path [path-to-image]　--device mps --visualize
     ```
     あるいは
     ```bash
     python predict.py --weights [path-to-weight] --img-path [path-to-image]　--device mps
     ```
     Note:
      --img-dir for batch prediction or --img-path for single image prediction

## Model Description

このプロジェクトでは、ResNet34をベースモデルとして使用し、transfer learningを使いました：
1.	Load the pre-trained weights
2.	Replace the final fully connected layer to fit the 3-class classification task
3.	Fine-tune the model

## traning.py　での parse_args 関数の説明

1.	--num_classes: モデルの分類クラス数を指定します（デフォルトは3クラス）。
2.	--epochs: トレーニングのエポック数（デフォルトは10）。
3.	--batch-size: バッチサイズ（デフォルトは32）。
4.	--lr: 学習率（デフォルトは1e-4）。
5.	--wd: 重み減衰（デフォルトは5e-2）。
6.	--version: モデルのバージョン（デフォルトは2）。
7.	--data-path: トレーニングデータのパス（デフォルトはdata_set/harmo）。
8.	--weights: 初期重みのパス（デフォルトはcheckpoint/resnet34-pre.pth）。
9.	--freeze-layers: 最後の層以外を凍結するかどうか（デフォルトはFalse）。
10.	--device: 使用するデバイス（CPU、CUDA、MPS）。
11.	--semi-supervised: 半教師あり学習を有効にするかどうか（オプション）。
12.	--unlabeled-data-path: 半教師あり学習用のラベルなしデータのパス（オプション）。
13.	--consistency-weight: 一貫性損失の重み（オプション）。
14.	--visualize: 可視化を有効にするかどうか（オプション）。
15.	--vis-interval: 可視化の間隔（エポック単位）。

## Notes

- トレーニングで、学習率やバッチサイズなどのハイパーパラメータを調整できます。
- モデルの重みは`checkpoint`に保存されます。