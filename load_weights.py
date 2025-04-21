import os
import torch
import torch.nn as nn
from model import resnet34
import urllib.request

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # download pretrain weights
    url = "https://download.pytorch.org/models/resnet34-333f7ec4.pth"
    model_weight_path = "checkpoint/resnet34-pre.pth"
    if not os.path.exists(model_weight_path):
        print(f"Downloading resnet34 to {model_weight_path}...")
        urllib.request.urlretrieve(url, model_weight_path)

    # load pretrain weights
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    # there are two options to load the pretrain weights and change the fc layer to 5 classes
    # option1
    net = resnet34()
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)

    # option2
    # net = resnet34(num_classes=5)
    # pre_weights = torch.load(model_weight_path, map_location=device, weights_only=False)
    # del_key = []
    # for key, _ in pre_weights.items():
    #     if "fc" in key:
    #         del_key.append(key)
    
    # for key in del_key:
    #     del pre_weights[key]

    # del the fc layer
    
    # missing_keys, unexpected_keys = net.load_state_dict(pre_weights, strict=False)
    # print("[missing_keys]:", *missing_keys, sep="\n")
    # print("[unexpected_keys]:", *unexpected_keys, sep="\n")

    # if you wanna check the pre_weights
    # for key in pre_weights.keys():
    #     print(f"{key}: {pre_weights[key].shape}")


if __name__ == '__main__':
    main()
