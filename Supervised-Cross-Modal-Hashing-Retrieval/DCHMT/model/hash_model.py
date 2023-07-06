import os
import torch
import logging
import torch.nn as nn
import numpy as np
from typing import Union

from model.model import build_model
from utils import get_logger, get_summary_writer

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class LinearHash(nn.Module):

    def __init__(self, inputDim=2048, outputDim=64):
        super(LinearHash, self).__init__()
        self.fc = nn.Linear(inputDim, outputDim)
        self.fc.apply(weights_init_kaiming)
        self.drop_out = nn.Dropout(p=0.2)
    
    def forward(self, data):
        result = self.fc(data)
        return torch.tanh(self.drop_out(result))


class HashLayer(nn.Module):

    LINEAR_EMBED = 128
    SIGMOID_ALPH = 10
    def __init__(self, inputDim=2048, outputDim=64):
        
        super(HashLayer, self).__init__()
        self.fc = nn.Linear(inputDim, self.LINEAR_EMBED)
        self.fc.apply(weights_init_kaiming)
        self.hash_list = nn.ModuleList([nn.Linear(self.LINEAR_EMBED, 2) for _ in range(outputDim)])
        for item in self.hash_list:
            item.apply(weights_init_kaiming)
    
    def forward(self, data):
        
        embed = self.fc(data)
        embed = torch.relu(embed)

        softmax_list = [torch.softmax(item(embed), dim=-1) for item in self.hash_list]

        return softmax_list

class HashLayer_easy_logic(nn.Module):

    LINEAR_EMBED = 128
    SIGMOID_ALPH = 10
    def __init__(self, inputDim=2048, outputDim=64):
        
        super(HashLayer, self).__init__()
        self.bit = outputDim
        self.fc = nn.Linear(inputDim, outputDim * 2)
        self.fc.apply(weights_init_kaiming)
        for item in self.hash_list:
            item.apply(weights_init_kaiming)
    
    def forward(self, data):
        
        embed = self.fc(data)
        softmax_list = embed.view(embed.shape[0], self.bit, 2)

        softmax_list = torch.softmax(softmax_list, dim=-1)

        return softmax_list


class DCMHT(nn.Module):

    def __init__(self, 
                outputDim=64, 
                clipPath="./ViT-B-32.pt", 
                writer=None, 
                saveDir="./result/log", 
                logger: logging.Logger=None, 
                is_train=True,
                linear=False):
        super(DCMHT, self).__init__()
        os.makedirs(saveDir, exist_ok=True)
        self.logger = logger if logger is not None else get_logger(os.path.join(saveDir, "train.log" if is_train else "test.log"))
        self.writer = writer if writer is not None and is_train else get_summary_writer(os.path.join(saveDir, "tensorboard"))
        embedDim, self.clip = self.load_clip(clipPath)
        # if is_train:
        #     self.clip.eval()
        # print("start freezen")
        # self.freezen()
        self.image_hash =  LinearHash(inputDim=embedDim, outputDim=outputDim) if linear else HashLayer(inputDim=embedDim, outputDim=outputDim)
        self.text_hash = LinearHash(inputDim=embedDim, outputDim=outputDim) if linear else HashLayer(inputDim=embedDim, outputDim=outputDim)
        # print(self.image_hash)
        # print(self.text_hash)

    def freezen(self):
        for name, param in self.clip.named_parameters():
            # print(name)
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                                        or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                # print("1")
                continue
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= 12:
                    # print("2")
                    continue
            if name.find("conv2.") == 0:
                # print("3")
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    def load_clip(self, clipPath: str) -> tuple:
        try:
            model = torch.jit.load(clipPath, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(clipPath, map_location="cpu")
        
        return state_dict["text_projection"].shape[1], build_model(state_dict)

    def encode_image(self, image):

        image_embed = self.clip.encode_image(image)
        image_embed = self.image_hash(image_embed)

        return image_embed
    
    def eval(self):
        self.image_hash.eval()
        self.text_hash.eval()
        # self.clip.eval()

    def train(self):
        self.image_hash.train()
        self.text_hash.train()
    
    def encode_text(self, text):

        text_embed = self.clip.encode_text(text)
        text_embed = self.text_hash(text_embed)

        return text_embed

    def forward(self, image, text):
        return self.encode_image(image), self.encode_text(text)

