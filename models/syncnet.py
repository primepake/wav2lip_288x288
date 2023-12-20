import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2d

class SyncNet_color(nn.Module):
    def __init__(self):
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(

            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)


        return audio_embedding, face_embedding
    
    
class SyncNet_color_384(nn.Module):
    def __init__(self):
        super(SyncNet_color_384, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 16, kernel_size=(7, 7), stride=1, padding=3, act="leaky"), # 192, 384

            Conv2d(16, 32, kernel_size=5, stride=(1, 2), padding=1, act="leaky"), # 192, 192
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),  

            Conv2d(32, 64, kernel_size=3, stride=2, padding=1, act="leaky"), # 96, 96
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1, act="leaky"), # 48, 48
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
    
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1, act="leaky"), # 24, 24
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),  

            ###################
            # Modified blocks
            ##################
            Conv2d(256, 512, kernel_size=3, stride=2, padding=1, act="leaky"),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),  # 12, 12

            Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, act="leaky"),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),  # 6, 6

            Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, act="leaky"), # 3, 3
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0, act="leaky"),
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, act="relu")) # 1, 1
            ##################

        # print(summary(self.face_encoder, (15, 96, 192)), act="relu")

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1, act="leaky"),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1, act="leaky"),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1, act="leaky"),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1, act="leaky"),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),

            ###################
            # Modified blocks
            ##################
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1, act="leaky"),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),

            Conv2d(512, 1024, kernel_size=3, stride=1, padding=0, act="relu"),
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, act="relu"))
            ##################

        # print(summary(self.audio_encoder, (1, 80, 16)))

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)


        return audio_embedding, face_embedding

    def audio_forward(self, audio_sequences):
        return self.audio_encoder(audio_sequences)
