import torch
from torch import nn
from torch.nn import functional as F
import math
from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class Wav2Lip_512_normal(nn.Module):
    def __init__(self, audio_encoder=None):
        super(Wav2Lip_512_normal, self).__init__()
        
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 6, kernel_size=7, stride=1, padding=3, act="leaky"),
            Conv2d(6, 6, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(6, 6, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),
            
            nn.Sequential(Conv2d(6, 6, kernel_size=3, stride=2, padding=1, act="leaky"), # 512, 512
            Conv2d(6, 6, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(6, 6, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),
            
            nn.Sequential(Conv2d(6, 8, kernel_size=3, stride=2, padding=1, act="leaky"), # 256, 256
            Conv2d(8, 8, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(8, 8, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")), 
            
            nn.Sequential(Conv2d(8, 16, kernel_size=3, stride=2, padding=1, act="leaky"), # 128, 128
            Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1, act="leaky"), # 64, 64
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1, act="leaky"),    # 32, 32
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1, act="leaky"),   # 16, 16
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1, act="leaky"),# 8, 8
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1, act="leaky"),     # 4, 4
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),),

            ###################
            # Modified blocks
            ##################
            nn.Sequential(Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, act="leaky"),       # 2, 2
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),])
            ##################

        if audio_encoder is None:
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
        else:
            self.audio_encoder = audio_encoder

        for p in self.audio_encoder.parameters():
            p.requires_grad = False

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, act="leaky"),),  #

            ###################
            # Modified blocks
            ##################
            nn.Sequential(Conv2dTranspose(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"), #  + 1024
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),), #

            nn.Sequential(Conv2dTranspose(1536, 1024, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),),

            nn.Sequential(Conv2dTranspose(1280, 768, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
            Conv2d(768, 768, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(768, 768, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),),

            nn.Sequential(Conv2dTranspose(896, 512, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),),
            ##################

            nn.Sequential(Conv2dTranspose(576, 256, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),),

            nn.Sequential(Conv2dTranspose(288, 128, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),),

            nn.Sequential(Conv2dTranspose(144, 80, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
            Conv2d(80, 80, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(80, 80, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),),
            
            nn.Sequential(Conv2dTranspose(88, 64, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),),
            
            nn.Sequential(Conv2dTranspose(70, 50, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
            Conv2d(50, 50, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(50, 50, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),),]) 

        self.output_block = nn.Sequential(Conv2d(56, 32, kernel_size=3, stride=1, padding=1, act="leaky"),
            Conv2d(32, 16, kernel_size=3, stride=1, padding=1, act="leaky"),
            nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh())

    def freeze_audio_encoder(self):
        for p in self.audio_encoder.parameters():
            p.requires_grad = False
    
    def forward(self, audio_sequences, face_sequences):


        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        
        audio_embedding = self.audio_encoder(audio_sequences)
        audio_embedding = audio_embedding.detach()
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        
        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)
        
        x = audio_embedding
        cnt = 0
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            feats.pop()
            cnt += 1

        x = self.output_block(x)
        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x
            
        return outputs
    
    


class Wav2Lip_disc_qual_512(nn.Module):
    def __init__(self):
        super(Wav2Lip_disc_qual_512, self).__init__()
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(nonorm_Conv2d(3, 8, kernel_size=7, stride=1, padding=3)), # 512, 512
            
            nn.Sequential(nonorm_Conv2d(8, 16, kernel_size=5, stride=2, padding=2), # 256, 256
            nonorm_Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
            nonorm_Conv2d(16, 16, kernel_size=5, stride=1, padding=2)),
            
            nn.Sequential(nonorm_Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # 128, 128
            nonorm_Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nonorm_Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            
            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # 64, 64
            nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),    # 32, 32
            nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),   # 16, 16
            nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),    # 8, 8
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),     # 4, 4
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),),

           # Modified Blocks
            nn.Sequential(nonorm_Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),     # 2, 2
            nonorm_Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),),

            nn.Sequential(nonorm_Conv2d(1024, 1024, kernel_size=2, stride=1, padding=0, norm=False),     # 1, 1
            nonorm_Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, norm=False)),])


      

        self.binary_pred = nn.Sequential(nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0),
                                         nn.Sigmoid())
        self.label_noise = .0
        self.bce_loss = nn.BCELoss()

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2)//2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        # TODO: change back to cuda
        # false_pred_loss = F.binary_cross_entropy_with_logits(torch.clamp(self.binary_pred(false_feats).view(len(false_feats), -1), min=-10, max=10), torch.ones((len(false_feats), 1)).cuda())
        pred = self.binary_pred(false_feats).view(len(false_feats), -1)
        target = torch.ones((len(false_feats), 1)).cuda()
        false_pred_loss = F.binary_cross_entropy(pred, target)
        # false_pred_loss = torch.mean(torch.square(pred - target))
        # false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1),
                                        # torch.ones((len(false_feats), 1)).cpu())

        return false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)