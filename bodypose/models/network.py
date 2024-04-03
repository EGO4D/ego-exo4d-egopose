import torch
import torch.nn as nn
from IPython import embed
import math
from utils import utils_transform


nn.Module.dump_patches = True




class EgoExo4D(nn.Module):
    def __init__(self, input_dim, output_dim, num_layer, embed_dim, nhead, device,opt):
        super(EgoExo4D, self).__init__()

        self.linear_embedding = nn.Linear(input_dim,embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)        

        out_dim =51

        self.stabilizer = nn.Sequential(
                        nn.Linear(embed_dim, 256),
                        nn.ReLU(),
                        nn.Linear(256, out_dim)
        )
        self.joint_rotation_decoder = nn.Sequential(
                            nn.Linear(embed_dim, 256),
                            nn.ReLU(),
                            nn.Linear(256, 126)
            )


    def forward(self, input_tensor,image=None, do_fk = True):
        input_tensor = input_tensor.reshape(input_tensor.shape[0],input_tensor.shape[1],-1)
        x = self.linear_embedding(input_tensor)
        x = x.permute(1,0,2)
        x = self.transformer_encoder(x)
        x = x.permute(1,0,2)[:, -1]
        global_orientation = self.stabilizer(x)
        return global_orientation
