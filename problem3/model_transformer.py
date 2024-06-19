# Course: Computer Vision and Artificial Intelligence for Autonomous Cars, ETH Zurich
# Material for Project 2
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch


import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout
        )

    def forward(self, x):
        # x should have shape [seq_len, batch_size, embed_dim]
        x = x.permute(2, 0, 1).contiguous()  # transpose for transformer layer
        x = self.transformer_encoder_layer(x)
        return x.permute(1, 2, 0).contiguous() # permutate again 
        

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.__dict__.update(config)

        # Encoder
        channel_in = self.channel_in
        self.set_abstraction = nn.ModuleList()
        self.transformers = nn.ModuleList()

        for k in range(len(self.npoint)):
            mlps = [channel_in] + self.mlps[k]
            npoint = self.npoint[k] if self.npoint[k]!=-1 else None
            self.set_abstraction.append(
                PointnetSAModule(
                    npoint=npoint,
                    radius=self.radius[k],
                    nsample=self.nsample[k],
                    mlp=mlps,
                    use_xyz=True,
                    bn=False
                )
            )
            channel_in = mlps[-1]

            # Add Transformer encoder for each set abstraction layer
            num_heads = min(channel_in // 64, 8)  # Adjust the number of heads based on channel size
            self.transformers.append(TransformerEncoderLayer(embed_dim=channel_in, num_heads=num_heads))


        # Classification head
        cls_layers = []
        pre_channel = channel_in
        for k in range(len(self.cls_fc)):
            cls_layers.extend([
                nn.Conv1d(pre_channel, self.cls_fc[k], kernel_size=1),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.cls_fc[k]
        cls_layers.extend([
            nn.Conv1d(pre_channel, 1, kernel_size=1),
            nn.Sigmoid()
        ])
        self.cls_layers = nn.Sequential(*cls_layers)

        # Regression head
        det_layers = []
        pre_channel = channel_in
        for k in range(len(self.reg_fc)):
            det_layers.extend([
                nn.Conv1d(pre_channel, self.reg_fc[k], kernel_size=1),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.reg_fc[k]
        det_layers.append(nn.Conv1d(pre_channel, 7, kernel_size=1))
        self.det_layers = nn.Sequential(*det_layers)

    def forward(self, x):
        # print("Input size:", x.size())
        xyz = x[..., 0:3].contiguous()
        feat = x[..., 3:].transpose(1, 2).contiguous()

        for i, (layer, transformer) in enumerate(zip(self.set_abstraction, self.transformers)):
            xyz, feat = layer(xyz, feat)
            # print(f"Set Abstraction Layer {i + 1} output size:", feat.size())

            # Apply Transformer encoder
            feat = transformer(feat)
            # print(f"Transformer Layer {i + 1} output size:", feat.size())


        # Processing through classification head
        cls_output = feat
        for i, layer in enumerate(self.cls_layers):
            cls_output = layer(cls_output)
            # print(f"Classification Head Layer {i + 1} output size:", cls_output.size())

        pred_class = cls_output.squeeze(dim=-1)
        # print("Final Classification Output size:", pred_class.size())

        # Processing through regression head
        reg_output = feat
        for i, layer in enumerate(self.det_layers):
            reg_output = layer(reg_output)
            # print(f"Regression Head Layer {i + 1} output size:", reg_output.size())

        pred_box = reg_output.squeeze(dim=-1)
        # print("Final Regression Output size:", pred_box.size())

        return pred_box, pred_class