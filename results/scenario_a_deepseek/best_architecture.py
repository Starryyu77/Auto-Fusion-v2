import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoFusionLayer(nn.Module):
    def __init__(self, input_dims=None):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU()
        )
        self.text_proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, visual, text):
        # Process visual features [B, 256, 1024] -> [B, 1024]
        visual = visual.mean(dim=1)
        visual = self.visual_proj(visual)  # [B, 256]
        
        # Process text features [B, 77, 768] -> [B, 768]
        text = text.mean(dim=1)
        text = self.text_proj(text)  # [B, 256]
        
        # Concatenate and fuse
        fused = torch.cat([visual, text], dim=1)  # [B, 512]
        return self.fusion(fused)  # [B, 4]