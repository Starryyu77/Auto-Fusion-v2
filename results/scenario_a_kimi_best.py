import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoFusionLayer(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.visual_dim = 1024
        self.text_dim = 768
        self.hidden_dim = 256
        
        # Visual projection: flatten spatial then project
        self.visual_proj = nn.Sequential(
            nn.Linear(self.visual_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Attention pooling for visual
        self.visual_attn_pool = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Attention pooling for text
        self.text_attn_pool = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Cross-modal fusion MLP
        self.fusion_net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        # Output classifier
        self.classifier = nn.Linear(128, 4)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, visual, text):
        B = visual.size(0)
        
        # Process visual: [B, 256, 1024]
        v_flat = visual.view(B * 256, self.visual_dim)  # [B*256, 1024]
        v_proj = self.visual_proj(v_flat)  # [B*256, hidden]
        v_proj = v_proj.view(B, 256, self.hidden_dim)  # [B, 256, hidden]
        
        # Process text: [B, 77, 768]
        t_flat = text.view(B * 77, self.text_dim)  # [B*77, 768]
        t_proj = self.text_proj(t_flat)  # [B*77, hidden]
        t_proj = t_proj.view(B, 77, self.hidden_dim)  # [B, 77, hidden]
        
        # Attention-based pooling for visual
        v_attn_weights = self.visual_attn_pool(v_proj)  # [B, 256, 1]
        v_pooled = (v_proj * v_attn_weights).sum(dim=1)  # [B, hidden]
        
        # Attention-based pooling for text
        t_attn_weights = self.text_attn_pool(t_proj)  # [B, 77, 1]
        t_pooled = (t_proj * t_attn_weights).sum(dim=1)  # [B, hidden]
        
        # Concatenate features
        fused = torch.cat([v_pooled, t_pooled], dim=-1)  # [B, hidden*2]
        
        # Fusion network
        features = self.fusion_net(fused)  # [B, 128]
        
        # Classification
        logits = self.classifier(features)  # [B, 4]
        
        return logits