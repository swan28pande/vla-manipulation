"""Vision-language-action policy."""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


def _load(name: str, pretrained: bool):
    if pretrained:
        return AutoModel.from_pretrained(name)
    return AutoModel.from_config(AutoConfig.from_pretrained(name))


class VLAPolicy(nn.Module):
    """Encodes image + instruction, fuses with cross-attention, regresses an EE action."""

    def __init__(self, vision_model, text_model, hidden_dim=256, action_dim=7,
                 pretrained=False, freeze_encoders=True):
        super().__init__()
        self.vision = _load(vision_model, pretrained)
        self.text = _load(text_model, pretrained)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        if freeze_encoders:
            for p in list(self.vision.parameters()) + list(self.text.parameters()):
                p.requires_grad = False

        self.v_proj = nn.Linear(self.vision.config.hidden_size, hidden_dim)
        self.t_proj = nn.Linear(self.text.config.hidden_size, hidden_dim)
        self.fuse = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def tokenize(self, instructions, device):
        tok = self.tokenizer(list(instructions), padding=True, truncation=True,
                             max_length=32, return_tensors="pt")
        return {k: v.to(device) for k, v in tok.items()}

    def forward(self, images, instructions):
        """images: (B,3,H,W) float in [0,1]; instructions: list[str]. Returns (B, action_dim)."""
        v = self.v_proj(self.vision(pixel_values=images).last_hidden_state)      # (B,Nv,D)
        tok = self.tokenize(instructions, images.device)
        t = self.t_proj(self.text(**tok).last_hidden_state)                       # (B,Nt,D)
        fused, _ = self.fuse(query=t, key=v, value=v,
                             key_padding_mask=None)                               # text attends to vision
        mask = tok["attention_mask"].unsqueeze(-1).float()
        pooled = (fused * mask).sum(1) / mask.sum(1).clamp(min=1)
        out = self.head(pooled)
        # first 6 dims: EE delta, last dim: gripper logit
        return torch.cat([torch.tanh(out[:, :6]), out[:, 6:]], dim=-1)
