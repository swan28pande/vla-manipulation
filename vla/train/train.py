"""Behavior-cloning training loop."""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from vla.data.dataset import DemoDataset, collate
from vla.model.policy import VLAPolicy


def bc_loss(pred, target):
    pose = F.mse_loss(pred[:, :6], target[:, :6])
    grip = F.binary_cross_entropy_with_logits(pred[:, 6], target[:, 6])
    return pose + grip, pose, grip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = DemoDataset(cfg["data_dir"], cfg["image_size"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate)
    model = VLAPolicy(cfg["vision_model"], cfg["text_model"], cfg["hidden_dim"],
                      cfg["action_dim"], cfg["pretrained"], cfg["freeze_encoders"]).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])

    for epoch in range(cfg["epochs"]):
        model.train()
        for step, (imgs, instrs, acts) in enumerate(dl):
            pred = model(imgs.to(device), instrs)
            loss, pose, grip = bc_loss(pred, acts.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 4 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} "
                      f"(pose {pose.item():.4f}, grip {grip.item():.4f})")

    Path(cfg["checkpoint"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg}, cfg["checkpoint"])
    print(f"saved {cfg['checkpoint']}")


if __name__ == "__main__":
    main()
