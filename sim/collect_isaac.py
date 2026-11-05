"""Isaac Sim demonstration collection (stub).

Run inside Isaac Sim's Python (`./python.sh sim/collect_isaac.py`). Scripts a
tabletop scene, runs a scripted expert, and writes .npz episodes consumed by
vla.data.DemoDataset.
"""
import argparse
from pathlib import Path

import numpy as np


def scripted_episode(instruction, steps=50, size=224):
    """Placeholder expert; replace with Isaac Sim camera render + IK controller."""
    images = np.zeros((steps, size, size, 3), dtype=np.uint8)
    actions = np.zeros((steps, 7), dtype=np.float32)
    actions[:, 2] = np.linspace(-0.5, 0.5, steps)  # descend then lift
    actions[steps // 2:, 6] = 1.0                  # close gripper halfway
    return images, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demos")
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from isaacsim import SimulationApp  # noqa: F401
        print("Isaac Sim detected; wire scene setup here.")
    except ImportError:
        print("Isaac Sim not found; writing placeholder episodes.")
    for i in range(args.episodes):
        instr = "pick up the red block"
        imgs, acts = scripted_episode(instr)
        np.savez_compressed(out / f"ep_{i:04d}.npz", images=imgs, actions=acts, instruction=instr)
    print(f"wrote {args.episodes} episodes to {out}")


if __name__ == "__main__":
    main()
