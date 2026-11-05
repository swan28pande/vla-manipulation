# VLA Robot Manipulation Learner

A minimal vision-language-action (VLA) model that maps an RGB image and a natural-language instruction to a robot end-effector action (Δxyz, Δrpy, gripper) for tabletop manipulation. Trained with behavior cloning on demonstration data.

**Stack:** Python, PyTorch, Hugging Face (vision + text encoders), ROS2 (inference node), Isaac Sim (data collection / eval).

## Layout
```
vla/model/policy.py      # VLA policy: ViT image encoder + text encoder + fusion + action head
vla/data/dataset.py      # Demonstration dataset (episodes of image, instruction, action)
vla/train/train.py       # Imitation-learning training loop
sim/collect_isaac.py     # Isaac Sim demo collection stub
ros2/vla_ros/node.py     # ROS2 node: subscribes to camera + instruction, publishes EE pose deltas
configs/default.yaml     # Hyperparameters
```

## Quickstart
```bash
pip install -r requirements.txt
python -m vla.train.train --config configs/default.yaml   # trains on synthetic data if no dataset given
```

Demonstrations are expected as `.npz` episodes with `images (T,H,W,3)`, `actions (T,7)` and an `instruction` string. Without a dataset path, a synthetic dataset is used so the pipeline runs end to end.

## ROS2
```bash
python -m ros2.vla_ros.node --checkpoint checkpoints/vla.pt
```
Subscribes to `/camera/image_raw` and `/instruction`, publishes `geometry_msgs/Twist` on `/ee_delta` and `std_msgs/Float32` on `/gripper`.

## Status
Early skeleton. Next: pretrained VLM backbone fine-tuning, action chunking, Isaac Sim eval harness.
