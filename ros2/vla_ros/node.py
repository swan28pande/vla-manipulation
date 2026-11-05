"""ROS2 inference node: camera + instruction -> end-effector action."""
import argparse

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from vla.model.policy import VLAPolicy


class VLANode(Node):
    def __init__(self, checkpoint):
        super().__init__("vla_policy")
        ckpt = torch.load(checkpoint, map_location="cpu")
        cfg = ckpt["config"]
        self.size = cfg["image_size"]
        self.model = VLAPolicy(cfg["vision_model"], cfg["text_model"], cfg["hidden_dim"],
                               cfg["action_dim"], pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.instruction = "pick up the red block"
        self.create_subscription(Image, "/camera/image_raw", self.on_image, 1)
        self.create_subscription(String, "/instruction", self.on_instruction, 1)
        self.twist_pub = self.create_publisher(Twist, "/ee_delta", 1)
        self.grip_pub = self.create_publisher(Float32, "/gripper", 1)

    def on_instruction(self, msg):
        self.instruction = msg.data
        self.get_logger().info(f"instruction: {msg.data}")

    def on_image(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)[..., :3]
        x = torch.from_numpy(img.copy()).permute(2, 0, 1).float()[None] / 255.0
        x = torch.nn.functional.interpolate(x, size=(self.size, self.size))
        with torch.no_grad():
            a = self.model(x, [self.instruction])[0]
        t = Twist()
        t.linear.x, t.linear.y, t.linear.z = map(float, a[:3])
        t.angular.x, t.angular.y, t.angular.z = map(float, a[3:6])
        self.twist_pub.publish(t)
        self.grip_pub.publish(Float32(data=float(torch.sigmoid(a[6]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/vla.pt")
    args, _ = ap.parse_known_args()
    rclpy.init()
    rclpy.spin(VLANode(args.checkpoint))


if __name__ == "__main__":
    main()
