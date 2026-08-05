#!/usr/bin/env python3
"""Add a six-camera cubemap rig to a semantic-mapping Gazebo world.

The source world is never modified. The generated SDF is a temporary copy
used by the DA360 simulation launcher.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path


FACES = ("front", "right", "back", "left", "top", "bottom")

# Gazebo's camera looks along its local +X axis. The mobile-base world uses
# +X forward, so these poses put six square cameras at one optical center.
FACE_POSES = {
    "front": "0.04 0 0 0 0 0",
    "right": "0.04 0 0 0 0 -1.57079632679",
    "back": "0.04 0 0 0 0 3.14159265359",
    "left": "0.04 0 0 0 0 1.57079632679",
    "top": "0.04 0 0 0 1.57079632679 0",
    "bottom": "0.04 0 0 0 -1.57079632679 0",
}


def add_camera(link: ET.Element, face: str, face_size: int, update_rate: float) -> None:
    name = f"da360_sim_{face}_camera"
    for sensor in list(link.findall("sensor")):
        if sensor.get("name") == name:
            link.remove(sensor)

    sensor = ET.SubElement(link, "sensor", {"name": name, "type": "camera"})
    ET.SubElement(sensor, "pose").text = FACE_POSES[face]
    ET.SubElement(sensor, "topic").text = f"/da360_sim/cubemap/{face}"
    ET.SubElement(sensor, "always_on").text = "true"
    ET.SubElement(sensor, "visualize").text = "false"
    ET.SubElement(sensor, "update_rate").text = f"{update_rate:.6f}"

    camera = ET.SubElement(sensor, "camera", {"name": f"da360_sim_{face}"})
    ET.SubElement(camera, "horizontal_fov").text = f"{math.pi / 2.0:.12f}"
    image = ET.SubElement(camera, "image")
    ET.SubElement(image, "width").text = str(face_size)
    ET.SubElement(image, "height").text = str(face_size)
    ET.SubElement(image, "format").text = "R8G8B8"
    clip = ET.SubElement(camera, "clip")
    ET.SubElement(clip, "near").text = "0.05"
    ET.SubElement(clip, "far").text = "20.0"


def generate(source: Path, target: Path, face_size: int, update_rate: float) -> None:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.parse(source, parser=parser).getroot()
    model = root.find(".//model[@name='mobile_base']")
    if model is None:
        raise RuntimeError(f"mobile_base model not found in {source}")

    link = model.find("link[@name='gimbal_pitch_link']")
    if link is None:
        link = model.find("link[@name='base_link']")
    if link is None:
        raise RuntimeError("camera mount link not found; expected gimbal_pitch_link or base_link")

    for face in FACES:
        add_camera(link, face, face_size, update_rate)

    target.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-world", required=True, type=Path)
    parser.add_argument("--output-world", required=True, type=Path)
    parser.add_argument("--face-size", type=int, default=360)
    parser.add_argument("--update-rate", type=float, default=5.0)
    args = parser.parse_args()

    if args.face_size < 32:
        parser.error("--face-size must be at least 32")
    if args.update_rate <= 0.0:
        parser.error("--update-rate must be positive")
    if not args.input_world.is_file():
        parser.error(f"world does not exist: {args.input_world}")

    generate(args.input_world, args.output_world, args.face_size, args.update_rate)
    print(f"generated {args.output_world} with faces: {', '.join(FACES)}")


if __name__ == "__main__":
    main()
