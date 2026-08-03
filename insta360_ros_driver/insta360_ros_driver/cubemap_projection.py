"""ERP-to-cubemap projection helpers used by the ROS cubemap node."""

import math

import cv2
import numpy as np


def build_face_map(face_size: int, face: str):
    """Build normalized ERP remap coordinates for one 90-degree cube face."""
    jj, ii = np.meshgrid(
        np.arange(face_size, dtype=np.float32),
        np.arange(face_size, dtype=np.float32),
    )
    a = 2.0 * (jj + 0.5) / face_size - 1.0
    b = 2.0 * (ii + 0.5) / face_size - 1.0

    if face == 'right':
        dx, dy, dz = np.ones_like(a), -b, -a
    elif face == 'left':
        dx, dy, dz = -np.ones_like(a), -b, a
    elif face == 'top':
        dx, dy, dz = a, np.ones_like(a), b
    elif face == 'bottom':
        dx, dy, dz = a, -np.ones_like(a), -b
    elif face == 'front':
        dx, dy, dz = a, -b, np.ones_like(a)
    elif face == 'back':
        dx, dy, dz = -a, -b, -np.ones_like(a)
    else:
        raise ValueError(f'unknown cubemap face: {face}')

    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx /= norm
    dy /= norm
    dz /= norm

    theta = np.arctan2(dz, dx)
    phi = np.arcsin(dy)
    map_x_norm = (theta + math.pi) / (2.0 * math.pi)
    map_y_norm = (math.pi / 2.0 - phi) / math.pi
    return map_x_norm.astype(np.float32), map_y_norm.astype(np.float32)


def remap_face(erp_image, map_x_norm, map_y_norm, interpolation, border_mode):
    """Project an ERP image into a cube face using normalized map coordinates."""
    height, width = erp_image.shape[:2]
    map_x = map_x_norm * (width - 1)
    map_y = map_y_norm * (height - 1)
    return cv2.remap(
        erp_image,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=border_mode,
    )
