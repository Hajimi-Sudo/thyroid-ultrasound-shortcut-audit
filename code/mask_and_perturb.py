"""
Image manipulation and perturbation pipeline for thyroid ultrasound shortcut analysis.
Implements:
1. Nodule ROI extraction with anatomical margin
2. Inpainting/occlusion of nodule using smooth parenchymal blending (avoiding sharp edge OOD artifacts)
3. Synthetic caliper injection for artifact sensitivity stress testing
4. Peripheral border/metadata stripping
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional, Union


def crop_nodule_roi(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],  # (xmin, ymin, xmax, ymax)
    margin_ratio: float = 0.10,
) -> Image.Image:
    """
    Extract nodule ROI with proportional margin to preserve boundary definition.
    """
    width, height = image.size
    xmin, ymin, xmax, ymax = bbox
    box_w = xmax - xmin
    box_h = ymax - ymin

    margin_x = int(box_w * margin_ratio)
    margin_y = int(box_h * margin_ratio)

    new_xmin = max(0, xmin - margin_x)
    new_ymin = max(0, ymin - margin_y)
    new_xmax = min(width, xmax + margin_x)
    new_ymax = min(height, ymax + margin_y)

    return image.crop((new_xmin, new_ymin, new_xmax, new_ymax))


def inpaint_nodule_background(
    image: Image.Image,
    mask: np.ndarray,  # 2D boolean or uint8 mask where 1/True indicates nodule
    blur_sigma: float = 12.0,
) -> Image.Image:
    """
    Inpaint/occlude nodule region using smoothed surrounding parenchymal texture.
    Avoids sharp step-function boundary artifacts that trigger false convolutional activations.
    """
    img_arr = np.array(image.convert("RGB"), dtype=np.float32)
    mask_binary = (mask > 0).astype(np.float32)

    # Dilate and smooth mask boundary for gradual alpha-blending
    smoothed_mask = gaussian_filter(mask_binary, sigma=3.0)
    smoothed_mask = np.clip(smoothed_mask, 0.0, 1.0)
    smoothed_mask_3d = np.repeat(smoothed_mask[:, :, np.newaxis], 3, axis=2)

    # Estimate background parenchymal fill by sampling outside the mask
    bg_pixels = img_arr[mask_binary == 0]
    if len(bg_pixels) > 0:
        bg_mean = np.mean(bg_pixels, axis=0)
        bg_std = np.std(bg_pixels, axis=0)
    else:
        bg_mean = np.array([128.0, 128.0, 128.0])
        bg_std = np.array([20.0, 20.0, 20.0])

    # Synthesize textured background matching ambient echogenicity and grain
    noise = np.random.normal(bg_mean, bg_std * 0.5, size=img_arr.shape)
    blurred_ambient = gaussian_filter(img_arr, sigma=(blur_sigma, blur_sigma, 0))
    synthetic_parenchyma = 0.6 * blurred_ambient + 0.4 * noise
    synthetic_parenchyma = np.clip(synthetic_parenchyma, 0, 255)

    # Smooth alpha blend
    inpainted = (1.0 - smoothed_mask_3d) * img_arr + smoothed_mask_3d * synthetic_parenchyma
    return Image.fromarray(np.uint8(np.clip(inpainted, 0, 255)))


def inject_synthetic_caliper(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    cross_size: int = 12,
    thickness: int = 1,
    color: Tuple[int, int, int] = (255, 255, 0),  # Yellow or white caliper cross
) -> Image.Image:
    """
    Inject realistic sonographer measurement crosshairs (+) at nodule extremities.
    Used for adversarial stress testing on benign nodules to quantify caliper shortcut bias.
    """
    img_copy = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copy)
    xmin, ymin, xmax, ymax = bbox

    # Points along longitudinal and transverse axes
    caliper_points = [
        ((xmin + xmax) // 2, ymin),  # Top
        ((xmin + xmax) // 2, ymax),  # Bottom
        (xmin, (ymin + ymax) // 2),  # Left
        (xmax, (ymin + ymax) // 2),  # Right
    ]

    for cx, cy in caliper_points:
        # Draw '+' crosshair
        draw.line([(cx - cross_size, cy), (cx + cross_size, cy)], fill=color, width=thickness)
        draw.line([(cx, cy - cross_size), (cx, cy + cross_size)], fill=color, width=thickness)

    # Optional dashed measurement line between calipers
    # (can be toggled for full measurement simulation)
    return img_copy


def strip_device_borders(
    image: Image.Image,
    crop_margins: Tuple[float, float, float, float] = (0.05, 0.08, 0.05, 0.05),
) -> Image.Image:
    """
    Remove top/bottom/left/right peripheral margins that typically contain
    vendor logos, patient name tags, depth scales, or mechanical frame borders.
    Margins: (left_pct, top_pct, right_pct, bottom_pct)
    """
    w, h = image.size
    l_pct, t_pct, r_pct, b_pct = crop_margins
    left = int(w * l_pct)
    top = int(h * t_pct)
    right = int(w * (1.0 - r_pct))
    bottom = int(h * (1.0 - b_pct))
    return image.crop((left, top, right, bottom))


def generate_caliper_mask(
    image_size: Tuple[int, int],
    bbox: Tuple[int, int, int, int],
    cross_size: int = 12,
    thickness: int = 1,
    dilation_pixels: int = 3,
) -> np.ndarray:
    """
    Generate a 2D binary mask M_caliper of shape (height, width) covering the
    synthetic caliper crosshairs, with morphological dilation.

    Formal definition matching Methods Section 2.5:
    M_caliper in {0, 1}^(H x W), where crosshairs are placed at nodule extremities
    and dilated by dilation_pixels (default 3) to capture adjacent gradient activations.
    """
    from scipy.ndimage import binary_dilation
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    xmin, ymin, xmax, ymax = bbox

    caliper_points = [
        ((xmin + xmax) // 2, ymin),
        ((xmin + xmax) // 2, ymax),
        (xmin, (ymin + ymax) // 2),
        (xmax, (ymin + ymax) // 2),
    ]

    for cx, cy in caliper_points:
        draw.line([(cx - cross_size, cy), (cx + cross_size, cy)], fill=255, width=thickness)
        draw.line([(cx, cy - cross_size), (cx, cy + cross_size)], fill=255, width=thickness)

    mask_arr = np.array(mask, dtype=np.uint8) > 0

    if dilation_pixels > 0:
        structure = np.ones((2 * dilation_pixels + 1, 2 * dilation_pixels + 1), dtype=bool)
        mask_arr = binary_dilation(mask_arr, structure=structure)

    return mask_arr.astype(np.uint8)
