"""
Visualization utilities for key-slices, ROIs, and attention maps.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.io import load_nifti, crop_roi


def visualize_key_slice(volume, key_slice, axis=0, title="", save_path=None):
    """Visualize a key slice from 3D volume."""
    if key_slice is None or key_slice < 0:
        return

    if axis == 0:
        slice_2d = volume[key_slice, :, :]
    elif axis == 1:
        slice_2d = volume[:, key_slice, :]
    else:
        slice_2d = volume[:, :, key_slice]

    plt.figure(figsize=(6, 6))
    plt.imshow(slice_2d, cmap='gray')
    plt.title(title)
    plt.colorbar()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_bbox(volume, bbox, title="", save_path=None):
    """Visualize volume with bounding box."""
    if bbox is None:
        return

    d_min, h_min, w_min, d_max, h_max, w_max = bbox
    center_d = (d_min + d_max) // 2

    plt.figure(figsize=(12, 4))

    # Axial
    plt.subplot(1, 3, 1)
    plt.imshow(volume[center_d, :, :], cmap='gray')
    rect = plt.Rectangle((w_min, h_min), w_max - w_min, h_max - h_min,
                       fill=False, color='red', linewidth=2)
    plt.gca().add_patch(rect)
    plt.title("%s - Axial" % title)

    # Coronal
    plt.subplot(1, 3, 2)
    plt.imshow(volume[:, (h_min + h_max) // 2, :], cmap='gray')
    rect = plt.Rectangle((w_min, d_min), w_max - w_min, d_max - d_min,
                       fill=False, color='red', linewidth=2)
    plt.gca().add_patch(rect)
    plt.title("%s - Coronal" % title)

    # Sagittal
    plt.subplot(1, 3, 3)
    plt.imshow(volume[:, :, (w_min + w_max) // 2], cmap='gray')
    rect = plt.Rectangle((h_min, d_min), h_max - h_min, d_max - d_min,
                       fill=False, color='red', linewidth=2)
    plt.gca().add_patch(rect)
    plt.title("%s - Sagittal" % title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_attention_map(attention, title="", save_path=None):
    """Visualize attention map."""
    plt.figure(figsize=(6, 6))
    plt.imshow(attention, cmap='viridis')
    plt.colorbar()
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_prediction_grid(images, predictions, exam_ids, save_path=None):
    """Visualize multiple predictions in a grid."""
    B = len(exam_ids)
    cols = min(4, B)
    rows = (B + cols - 1) // cols

    plt.figure(figsize=(4 * cols, 4 * rows))

    for i, (img, pred) in enumerate(zip(images, predictions)):
        plt.subplot(rows, cols, i + 1)
        slice_idx = img.shape[0] // 2
        plt.imshow(img[slice_idx], cmap='gray')

        # Add predictions as text
        pred_str = ", ".join(["%s: %.2f" % (d, p) for d, p in zip(
            ["SST", "IST", "SSC", "LHBT", "IGHL", "RIPI", "GHOA"], pred)])
        plt.title("%s\n%s" % (exam_ids[i], pred_str[:50]), fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def save_volume_slices(volume, slice_indices, output_dir, exam_id, prefix=""):
    """Save multiple slices of a 3D volume."""
    os.makedirs(output_dir, exist_ok=True)

    for i, idx in enumerate(slice_indices):
        if idx >= 0 and idx < volume.shape[0]:
            slice_2d = volume[idx, :, :]
            save_path = os.path.join(output_dir, "%s_%s_slice_%d.png" % (exam_id, prefix, i))
            plt.figure(figsize=(6, 6))
            plt.imshow(slice_2d, cmap='gray')
            plt.title("%s slice %d" % (prefix, idx))
            plt.savefig(save_path)
            plt.close()