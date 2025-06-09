from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt


def initialize_sam_model(device, sam_model_type, sam_checkpoint):
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor_sam = SamPredictor(sam)
    return predictor_sam


def mask2box(mask: torch.Tensor):
    row = torch.nonzero(mask.sum(axis=0))[:, 0]
    if len(row) == 0:
        return None
    x1 = row.min().item()
    x2 = row.max().item()
    col = np.nonzero(mask.sum(axis=1))[:, 0]
    y1 = col.min().item()
    y2 = col.max().item()
    return x1, y1, x2 + 1, y2 + 1


def mask2box_multi_level(mask: torch.Tensor, level, expansion_ratio):
    x1, y1, x2, y2 = mask2box(mask)
    if level == 0:
        return x1, y1, x2, y2
    shape = mask.shape
    x_exp = int(abs(x2 - x1) * expansion_ratio) * level
    y_exp = int(abs(y2 - y1) * expansion_ratio) * level
    return max(0, x1 - x_exp), max(0, y1 - y_exp), min(shape[1], x2 + x_exp), min(shape[0], y2 + y_exp)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def run_sam(image, num_random_rounds, num_selected_points, point_coords, predictor_sam):
    best_score = 0
    best_mask = np.zeros_like(image, dtype=bool)

    point_coords_new = np.zeros_like(point_coords)
    point_coords_new[:, 0] = point_coords[:, 0]
    point_coords_new[:, 1] = point_coords[:, 1]

    # Get only a random subsample of them for num_random_rounds times and choose the mask with highest confidence score
    for i in range(num_random_rounds):
        np.random.shuffle(point_coords_new)
        masks, scores, logits = predictor_sam.predict(
            point_coords=point_coords_new[:num_selected_points],
            point_labels=np.ones(point_coords_new[:num_selected_points].shape[0]),
            multimask_output=False,
        )

        # # Visualize segment result of SAM
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # show_mask(masks, plt.gca())
        # show_points(point_coords_new[:num_selected_points], np.ones(point_coords_new[:num_selected_points].shape[0]), plt.gca())
        # plt.show()

        if scores[0] > best_score:
            best_score = scores[0]
            best_mask = masks[0]

    return best_mask
