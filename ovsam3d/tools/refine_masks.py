import hydra
from omegaconf import DictConfig
import numpy as np
from ovsam3d.data.load import Camera, SuperPoints, Images, PointCloud, get_number_of_images
import torch
import os
from glob import glob
from tqdm import tqdm
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


@hydra.main(config_path="../configs", config_name="ovsam3d_scannet200")
def main(ctx: DictConfig):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Set the path for loading data and saving results
    os.chdir(hydra.utils.get_original_cwd())
    ctx.data.scans_path = os.path.abspath(ctx.data.scans_path)
    ctx.data.label_path = os.path.abspath(ctx.data.label_path)
    ctx.output.output_mask = os.path.abspath(ctx.output.output_mask)
    ctx.output.output_feature = os.path.abspath(ctx.output.output_feature)

    output_mask = ctx.output.output_mask
    output_feature = ctx.output.output_feature
    if not os.path.exists(output_mask):
        os.makedirs(output_mask)
    if not os.path.exists(output_feature):
        os.makedirs(output_feature)

    # Refine coarse 3D masks with overlapping score table
    scans = sorted(os.listdir(ctx.data.scans_path))
    for scan in scans:
        path = os.path.join(ctx.data.scans_path, scan)
        super_points_path = glob(os.path.join(path, '*.0.010000.segs.json'))[0]

        # Load the superpoints
        super_points = SuperPoints(super_points_path, top_num=ctx.data.superpoints.top_num)

        # Load the overlapping score table and coarse mask features
        mask_super_points = torch.load(os.path.join(output_mask, scan, 'score_super_points.pt'))
        mask_clip = np.load(os.path.join(output_feature, scan, 'all_features.npy'))

        # Refine coarse 3D masks via overlapping scores
        # Judge the coarse 3D masks if have been merged
        combine_judge = np.ones(len(super_points.top_index), dtype=bool)
        # Mark which the coarse 3D masks merged with
        combine_index = np.eye(len(super_points.top_index))

        for i in range(len(super_points.top_index)):
            if not combine_judge[i]:
                continue
            # Calculate the overlap between coarse masks
            covered_index = np.where(combine_judge)[0]
            covered_score = mask_super_points[:, i] @ mask_super_points[:, combine_judge]
            num_sub_points = np.sum(np.count_nonzero(mask_super_points[:, i]))
            select_index = covered_index[covered_score >= max(1, num_sub_points/ctx.hyperparameters.temperature)]

            # Continuously update the coarse 3D mask and record the composition of the merged result
            if len(select_index) < 2:
                continue
            min_index = np.min(select_index)
            mask_super_points[:, min_index] = np.max(mask_super_points[:, select_index], axis=1).squeeze()
            combine_index[:, min_index] = np.max(combine_index[:, select_index], axis=1).squeeze()
            delete_index = select_index[select_index > min_index]
            combine_judge[delete_index] = False

        # Save refined masks with compositions
        combine_index = combine_index[:, combine_judge]
        mask_super_points = mask_super_points[:, combine_judge]
        label_counts = np.sum(mask_super_points, axis=1)
        np.save(os.path.join(output_feature, scan, 'combine_mark.npy'), combine_index)
        label_super_points = np.argmax(mask_super_points, axis=1)
        label_super_points[label_counts == 0] = -1
        modified_masks = np.zeros((super_points.super_points_masks.shape[0], mask_super_points.shape[1]), dtype=np.float16)
        for i in range(mask_super_points.shape[1]):
            modified_masks[:, i] = np.sum(super_points.super_points_masks[:, label_super_points == i], axis=1)
        torch.save(modified_masks.astype(np.float16), os.path.join(output_mask, scan, 'final_mask.pt'))
        print(f"[INFO] Mask with features for scene {scan} saved.")


if __name__ == "__main__":
    main()
