import hydra
from omegaconf import DictConfig
import numpy as np
from ovsam3d.data.load import Camera, SuperPoints, Images, PointCloud, get_number_of_images
from ovsam3d.tools.utils import back_project, matmul_accelerate
from ovsam3d.extractor_features.features_extractor import PointProjector
from ovsam3d.extractor_features.utils import initialize_sam_model, run_sam, mask2box_multi_level
import torch
import clip
import os
from glob import glob
from tqdm import tqdm


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
    ctx.external.sam_checkpoint = os.path.abspath(ctx.external.sam_checkpoint)

    output_mask = ctx.output.output_mask
    output_feature = ctx.output.output_feature

    if not os.path.exists(output_mask):
        os.makedirs(output_mask)
    if not os.path.exists(output_feature):
        os.makedirs(output_feature)

    # Initialize SAM and CLIP
    predictor_sam = initialize_sam_model(device, ctx.external.sam_model_type, ctx.external.sam_checkpoint)
    clip_model, clip_preprocess = clip.load(ctx.external.clip_model, device)

    # Load the path list of ScanNet200 scenes
    scans = sorted(os.listdir(ctx.data.scans_path))
    # slice = int(len(scans) / 2)  # You can set “slice” for extracting synchronously to save time
    for scan in scans:
        print(f"[INFO] Processing scene id: {scan}")
        path = os.path.join(ctx.data.scans_path, scan)
        poses_path = os.path.join(path, ctx.data.camera.poses_path)
        point_cloud_path = glob(os.path.join(path, '*vh_clean_2.ply'))[0]
        intrinsic_path = os.path.join(path, ctx.data.camera.intrinsic_path)
        images_path = os.path.join(path, ctx.data.images.images_path)
        depths_path = os.path.join(path, ctx.data.depths.depths_path)
        super_points_path = glob(os.path.join(path, '*.0.010000.segs.json'))[0]

        # Load the camera configurations
        camera = Camera(intrinsic_path=intrinsic_path,
                        intrinsic_resolution=ctx.data.camera.intrinsic_resolution,
                        poses_path=poses_path,
                        depths_path=depths_path,
                        extension_depth=ctx.data.depths.depths_ext,
                        depth_scale=ctx.data.depths.depth_scale)

        # Load the superpoints
        super_points = SuperPoints(super_points_path, top_num=ctx.data.superpoints.top_num)

        # Load the images
        indices = np.arange(0, get_number_of_images(poses_path), step=ctx.hyperparameters.frequency)
        images = Images(images_path=images_path,
                        extension=ctx.data.images.images_ext,
                        indices=indices)
        np_images = images.get_as_np_list()

        # Load the pointcloud
        pointcloud = PointCloud(point_cloud_path)

        # Initialize the PointProjector
        point_projector = PointProjector(images_path=images_path,
                                         camera=camera,
                                         point_cloud=pointcloud,
                                         vis_threshold=ctx.hyperparameters.vis_threshold,
                                         indices=images.indices)

        # Select the views with the most initial 3D prompts projection points
        visible_super_points = matmul_accelerate(point_projector.visible_points_view,
                                                 super_points.super_points_masks,
                                                 device)
        topk_indices_super_points = np.argsort(-visible_super_points[:, super_points.top_index], axis=0)[
                                    :ctx.hyperparameters.top_k, :].T

        # Initialize overlapping scores table and mask features, which is also the saved result of the first step
        overlapping_score_table = np.zeros((super_points.super_points_masks.shape[1], super_points.top_index.shape[0]), 
                                            dtype=np.float16)
        mask_feature = np.zeros((len(super_points.top_index), 768))

        # Generate coarse masks
        print(f"[INFO] Calculating overlapping scores table and mask features...")
        for id_count, super_points_id in tqdm(enumerate(super_points.top_index), total=len(super_points.top_index)):
            images_crops = []
            for view in topk_indices_super_points[id_count]:
                # Get visible 3D prompts' coordinates in 2d images
                visible_index_per_view = super_points.super_points_masks[:, super_points_id] * \
                                         point_projector.visible_points_view[view]
                point_coords = point_projector.projected_points[view, visible_index_per_view, :]
                
                if (point_coords.shape[0] > 0):
                    # Get the best segmentation masks with SAM via initial 3D prompts
                    predictor_sam.set_image(np_images[view])
                    best_mask = run_sam(image=np_images[view],
                                        num_random_rounds=ctx.hyperparameters.num_random_rounds,
                                        num_selected_points=ctx.hyperparameters.num_selected_points,
                                        point_coords=point_coords,
                                        predictor_sam=predictor_sam, )

                    # Back project from posed images to 3D mesh
                    back_mask3d = back_project(best_mask, point_projector.projected_points[view][
                                                          point_projector.visible_points_view[view], :])

                    # Calculate the overlapping scores between all superpoints and initial 3D prompts
                    non_zero_super_points = np.nonzero(visible_super_points[view])[0]
                    visible_index = super_points.super_points_masks[point_projector.visible_points_view[view], :]
                    covered_super_points = matmul_accelerate(np.expand_dims(back_mask3d, axis=0),
                                                             visible_index[:, non_zero_super_points],
                                                             device)
                    repetition_super_points = covered_super_points / visible_super_points[view, non_zero_super_points]

                    # Merge superpoints with high overlapping scores
                    similar_super_points = non_zero_super_points[repetition_super_points[0] > ctx.hyperparameters.overlapping_ratio]
                    overlapping_score_table[similar_super_points, id_count] += 1

                    # FIXME: Crop sub-images from views referring to openmask3d
                    for level in range(ctx.hyperparameters.num_of_levels):
                        x1, y1, x2, y2 = mask2box_multi_level(torch.from_numpy(best_mask), level,
                                                              ctx.hyperparameters.multi_level_expansion_ratio)
                        cropped_img = images.images[view].crop((x1, y1, x2, y2))
                        cropped_img_processed = clip_preprocess(cropped_img)
                        images_crops.append(cropped_img_processed)

            # Get CLIP features from cropped images
            if (len(images_crops) > 0):
                image_input = torch.tensor(np.stack(images_crops))
                with torch.no_grad():
                    image_features = clip_model.encode_image(image_input.to(device)).float()
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                mask_feature[id_count] = image_features.mean(axis=0).cpu().numpy()

        if not os.path.exists(os.path.join(output_feature, scan)):
            os.makedirs(os.path.join(output_feature, scan))
        if not os.path.exists(os.path.join(output_mask, scan)):
            os.makedirs(os.path.join(output_mask, scan))
        np.save(os.path.join(output_feature, scan, 'all_features.npy'), mask_feature)
        torch.save(overlapping_score_table.astype(np.float16), os.path.join(output_mask, scan, 'score_super_points.pt'))
        print(f"[INFO] Generate coarse masks and save!")


if __name__ == "__main__":
    main()
