import clip
import numpy as np
import imageio.v2 as imageio
import torch
from tqdm import tqdm
import os
from ovsam3d.data.load import Camera, SuperPoints, Images, PointCloud, get_number_of_images
from ovsam3d.extractor_features.utils import initialize_sam_model, mask2box_multi_level, run_sam


class PointProjector:
    def __init__(self,
                 images_path,
                 camera: Camera,
                 point_cloud: PointCloud,
                 vis_threshold,
                 indices) -> object:
        self.images_path = images_path
        self.vis_threshold = vis_threshold
        self.indices = indices
        self.camera = camera
        self.point_cloud = point_cloud
        print(f"[INFO] Projecting pointcloud to per view...")
        self.visible_points_view, self.projected_points = self.get_visible_points_view()
        print(f"[INFO] Complete project!")

    def get_visible_points_view(self):
        # Initialization
        vis_threshold = self.vis_threshold
        indices = self.indices
        depth_scale = self.camera.depth_scale
        poses = self.camera.load_poses(indices)
        X = self.point_cloud.get_homogeneous_coordinates()
        n_points = self.point_cloud.num_points
        depths_path = self.camera.depths_path
        # Get the resolution of depth images and color images, maybe they are not equal
        depth_resolution = imageio.imread(os.path.join(depths_path, '0.png')).shape
        color_resolution = imageio.imread(os.path.join(self.images_path, '0.jpg')).shape[:2]
        height = depth_resolution[0]
        width = depth_resolution[1]
        depth_intrinsic = self.camera.get_adapted_intrinsic(depth_resolution)
        color_intrinsic = self.camera.get_adapted_intrinsic(color_resolution)

        depth_projected_points = np.zeros((len(indices), n_points, 2), dtype=int)
        color_projected_points = np.zeros((len(indices), n_points, 2), dtype=int)
        visible_points_view = np.zeros((len(indices), n_points), dtype=bool)

        for i, idx in tqdm(enumerate(indices)):
            # Get the coordinates of the projected points in the i-th view (i.e. the view with index idx)
            depth_projected_points_not_norm = (depth_intrinsic @ poses[i] @ X.T).T
            color_projected_points_not_norm = (color_intrinsic @ poses[i] @ X.T).T
            # Get the mask of the points which have a non-null third coordinate to avoid division by zero
            depth_mask = (depth_projected_points_not_norm[:, 2] != 0)
            color_mask = (color_projected_points_not_norm[:, 2] != 0)
            # Get non homogeneous coordinates of valid points (2D in the image)
            depth_projected_points[i][depth_mask] = np.column_stack(
                [[depth_projected_points_not_norm[:, 0][depth_mask] / depth_projected_points_not_norm[:, 2][depth_mask],
                  depth_projected_points_not_norm[:, 1][depth_mask] / depth_projected_points_not_norm[:, 2][depth_mask]]]).T
            color_projected_points[i][color_mask] = np.column_stack(
                [[color_projected_points_not_norm[:, 0][color_mask] / color_projected_points_not_norm[:, 2][color_mask],
                  color_projected_points_not_norm[:, 1][color_mask] / color_projected_points_not_norm[:, 2][color_mask]]]).T
            # Load the depth from the sensor
            depth_path = os.path.join(depths_path, str(idx) + '.png')
            sensor_depth = imageio.imread(depth_path) / depth_scale
            inside_mask = (depth_projected_points[i, :, 0] >= 0) * (depth_projected_points[i, :, 1] >= 0) \
                          * (depth_projected_points[i, :, 0] < width) * (depth_projected_points[i, :, 1] < height) \
                          * (color_projected_points[i, :, 0] < color_resolution[1]) * (color_projected_points[i, :, 1] < color_resolution[0])
            pi = depth_projected_points[i].T
            # Depth of the points of the pointcloud, projected in the i-th view, computed using the projection matrices
            point_depth = depth_projected_points_not_norm[:, 2]
            # Compute the visibility mask, true for all the points which are visible from the i-th view
            visibility_mask = (np.abs(sensor_depth[pi[1][inside_mask], pi[0][inside_mask]]
                                      - point_depth[inside_mask]) <= vis_threshold).astype(bool)
            inside_mask[inside_mask == True] = visibility_mask
            visible_points_view[i] = inside_mask
        return visible_points_view, color_projected_points


class FeaturesExtractor:
    def __init__(self,
                 camera,
                 clip_model,
                 images,
                 masks,
                 pointcloud,
                 sam_model_type,
                 sam_checkpoint,
                 vis_threshold,
                 device):
        self.camera = camera
        self.images = images
        self.device = device
        self.point_projector = PointProjector(camera, pointcloud, masks, vis_threshold, images.indices)
        self.predictor_sam = initialize_sam_model(device, sam_model_type, sam_checkpoint)
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device)

    def extract_features(self, topk, multi_level_expansion_ratio, num_levels, num_random_rounds, num_selected_points,
                         save_crops, out_folder, optimize_gpu_usage=False):
        if (save_crops):
            out_folder = os.path.join(out_folder, "crops")
            os.makedirs(out_folder, exist_ok=True)

        topk_indices_per_mask = self.point_projector.get_top_k_indices_per_mask(topk)

        num_masks = self.point_projector.masks.num_masks
        mask_clip = np.zeros((num_masks, 768))  # initialize mask clip

        np_images = self.images.get_as_np_list()
        for mask in tqdm(range(num_masks)):  # for each mask
            images_crops = []
            if (optimize_gpu_usage):
                self.clip_model.to(torch.device('cpu'))
                self.predictor_sam.model.cuda()
            for view_count, view in enumerate(topk_indices_per_mask[mask]):  # for each view
                if (optimize_gpu_usage):
                    torch.cuda.empty_cache()

                # Get original mask points coordinates in 2d images
                point_coords = np.transpose(
                    np.where(self.point_projector.visible_points_in_view_in_mask[view][mask] == True))
                if (point_coords.shape[0] > 0):
                    self.predictor_sam.set_image(np_images[view])

                    # SAM
                    best_mask = run_sam(image_size=np_images[view],
                                        num_random_rounds=num_random_rounds,
                                        num_selected_points=num_selected_points,
                                        point_coords=point_coords,
                                        predictor_sam=self.predictor_sam, )

                    # MULTI LEVEL CROPS
                    for level in range(num_levels):
                        # get the bbox and corresponding crops
                        x1, y1, x2, y2 = mask2box_multi_level(torch.from_numpy(best_mask), level,
                                                              multi_level_expansion_ratio)
                        cropped_img = self.images.images[view].crop((x1, y1, x2, y2))

                        if (save_crops):
                            cropped_img.save(os.path.join(out_folder, f"crop{mask}_{view}_{level}.png"))

                        # Compute the CLIP feature using the standard clip model
                        cropped_img_processed = self.clip_preprocess(cropped_img)
                        images_crops.append(cropped_img_processed)

            if (optimize_gpu_usage):
                self.predictor_sam.model.cpu()
                self.clip_model.to(torch.device('cuda'))
            if (len(images_crops) > 0):
                image_input = torch.tensor(np.stack(images_crops))
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input.to(self.device)).float()
                    image_features /= image_features.norm(dim=-1, keepdim=True)  # normalize

                mask_clip[mask] = image_features.mean(axis=0).cpu().numpy()

        return mask_clip
