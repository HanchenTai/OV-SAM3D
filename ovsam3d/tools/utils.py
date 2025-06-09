import numpy as np
import torch


def back_project(mask_2d, projected_points):
    """
    Get the 3D back-projection result of the 2D mask projected into 3D space.
    """
    mask_3d = np.zeros(projected_points.shape[0], dtype=bool)
    projected_points[:, [0, 1]] = projected_points[:, [1, 0]]
    coords = np.transpose(np.where(mask_2d))
    array_tuples = {tuple(row) for row in coords}
    point_indices = [i for i, row in enumerate(projected_points) if tuple(row) in array_tuples]
    mask_3d[point_indices] = True
    return mask_3d


def matmul_accelerate(mat1, mat2, device):
    """
    Transfer the matrix data to be multiplied from the CPU to the GPU to accelerate computation.
    """
    mat1 = torch.from_numpy(mat1.astype(np.float16)).to(device)
    mat2 = torch.from_numpy(mat2.astype(np.float16)).to(device)
    result = mat1 @ mat2
    result = result.cpu().numpy()
    return result
