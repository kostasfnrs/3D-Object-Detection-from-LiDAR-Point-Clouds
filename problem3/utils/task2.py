import numpy as np
from utils.task1 import label2corners

def get_residuals(centroids, xyz):
    return centroids - xyz

def is_inside(centroid, box_size, normalized_box_directions, xyz, inside_extra_filtering_indices):

    # get valid points for dot product
    filterd_xyz = xyz[inside_extra_filtering_indices]

    # get the projections of the residuals to the normalized box directions
    residuals = get_residuals(centroid, filterd_xyz)
    projections = residuals @ normalized_box_directions

    # get sizes absolut half and put them in the centroid dimension
    box_size = box_size * 0.5

    # positive projections
    projections = np.absolute(projections, dtype=np.float32)

    is_inside_indices = np.all(projections <= box_size, axis=1)
    first_selected_indices = np.where(inside_extra_filtering_indices)[0]
    correct_selected_indices = first_selected_indices[is_inside_indices]

    return correct_selected_indices

def get_box_sizes(box):
    # the ordering of the points in task 1 mastters for this calculation
    direction_x = box[:, 0, :] - box[:, 1, :]
    direction_y = box[:, 4, :] - box[:, 0, :]
    direction_z = box[:, 3, :] - box[:, 0, :]
    size_x = np.linalg.norm(direction_x, axis=1, keepdims=True)
    size_y = np.linalg.norm(direction_y, axis=1, keepdims=True)
    size_z = np.linalg.norm(direction_z, axis=1, keepdims=True)
    size_xyz = np.vstack([size_x.T, size_y.T, size_z.T]).T
    normalized_x = direction_x / size_x
    normalized_y = direction_y / size_y
    normalized_z = direction_z / size_z

    return (size_xyz, np.stack([normalized_x, normalized_y, normalized_z], axis=2))


def random_sampler_indices(max_points, indices_array):
    K = len(indices_array)  # Use the length of the indices array

    if K < max_points:
        # Randomly duplicate indices from the provided array
        random_indices = np.random.choice(indices_array, size=max_points - K, replace=True)
        final_indices = np.concatenate((indices_array, random_indices))
    elif K > max_points:
        # Randomly sample indices from the provided array
        final_indices = np.random.choice(indices_array, size=max_points, replace=False)
    else:
        final_indices = indices_array
        
    return final_indices

def random_sampler(max_points, xyz, feat):
    K = xyz.shape[0]

    if K < max_points:
        # Randomly duplicate points and concatenate in one step
        indices = np.random.randint(0, K, size=max_points - K)
        xyz = np.concatenate((xyz, xyz[indices]), axis=0)
        feat = np.concatenate((feat, feat[indices]), axis=0)
    elif K > max_points:
        # Randomly sample points
        indices = np.random.choice(K, size=max_points, replace=False)
        xyz = xyz[indices]
        feat = feat[indices]
        
    return xyz, feat

def inside_points(xyz, boxes, index="all"):
    min_x = np.min(boxes[:,:,0]) if index=="all" else np.min(boxes[index,:,0])
    min_y = np.min(boxes[:,:,1]) if index=="all" else np.min(boxes[index,:,1])
    min_z = np.min(boxes[:,:,2]) if index=="all" else np.min(boxes[index,:,2])
    max_x = np.max(boxes[:,:,0]) if index=="all" else np.max(boxes[index,:,0])
    max_y = np.max(boxes[:,:,1]) if index=="all" else np.max(boxes[index,:,1])
    max_z = np.max(boxes[:,:,2]) if index=="all" else np.max(boxes[index,:,2])
    inside_x = (xyz[:, 0] >= min_x) & (xyz[:, 0] <= max_x)
    inside_y = (xyz[:, 1] >= min_y) & (xyz[:, 1] <= max_y)
    inside_z = (xyz[:, 2] >= min_z) & (xyz[:, 2] <= max_z)
    return inside_x & inside_y & inside_z


def roi_pool(pred, xyz, feat, config):
    '''
    Task 2
    a. Enlarge predicted 3D bounding boxes by delta=1.0 meters in all directions.
       As our inputs consist of coarse detection results from the stage-1 network,
       the second stage will benefit from the knowledge of surrounding points to
       better refine the initial prediction.
    b. Form ROI's by finding all points and their corresponding features that lie 
       in each enlarged bounding box. Each ROI should contain exactly 512 points.
       If there are more points within a bounding box, randomly sample until 512.
       If there are less points within a bounding box, randomly repeat points until
       512. If there are no points within a bounding box, the box should be discarded.
    input
        pred (N,7) bounding box labels
        xyz (N,3) point cloud
        feat (N,C) features
        config (dict) data config
    output
        valid_pred (K',7)
        pooled_xyz (K',M,3)
        pooled_feat (K',M,C)
            with K' indicating the number of valid bounding boxes that contain at least
            one point
    useful config hyperparameters
        config['delta'] extend the bounding box by delta on all sides (in meters)
        config['max_points'] number of points in the final sampled ROI
    '''

    # Enlarge the 3d bounted boxes
    boxes = label2corners(pred + np.array([0,0,0,2*config['delta'],2*config['delta'],2*config['delta'],0]))

    # centroinds
    centroids = np.mean(boxes, axis=1)

    # filter points that are ousite of the max and min bounded boxes in general
    inside_all = inside_points(xyz, boxes, index="all")

    filtered_points = xyz[inside_all]
    filtered_feat = feat[inside_all]
    xyz = filtered_points
    feat = filtered_feat

    valid_pred = []
    pooled_xyz = []
    pooled_feat = []
    is_inside_indices_list = []

    box_sizes, normalized_box_directions = get_box_sizes(boxes)

    for i in range(len(boxes)):
        # extra filtering
        inside_extra_filtering_indices = inside_points(xyz, boxes, index=i)
        is_inside_indices = is_inside(centroids[i], box_sizes[i], normalized_box_directions[i], xyz, inside_extra_filtering_indices)
        if len(is_inside_indices) != 0:
            is_inside_indices_list.append(random_sampler_indices(config['max_points'], is_inside_indices))
            valid_pred.append(pred[i])

    
    is_inside_indices_list = np.array(is_inside_indices_list).squeeze()
    pooled_xyz = xyz[is_inside_indices_list]
    pooled_feat = feat[is_inside_indices_list]
    
    return np.array(valid_pred), np.array(pooled_xyz), np.array(pooled_feat)
