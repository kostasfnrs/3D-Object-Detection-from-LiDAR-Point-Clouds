import numpy as np
from shapely.geometry import Polygon

def label2corners(label, target=False):
    '''
    Task 1
    input
        label (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
    output
        corners (N,8,3) corner coordinates in the rectified reference frame
    '''
    if target:
        # get the scenes that have no labels
        pattern = np.array([-1, -1, -1, -100, -100, -100, -1])
        zerp_target_points = np.all(np.equal(label, pattern), axis=1)

    r = label[:, 6:7]
    cos_alpha = np.cos(r)
    sin_alpha = np.sin(r)

    zeros = np.zeros_like(cos_alpha)
    ones = np.ones_like(cos_alpha)
    pitch_rotation_array_1 = np.concatenate([cos_alpha, zeros, sin_alpha], axis=1)
    pitch_rotation_array_2 = np.concatenate([zeros, ones, zeros], axis=1)
    pitch_rotation_array_3 = np.concatenate([-sin_alpha,     zeros,     cos_alpha], axis=1)
    pitch_rotation_matrix = np.stack([pitch_rotation_array_1, pitch_rotation_array_2, pitch_rotation_array_3], axis=1)

    # xyz -> lhw
    h = np.expand_dims(label[:, 3], axis=-1)
    w = np.expand_dims(label[:, 4], axis=-1)
    l = np.expand_dims(label[:, 5], axis=-1)


    point_1 = label[:, 0:3] + (pitch_rotation_matrix @ np.stack([+ l/2, + zeros, - w/2], axis=1)).squeeze(-1)
    point_2 = label[:, 0:3] + (pitch_rotation_matrix @ np.stack([- l/2, + zeros, - w/2], axis=1)).squeeze(-1)
    point_3 = label[:, 0:3] + (pitch_rotation_matrix @ np.stack([- l/2, + zeros, + w/2], axis=1)).squeeze(-1)
    point_4 = label[:, 0:3] + (pitch_rotation_matrix @ np.stack([+ l/2, + zeros, + w/2], axis=1)).squeeze(-1)
    point_5 = label[:, 0:3] + (pitch_rotation_matrix @ np.stack([+ l/2, - h, - w/2], axis=1)).squeeze(-1)
    point_6 = label[:, 0:3] + (pitch_rotation_matrix @ np.stack([- l/2, - h, - w/2], axis=1)).squeeze(-1)
    point_7 = label[:, 0:3] + (pitch_rotation_matrix @ np.stack([- l/2, - h, + w/2], axis=1)).squeeze(-1)
    point_8 = label[:, 0:3] + (pitch_rotation_matrix @ np.stack([+ l/2, - h, + w/2], axis=1)).squeeze(-1)

    corners_array = np.stack([point_1, point_2, point_3, point_4, point_5, point_6, point_7, point_8], axis=1)

    if target:
        corners_array[zerp_target_points] = 0

    return corners_array

def get_iou(pred, target):
    '''
    Task 1
    input
        pred (N,7) 3D bounding box corners
        target (M,7) 3D bounding box corners
    output
        iou (N,M) pairwise 3D intersection-over-union
    '''
    # get the volumn
    volumn_targ = np.prod(target[:, 3:6], axis=1)
    volumn_pred = np.prod(pred[:, 3:6], axis=1)
    
    pred = label2corners(pred)
    target = label2corners(target, True)
    iou = []
    for i, prediction in enumerate(pred):
        iou_row = []
        # get only x and z and get y seperatly
        pred_y_low = np.min([prediction[0,1], prediction[4,1]])
        pred_y_high = np.max([prediction[0,1], prediction[4,1]])
        pred_polygon = Polygon([[prediction[0,0], prediction[0,2], 0],
                                [prediction[1,0], prediction[1,2], 0],
                                [prediction[2,0], prediction[2,2], 0],
                                [prediction[3,0], prediction[3,2], 0]])

        for j, targ in enumerate(target):
            target_y_low = np.min([targ[0,1], targ[4,1]])
            target_y_high = np.max([targ[0,1], targ[4,1]])

            # if their y do not intersect
            if target_y_low > pred_y_high or target_y_high < pred_y_low:
                iou_row.append(0)
            else:
                target_polygon = Polygon([[targ[0,0], targ[0,2], 0],
                                          [targ[1,0], targ[1,2], 0],
                                          [targ[2,0], targ[2,2], 0],
                                          [targ[3,0], targ[3,2], 0]])
                
                intersection = target_polygon.intersection(pred_polygon).area

                # take cases
                if target_y_low > pred_y_low:
                    if pred_y_high > target_y_high:
                        height = target_y_high - target_y_low
                    else:
                        height = pred_y_high - target_y_low
                else:
                    if target_y_high > pred_y_high:
                        height = pred_y_high - pred_y_low
                    else:
                        height = target_y_high - pred_y_low

                intersection_3d = height * intersection
                union_3d = volumn_pred[i] + volumn_targ[j] - intersection_3d

                iou_row.append(intersection_3d / union_3d)
            
        iou.append(iou_row)

    iou_array = np.array(iou)
    if iou_array.shape[0] == 0:
        if len(target) == 0:
            return np.zeros((0,0))
        else:
            return np.zeros((0,len(target)))

    return iou_array

def compute_recall(pred, target, threshold):
    '''
    Task 1
    input
        pred (N,7) proposed 3D bounding box labels
        target (M,7) ground truth 3D bounding box labels
        threshold (float) threshold for positive samples
    output
        recall (float) recall for the scene
    '''

    iou_matrix = get_iou(pred, target)

    # get the valid best predictions
    iou_matrix = np.max(iou_matrix, axis=0)

    TP = (iou_matrix >= threshold).sum()
    FN = (iou_matrix < threshold).sum()
    
    return TP / (TP + FN)