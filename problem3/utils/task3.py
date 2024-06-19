import numpy as np

from .task1 import get_iou

import numpy as np
import math

def sample_or_duplicate(arrays, threshold):
    N_samples = len(arrays[0])

    if N_samples > threshold:
        # Randomly sample indices with replace false
        sampled_indices = np.random.choice(N_samples, threshold, replace=False)
    else:
        # Randomly duplicate indices with replace true
        sampled_indices = np.random.choice(N_samples, threshold, replace=True)
    sampled_arrays = [element[sampled_indices] for element in arrays]
    return sampled_arrays

def concatenate_arrays(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("The lists must have the same number of arrays of xyz, features etc..")
    
    concatenated_arrays = []
    for arr1, arr2 in zip(list1, list2):
        if arr1.shape[1:] != arr2.shape[1:]:
            raise ValueError("Array dimensions must match, except for the first dimension wich is the rows")
        concatenated = np.concatenate((arr1, arr2), axis=0)
        concatenated_arrays.append(concatenated)
    return concatenated_arrays


def background_sampling_last_case(easy_background_samples, hard_background_samples, remainig_background_sample_number):
    easy_sampled = math.ceil(remainig_background_sample_number / 2)
    hard_sampled = remainig_background_sample_number - easy_sampled
    sampled_easy_background = sample_or_duplicate(easy_background_samples, easy_sampled)
    sampled_hard_background = sample_or_duplicate(hard_background_samples, hard_sampled)
    return concatenate_arrays(sampled_hard_background, sampled_easy_background)

def sample_background(easy_background_samples, hard_background_samples, easy_background_samples_number, hard_background_samples_number, remainig_background_sample_number):
    if hard_background_samples_number == 0:
                sampled_background = sample_or_duplicate(easy_background_samples, remainig_background_sample_number)
    elif easy_background_samples_number == 0:
        sampled_background = sample_or_duplicate(hard_background_samples, remainig_background_sample_number)
    else:
        sampled_background = background_sampling_last_case(easy_background_samples, hard_background_samples, remainig_background_sample_number)
    return sampled_background


def sample_proposals(pred, target, xyz, feat, config, train=False):
    '''
    Task 3
    a. Using the highest IoU, assign each proposal a ground truth annotation. For each assignment also
       return the IoU as this will be required later on.
    b. Sample 64 proposals per scene. If the scene contains at least one foreground and one background
       proposal, of the 64 samples, at most 32 should be foreground proposals. Otherwise, all 64 samples
       can be either foreground or background. If there are less background proposals than 32, existing
       ones can be repeated.
       Furthermore, of the sampled background proposals, 50% should be easy samples and 50% should be
       hard samples when both exist within the scene (again, can be repeated to pad up to equal samples
       each). If only one difficulty class exists, all samples should be of that class.
    input
        pred (N,7) predicted bounding box labels
        target (M,7) ground truth bounding box labels
        xyz (N,512,3) pooled point cloud
        feat (N,512,C) pooled features
        config (dict) data config containing thresholds
        train (string) True if training
    output
        assigned_targets (64,7) target box for each prediction based on highest iou
        xyz (64,512,3) indices 
        feat (64,512,C) indiceses
        iou (64,) iou of each prediction and its assigned target box
    useful config hyperparameters
        config['t_bg_hard_lb'] threshold background lower bound for hard difficulty
        config['t_bg_up'] threshold background upper bound
        config['t_fg_lb'] threshold foreground lower bound
        config['num_fg_sample'] maximum allowed number of foreground samples
        config['bg_hard_ratio'] background hard difficulty ratio (#hard samples/ #background samples)
    '''
    iou = get_iou(pred, target)
    best_detection = np.argmax(iou, axis=1)
    assigned_targets = target[best_detection, :]
    best_iou = iou.max(axis=1)

    if train == True:
        """
        Furthermore, for 2-stage detectors, each bounding box that has the highest
        IoU for a ground truth bounding box is considered an additional foreground sample regardless
        of its IoU (assuming > 0).
        """
        foreground_samples_indices = np.where(best_iou >= config["t_fg_lb"])[0]
        best_assignment_of_every_label = np.argmax(iou, axis=0)
        best_assignment_of_every_label_iou = best_iou[best_assignment_of_every_label]
        indeces_of_labels_with_low_iou = np.where(best_assignment_of_every_label_iou < config["t_fg_lb"])[0]
        indeces_of_labels_with_low_iou = best_assignment_of_every_label[indeces_of_labels_with_low_iou]
        list_total = foreground_samples_indices.tolist()
        list_extra = indeces_of_labels_with_low_iou.tolist()
        list_total.extend(list_extra)
        
        foreground_samples_indices = np.array(list_total)
        foreground_samples = [foreground_samples_indices, assigned_targets[foreground_samples_indices], xyz[foreground_samples_indices], feat[foreground_samples_indices], best_iou[foreground_samples_indices]]

        hard_background_samples_indices = np.where(np.logical_and(best_iou < config["t_bg_up"], best_iou >= config["t_bg_hard_lb"]))[0]
        hard_background_samples = [hard_background_samples_indices, assigned_targets[hard_background_samples_indices], xyz[hard_background_samples_indices], feat[hard_background_samples_indices], best_iou[hard_background_samples_indices]]

        easy_background_samples_indices = np.where(best_iou < config["t_bg_hard_lb"])[0]
        easy_background_samples = [easy_background_samples_indices, assigned_targets[easy_background_samples_indices], xyz[easy_background_samples_indices], feat[easy_background_samples_indices], best_iou[easy_background_samples_indices]]

        total_pred = pred.shape[0]
        foreground_samples_number = foreground_samples_indices.shape[0]
        hard_background_samples_number = hard_background_samples_indices.shape[0]
        easy_background_samples_number = easy_background_samples_indices.shape[0]


        # cases
        if foreground_samples_number >= total_pred:
            print("case1")
            return sample_or_duplicate(foreground_samples, config["num_samples"])[1:]
        

        if (easy_background_samples_number + hard_background_samples_number) >= total_pred:
            print("case2", total_pred)
            return sample_background(easy_background_samples, hard_background_samples, easy_background_samples_number, hard_background_samples_number, config["num_samples"])[1:]
        

        if foreground_samples_number >= config["num_fg_sample"]:
            print("case3-1")
            sampled_foreground = sample_or_duplicate(foreground_samples, config["num_fg_sample"])
            # sample background 
            sampled_background = sample_background(easy_background_samples, hard_background_samples, easy_background_samples_number, hard_background_samples_number, config["num_fg_sample"])
            return concatenate_arrays(sampled_foreground, sampled_background)[1:]
        
        
        if foreground_samples_number < config["num_fg_sample"]:
            print("case3-2")
            remainig_background_sample_number = config["num_samples"] - foreground_samples_number
            # sample background 
            sampled_background = sample_background(easy_background_samples, hard_background_samples, easy_background_samples_number, hard_background_samples_number, remainig_background_sample_number)
            return concatenate_arrays(foreground_samples, sampled_background)[1:]

    
    return assigned_targets, xyz, feat, best_iou