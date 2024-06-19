import numpy as np

from utils.task1 import get_iou

def nms(pred, score, threshold):
    '''
    Task 5
    Implement NMS to reduce the number of predictions per frame with a threshold
    of 0.1. The IoU should be calculated only on the BEV.
    input
        pred (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
        score (N,) confidence scores
        threshold (float) upper bound threshold for NMS
    output
        s_f (M,7) 3D bounding boxes after NMS
        c_f (M,1) corresopnding confidence scores
    '''
    
    # Initialize final predictions and confidences
    final_predictions = []
    final_confidences = []

    # Convert predictions to BEV format
    bev_predictions = pred.copy()
    bev_predictions[:, [1, 3]] = 0, 1

    while len(pred) > 0:
        # Find index of prediction with the highest confidence
        max_confidence_index = np.argmax(score)
        
        # Select and remove the prediction with the highest confidence
        selected_prediction = pred[max_confidence_index]
        selected_bev_prediction = bev_predictions[max_confidence_index]
        pred = np.delete(pred, max_confidence_index, axis=0)
        bev_predictions = np.delete(bev_predictions, max_confidence_index, axis=0)
        selected_confidence = score[max_confidence_index]
        score = np.delete(score, max_confidence_index)

        # Append selected prediction and confidence to the final lists
        final_predictions.append(selected_prediction)
        final_confidences.append(selected_confidence)

        # Calculate IoU with the remaining predictions
        ious = get_iou(bev_predictions, selected_bev_prediction.reshape(-1, pred.shape[1]))

        # Remove predictions with IoU above the threshold
        overlapping_indices = np.where(ious >= threshold)[0]
        pred = np.delete(pred, overlapping_indices, axis=0)
        bev_predictions = np.delete(bev_predictions, overlapping_indices, axis=0)
        score = np.delete(score, overlapping_indices)
        
        s_f = np.array(final_predictions)
        c_f =  np.array(final_confidences).reshape(-1, 1)

    return s_f, c_f