import torch
import torch.nn as nn

class RegressionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.SmoothL1Loss()

    def forward(self, pred, target, iou):
        '''
        Task 4.a
        We do not want to define the regression loss over the entire input space.
        While negative samples are necessary for the classification network, we
        only want to train our regression head using positive samples. Use 3D
        IoU ≥ 0.55 to determine positive samples and alter the RegressionLoss
        module such that only positive samples contribute to the loss.
        input
            pred (N,7) predicted bounding boxes
            target (N,7) target bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_reg_lb'] lower bound for positive samples
        '''
        device = pred.device
        positive_reg_lb_idx = torch.where(iou >= self.config['positive_reg_lb'])[0]
        
        
        if len(positive_reg_lb_idx) > 0:
            return self.loss(pred[positive_reg_lb_idx][:, :3], target[positive_reg_lb_idx][:, :3]) + 3 * self.loss(pred[positive_reg_lb_idx][:, 3:6], target[positive_reg_lb_idx][:, 3:6]) + self.loss(pred[positive_reg_lb_idx][:, 6], target[positive_reg_lb_idx][:, 6])
        else:
            return torch.zeros(1, device=device)

class ClassificationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.BCELoss()

    def forward(self, pred, iou):
        '''
        Task 4.b
        Extract the target scores depending on the IoU. For the training
        of the classification head we want to be more strict as we want to
        avoid incorrect training signals to supervise our network.  A proposal
        is considered as positive (class 1) if its maximum IoU with ground
        truth boxes is ≥ 0.6, and negative (class 0) if its maximum IoU ≤ 0.45.
            pred (N,7) predicted bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_cls_lb'] lower bound for positive samples
            self.config['negative_cls_ub'] upper bound for negative samples
        '''
        device = pred.device

        negative_cls_ub_idx = torch.where(iou <= self.config['negative_cls_ub'])[0]
        positive_cls_lb_idx = torch.where(iou >= self.config['positive_cls_lb'])[0]
        idxs = torch.sort(torch.cat((negative_cls_ub_idx, positive_cls_lb_idx), dim=0))[0]


        if len(idxs) > 0:
            target = torch.zeros((iou.size(0), 1), device=device)
            target[positive_cls_lb_idx] = 1

            return self.loss(pred[idxs], target[idxs])
        else:
            return torch.zeros(1, device=device)