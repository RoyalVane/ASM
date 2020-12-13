import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

mce_loss = nn.MSELoss()


def channel_1toN(img, num_channel):
    T = torch.LongTensor(num_channel, img.shape[1], img.shape[2]).zero_()
    mask = torch.LongTensor(img.shape[1], img.shape[2]).zero_()
    for i in range(num_channel):
        T[i] = T[i] + i
        layer = T[i] - img
        T[i] = torch.from_numpy(np.logical_not(np.logical_xor(layer.numpy(), mask.numpy())).astype(int))
    return T.float()


class WeightedBCEWithLogitsLoss(nn.Module):
    
    def __init__(self, size_average=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.size_average = size_average
        
    def weighted(self, input, target, weight, alpha, beta):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
                
        if weight is not None:
            loss = alpha * loss + beta * loss * weight

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
    
    def forward(self, input, target, weight, alpha, beta):
        if weight is not None:
            return self.weighted(input, target, weight, alpha, beta)
        else:
            return self.weighted(input, target, None, alpha, beta)


# =============================================================================
# class CrossEntropy2d(nn.Module):
#     
#     def __init__(self, class_num, alpha=None, gamma=2, size_average=True, ignore_label=255):
#         super(CrossEntropy2d, self).__init__()
#         if alpha is None:
#             self.alpha = Variable(torch.ones(class_num, 1))
#         else:
#             if isinstance(alpha, Variable):
#                 self.alpha = alpha
#             else:
#                 self.alpha = Variable(alpha)
#         self.gamma = gamma
#         self.class_num = class_num
#         self.size_average = size_average
#         self.ignore_label = ignore_label
# 
#     def forward(self, predict, target):
#         N, C, H, W = predict.size()
#         sm = nn.Softmax2d()
#         
#         P = sm(predict)
#         P = torch.clamp(P, min = 1e-9, max = 1-(1e-9))
#         
#         target_mask = (target >= 0) * (target != self.ignore_label)
#         target = target[target_mask].view(1, -1)
#         predict = P[target_mask.view(N, 1, H, W).repeat(1, C, 1, 1)].view(C, -1)
#         probs = torch.gather(predict, dim = 0, index = target)
#         log_p = probs.log()
#         batch_loss = -(torch.pow((1-probs), self.gamma))*log_p 
# 
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             
#             loss = batch_loss.sum()
#         return loss
# =============================================================================

        
class CrossEntropy2d(nn.Module):
    
    def __init__(self, class_num=19, alpha=None, gamma=2, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target):
        N, C, H, W = predict.size()
        sm = nn.Softmax(dim = 0)
        predict = predict.transpose(0, 1).contiguous()
        P = sm(predict)
        P = torch.clamp(P, min = 1e-9, max = 1-(1e-9))
        
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = P[target_mask.view(1, N, H, W).repeat(C, 1, 1, 1)].view(C, -1)
        probs = torch.gather(predict, dim = 0, index = target.view(1, -1))
        log_p = probs.log()
        batch_loss = -(torch.pow((1-probs), self.gamma))*log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            
            loss = batch_loss.sum()
        return loss


# =============================================================================
# class CrossEntropy2d(nn.Module):
# 
#     def __init__(self, size_average=True, reduce=None, ignore_label=255):
#         super(CrossEntropy2d, self).__init__()
#         self.size_average = size_average
#         self.ignore_label = ignore_label
#         self.reduce = reduce
# 
#     def forward(self, predict, target, weight=None):
#         """
#             Args:
#                 predict:(n, c, h, w)
#                 target:(n, h, w)
#                 weight (Tensor, optional): a manual rescaling weight given to each class.
#                                            If given, has to be a Tensor of size "nclasses"
#         """
#         assert not target.requires_grad
#         assert predict.dim() == 4
#         assert target.dim() == 3
#         assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
#         assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
#         assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
#         n, c, h, w = predict.size()
#         target_mask = (target >= 0) * (target != self.ignore_label)
#         target = target[target_mask] #Dim = 1
#         if not target.data.dim():
#             return Variable(torch.zeros(1))
#         predict = predict.transpose(1, 2).transpose(2, 3).contiguous()        
#         predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
#         loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average, reduce = self.reduce)
#         return loss
# =============================================================================
