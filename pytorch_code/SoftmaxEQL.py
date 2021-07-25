import torch
import pickle
import numpy as np
import torch.nn.functional as F
from collections import Counter


# def get_eql_class_weights(lambda_):
#     # initialize class weights array, there are total of 1000 different classes
#     class_weights = np.zeros(1000)
#     labels = []
#     with open('ImageNet_LT_train.txt', 'r') as f:
#         for line in f:
#             _, label = line.split()
#             labels.append(int(label))
#     # count the occurrence of each label
#     label_count = Counter(labels)
#     for idx, (label, count) in enumerate(sorted(label_count.items(), key=lambda x: -x[1])):
#         # if number of count is more than threshold, then set the weight to 1
#         class_weights[label] = 1 if count > lambda_ else 0
#         print('idx: {}, cls: {} img: {}, weight: {}'.format(idx, label, count, class_weights[label]))
#     return class_weights


def get_eql_class_weights(opt):
    # read all labels
    labels = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))[1]
    labels = [l - 1 for l in labels]  # minus 1 since 0 value is used for padding and it is not used during training
    # count the occurrence of each label
    label_count = Counter(labels)
    num_classes = max(set(labels)) + 1
    # initialize class weights array
    class_weights = np.zeros(num_classes)
    for idx, (label, count) in enumerate(sorted(label_count.items(), key=lambda x: -x[1])):
        # if number of count is more than threshold, then set the weight to 1
        if count < opt.lambda_low:
            class_weights[label] = 0
        elif count >= opt.lambda_low and count <= opt.lambda_high:
            class_weights[label] = 0.5
        else:
            class_weights[label] = 1
        print('idx: {}, cls: {} count: {}, weight: {}'.format(idx, label, count, class_weights[label]))
    return class_weights


def replace_masked_values(tensor, mask, replace_with):
    assert tensor.dim() == mask.dim(), '{} vs {}'.format(tensor.shape, mask.shape)
    one_minus_mask = 1 - mask
    values_to_add = replace_with * one_minus_mask
    return tensor * mask + values_to_add


class SoftmaxEQL(object):
    def __init__(self, ignore_prob, opt):
        self.ignore_prob = ignore_prob
        self.class_weight = torch.Tensor(get_eql_class_weights(opt)).cuda()

    def __call__(self, input, target):
        N, C = input.shape  # batch__size x number_node
        # reshape the class_weight array
        not_ignored = self.class_weight.view(1, C).repeat(N, 1)
        # generate list of random numbers and compare with a threshold
        over_prob = (torch.rand(input.shape).cuda() > self.ignore_prob).float()
        is_gt = target.new_zeros((N, C)).float()
        # set the value at target index position to 1
        is_gt[torch.arange(N), target] = 1

        # if less than or equal to zero, then weight = 0
        weights = ((not_ignored + over_prob + is_gt) > 0).float()
        # replace with zero
        new_input = replace_masked_values(input, weights, -1e7)
        loss = F.cross_entropy(new_input, target)
        return loss
