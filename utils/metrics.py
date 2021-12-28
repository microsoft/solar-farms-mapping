'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import numpy as np


def IoU(pred_segm, gt_segm):
    intersection = np.logical_and(gt_segm, pred_segm)
    union = np.logical_or(gt_segm, pred_segm)
    iou_score = np.sum(intersection) / (np.sum(union) + 0.001)
    return iou_score


def precision(pred, true, label=1):
    tp = np.sum((pred == label) & (true == label))
    fp = np.sum((pred == label) & (true != label))
    result = np.true_divide(tp, (tp + fp + 0.00001))
    return result


def tp_fp_fn(pred, true, label=1):
    tp = np.sum((pred == label) & (true == label))
    fp = np.sum((pred == label) & (true != label))
    fn = np.sum((pred != label) & (true == label))
    return tp, fp, fn


def recall(pred, true, label=1):
    tp = np.sum((pred == label) & (true == label))
    fn = np.sum((pred != label) & (true == label))
    result = np.true_divide(tp, (tp + fn + 0.00001))
    return result
    

def pixel_accuracy(pred_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(pred_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    pred_mask, gt_mask = extract_both_masks(pred_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, _ in enumerate(cl):
        curr_pred_mask = pred_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_pred_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

    if (sum_t_i == 0):
        pixel_accuracy = 0
    else:
        pixel_accuracy = sum_n_ii / sum_t_i

    return pixel_accuracy


def mean_accuracy(pred_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(pred_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    pred_mask, gt_mask = extract_both_masks(pred_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, _ in enumerate(cl):
        curr_eval_mask = pred_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)

        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy = np.mean(accuracy)
    return mean_accuracy


def mean_IoU(pred_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(pred_segm, gt_segm)

    cl, n_cl = union_classes(pred_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    pred_mask, gt_mask = extract_both_masks(pred_segm, gt_segm, cl, n_cl)

    IoU = list([0]) * n_cl

    for i, _ in enumerate(cl):
        curr_pred_mask = pred_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_pred_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_pred_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_pred_mask)

        IoU[i] = n_ii / (t_i + n_ij - n_ii)

    _mean_IoU = np.sum(IoU) / n_cl_gt
    return _mean_IoU


def frequency_weighted_IoU(pred_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(pred_segm, gt_segm)

    cl, n_cl = union_classes(pred_segm, gt_segm)
    pred_mask, gt_mask = extract_both_masks(pred_segm, gt_segm, cl, n_cl)

    _frequency_weighted_IoU = list([0]) * n_cl

    for i, _ in enumerate(cl):
        curr_pred_mask = pred_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_pred_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_pred_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_pred_mask)

        _frequency_weighted_IoU[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

    sum_k_t_k = get_pixel_area(pred_segm)

    _frequency_weighted_IoU = np.sum(_frequency_weighted_IoU) / sum_k_t_k
    return _frequency_weighted_IoU


'''
Auxiliary functions used during evaluation.
'''

def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(pred_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(pred_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)
    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)
    return cl, n_cl


def union_classes(pred_segm, gt_segm):
    eval_cl, _ = extract_classes(pred_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)
    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))
    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c
    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise SizeMismatchErr("DiffDim: Different dimensions of matrices!")


'''
Exceptions
'''
class SizeMismatchErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)




def compute_pre_rec(gt,mask,mybins=np.arange(0,256)):

    if(len(gt.shape)<2 or len(mask.shape)<2):
        print("ERROR: gt or mask is not matrix!")
        exit()
    if(len(gt.shape)>2): # convert to one channel
        gt = gt[:,:,0]
    if(len(mask.shape)>2): # convert to one channel
        mask = mask[:,:,0]
    if(gt.shape!=mask.shape):
        print("ERROR: The shapes of gt and mask are different!")
        exit()

    gtNum = gt[gt>128].size # pixel number of ground truth foreground regions
    pp = mask[gt>128] # mask predicted pixel values in the ground truth foreground region
    nn = mask[gt<=128] # mask predicted pixel values in the ground truth bacground region

    pp_hist,pp_edges = np.histogram(pp,bins=mybins) #count pixel numbers with values in each interval [0,1),[1,2),...,[mybins[i],mybins[i+1]),...,[254,255)
    nn_hist,nn_edges = np.histogram(nn,bins=mybins)

    pp_hist_flip = np.flipud(pp_hist) # reverse the histogram to the following order: (255,254],...,(mybins[i+1],mybins[i]],...,(2,1],(1,0]
    nn_hist_flip = np.flipud(nn_hist)

    pp_hist_flip_cum = np.cumsum(pp_hist_flip) # accumulate the pixel number in intervals: (255,254],(255,253],...,(255,mybins[i]],...,(255,0]
    nn_hist_flip_cum = np.cumsum(nn_hist_flip)

    precision = pp_hist_flip_cum/(pp_hist_flip_cum + nn_hist_flip_cum+1e-8) #TP/(TP+FP)
    recall = pp_hist_flip_cum/(gtNum+1e-8) #TP/(TP+FN)

    precision[np.isnan(precision)]= 0.0
    recall[np.isnan(recall)] = 0.0

    return np.reshape(precision,(len(precision))),np.reshape(recall,(len(recall)))



if __name__ == "__main__":
    a = np.array([0, 0, 0])
    b = np.array([0, 0, 0])
    print(IoU(a, b))