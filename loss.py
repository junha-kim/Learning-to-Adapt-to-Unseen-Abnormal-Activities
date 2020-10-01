import torch


def custom_objective(y_pred, y_true):
    '''
    :param y_true: batch_size
    :param y_pred: batch_size x num_segment
    :return:
    '''
    'Custom Objective function'
    n_seg = 32
    batchsize = len(y_true)
    n_exp = batchsize // 2 # number of abnormal/normal videos, actually same
    labels = torch.empty(batchsize) # sum of label for each label
    max_score_per_bag = torch.empty(batchsize) # the highest score in a bag (video).
    sparsity = torch.empty(batchsize) # the summation of score in a bag (video), for sparsity term
    temp_smooth = torch.Tensor(batchsize)

    device = torch.device('cuda')
    labels = labels.to(device)
    max_score_per_bag = max_score_per_bag.to(device)
    sparsity = sparsity.to(device)
    temp_smooth = temp_smooth.to(device)

    for i in range(batchsize):
        # For Labels
        labels[i] = y_true[i] # Just to keep track of abnormal and normal vidoes, all label in a bag must be same

        # For Features scores
        Feat_Score = y_pred[i * n_seg:i * n_seg + n_seg] # scores of a Bag
        max_score_per_bag[i] = torch.max(Feat_Score) # Keep the maximum score of scores of all instances in a Bag (video)
        sparsity[i] = torch.sum(Feat_Score) # Keep the sum of scores of all instances in a Bag (video)

        z1 = torch.ones(1,1)
        z1 = z1.to(device)
        z2 = torch.cat([z1, Feat_Score])
        z3 = torch.cat([Feat_Score, z1])
        z = z2 - z3
        temp_smooth[i] = torch.sum(z[1:32] ** 2)

    # only compute for abnormal cases
    sparsity = sparsity[:n_exp]
    temp_smooth = temp_smooth[:n_exp]
    #
    # max_score_per_bag = max_score_per_bag

    indx_nor = torch.eq(labels, 0)
    indx_abn = torch.eq(labels, 1)
    max_score_norm = max_score_per_bag[indx_nor] # Maximum Score for each of abnormal video
    max_score_abnorm = max_score_per_bag[indx_abn] # Maximum Score for each of normal video

    hinge_loss = 0
    for i in range(n_exp):
        hinge_loss += torch.max(torch.zeros_like(max_score_abnorm).to(device),\
                                torch.ones_like(max_score_abnorm)*1.0-max_score_abnorm + max_score_norm[i]).sum()
    sparsity_loss = sparsity.sum()
    temp_smooth_loss = temp_smooth.sum()
    total_loss = hinge_loss/n_exp + 0.00008 * sparsity_loss + 0.00008 * temp_smooth_loss
    return total_loss
