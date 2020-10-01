import numpy as np
import torch
import train
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy

def test_with_dataloader(model, test_loader, weight=None):
    all_score = []
    all_gt = []
    all_frame = []
    all_name = []
    model.eval()
    for i, data in enumerate(test_loader):
        inputs, gts, frames, names = data
        inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))
        pred = model(inputs, vars=weight)
        all_score.append(pred.cpu().detach().numpy())
        all_gt += gts
        all_frame += frames
        all_name += names
    model.train()
    num_video = len(all_frame)
    all_score = np.concatenate(all_score).reshape(num_video, -1)
    return all_score, all_gt, all_frame, all_name


def get_auc(score_by_frame, gt_by_frame):
    idx = np.argsort(-score_by_frame)
    tp = gt_by_frame[idx] > 0
    fp = gt_by_frame[idx] == 0

    cumsum_tp = np.cumsum(tp)
    cumsum_fp = np.cumsum(fp)

    nrpos = np.sum(gt_by_frame)
    rec = cumsum_tp / nrpos
    fpr = cumsum_fp / np.sum(gt_by_frame == 0)

    auc = np.trapz(rec, fpr)
    return auc


def get_fpr(score_by_frame, gt_by_frame, threshold=0.5):
    pred = (score_by_frame > threshold).astype(int)
    n_neg = np.sum(gt_by_frame==0)
    fp = ((pred == 1).astype(int) * (pred != gt_by_frame).astype(int)).sum()
    return fp/n_neg


def seg2frame(actual_frames, score, gt, fixed_len=True):
    detection_score_per_frame = np.zeros(actual_frames)
    if not fixed_len:
        for i in range((actual_frames-1)//16+1):
            detection_score_per_frame[i*16:(i+1)*16] = score[i]
    else:
        thirty2_shots = np.round(np.linspace(0, actual_frames//16, 33))
        for i in range(len(thirty2_shots)-1):
            ss = int(thirty2_shots[i])
            ee = int(thirty2_shots[i+1])-1

            if ee<ss:
                detection_score_per_frame[ss*16:(ss+1)*16] = score[i]
            else:
                detection_score_per_frame[ss*16:(ee+1)*16] = score[i]

    gt_by_frame = np.zeros(actual_frames)

    # check index start 0 or 1
    for i in range(len(gt)//2):
        s = gt[i*2]
        e = min(gt[i*2+1], actual_frames)
        gt_by_frame[s-1:e] = 1
    return detection_score_per_frame, gt_by_frame


def calc_auc(all_score, all_gt, all_num_frame):
    assert len(all_score) == len(all_gt) and len(all_gt) == len(all_num_frame)
    score_by_frame = []
    gt_by_frame = []

    for i in range(len(all_score)):
        if len(all_gt[i]) == 0:
            continue
        score, gt = seg2frame(all_num_frame[i], all_score[i], all_gt[i], True)
        score_by_frame.append(score)
        gt_by_frame.append(gt)

    score_by_frame = np.concatenate(score_by_frame)
    gt_by_frame = np.concatenate(gt_by_frame)

    auc = get_auc(score_by_frame, gt_by_frame)
    score_round = np.round(score_by_frame)
    correct = np.sum(np.equal(score_round,gt_by_frame).astype(float))
    acc = correct/np.shape(gt_by_frame)[0]
    return auc, acc


def calc_auc_per_video(all_score, all_gt, all_num_frame):
    assert len(all_score) == len(all_gt) and len(all_gt) == len(all_num_frame)
    auc_list = []
    for i in range(len(all_score)):
        if len(all_gt[i]) == 0:
            continue
        if all_gt[i][1] - all_gt[i][0] + 1 == all_num_frame[i]:
            continue
        score, gt = seg2frame(all_num_frame[i], all_score[i], all_gt[i], True)
        auc = get_auc(score, gt)
        auc_list.append(auc)

    auc = np.mean(np.array(auc_list))
    return auc


def calc_fpr(all_score, all_gt, all_num_frame):
    score_by_frame = []
    gt_by_frame = []

    for i in range(len(all_score)):
        score, gt = seg2frame(all_num_frame[i], all_score[i], all_gt[i], True)
        score_by_frame.append(score)
        gt_by_frame.append(gt)

    score_by_frame = np.concatenate(score_by_frame)
    gt_by_frame = np.concatenate(gt_by_frame)

    fpr = get_fpr(score_by_frame, gt_by_frame)

    return fpr


def eval_with_cv(model, origin_train_loader, cv_loader_lst, eval_loader, criterion, args):
    auc_per_epoch_lst = []
    for train_loader, valid_loader in cv_loader_lst:
        auc_per_epoch = validation(model, train_loader, valid_loader, criterion, args, args.ft_max_epoch)
        auc_per_epoch_lst.append(np.array(auc_per_epoch).argsort().argsort())  # rank
    avg_auc_per_epoch = np.mean(np.array(auc_per_epoch_lst), axis=0)
    best_epoch_idx = np.argmax(avg_auc_per_epoch)
    auc_per_epoch = validation(model, origin_train_loader, eval_loader, criterion, args, args.ft_max_epoch)

    return auc_per_epoch[best_epoch_idx], best_epoch_idx


def validation(model, trainloader, an_validloader, criterion, args, inner_epoch):
    auc_per_epoch = []
    device = torch.device('cuda')
    tmp_model = deepcopy(model)

    optimizer = torch.optim.Adam(tmp_model.parameters(), lr=args.ft_lr, weight_decay=args.w_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience,
                                  threshold=args.threshold, threshold_mode='abs')
    tmp_model.train()
    inputs, labels = next(iter(trainloader))
    inputs = inputs.view(-1, inputs.size(-1)).to(device)
    labels = labels.view(-1).int().to(device)

    fast_weights = None
    optimizer = torch.optim.Adam(tmp_model.parameters(), lr=args.ft_lr, weight_decay=args.w_decay)

    all_score, all_gt, all_frame, _ = test_with_dataloader(tmp_model, an_validloader, fast_weights)
    auc = calc_auc_per_video(all_score, all_gt, all_frame)
    auc_per_epoch.append(auc)

    for i in range(inner_epoch):
        train.train_iter(tmp_model, inputs, labels, criterion, optimizer)
        if i % args.valid_step == 0:
            all_score, all_gt, all_frame, _ = test_with_dataloader(tmp_model, an_validloader, fast_weights)
            auc = calc_auc_per_video(all_score, all_gt, all_frame)
            scheduler.step(auc)
            auc_per_epoch.append(auc)

    return auc_per_epoch


def metatest(model, ft_loader, cv_loader_list, eval_loader, criterion, args, sampling=False):
    '''
    1. if is_sampling == True --> M_s or M_g (For M_g, we manually find best model
    by collecting outputs of different experiments)
    2. if is_sampling == False
        2.1 if args.chpt is None --> S (Scratch)
        2.2 else --> P (Pretrain)
    '''
    if sampling:
        # Meta-model test code
        # Select best model sampled every 300 epochs
        start_epoch, end_epoch = 0, 3000
        epoch_step = 300
        best_epoch, best_auc = -1, -1
        for i in range(start_epoch, end_epoch + 1, epoch_step):
            if args.chpt is not None:
                # Name of chpt file is like '{}epochs_exp1_seed1_lr0.001_split1.pkl'
                model_path = os.path.join('meta_model', args.chpt.format(i))
                model.load_state_dict(torch.load(model_path))
            auc, _ = eval_with_cv(model, ft_loader, cv_loader_list, eval_loader, criterion, args)
            print(auc)
            if auc > best_auc:
                best_auc = auc
                best_epoch = i
        print("Best AUC: {} at {} meta-iters".format(best_auc, best_epoch))
    else:
        # Pretrain (baseline) test code
        if args.chpt is not None:
            model_path = os.path.join('pretrain', args.chpt)
            model.load_state_dict(torch.load(model_path))

        auc, best_epoch = eval_with_cv(model, ft_loader, cv_loader_list, eval_loader, criterion, args)
        print("Best AUC: {} ".format(auc))

