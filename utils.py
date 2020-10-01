import os
import pickle
import numpy as np
import random
import torch

def load_all_data(args):
    if args.dataset == 'UCF-Crime':
        '''
        abnorm_train_dict: train data split for all classes. Used as train data for pretrain,
         support/query set of meta-train, and support set of meta-test.
        abnorm_test_dict: test data split for all classes. Used as validation data for pretrain,
         and query set of meta-test (final evaluation)
        '''
        gt_dict = read_pickle_file(args.gt_dict_path)
        frame_dict = read_pickle_file(args.frame_dict_path)
        exclusion_lst = read_pickle_file(args.exclusion_list_path)

        train_classes, test_classes = get_class_split(args.seed)
        if args.all_class:
            # When train/test with all 13 classes, seen anomaly detection
            train_classes = test_classes = train_classes + test_classes
        abnorm_train_dict, abnorm_test_dict = get_abnormal_feat_dict(
            rgb_dir=args.rgb_dir, flow_dir=args.flow_dir, split_num=args.split_num,
            split_dir=args.split_dir,  dataset_name=args.dataset, feat_type=args.feat_type,
            exclusion=exclusion_lst)

        normal_dict = get_normal_feat_dict(
            rgb_dir=os.path.join(args.rgb_dir, args.normal_dir), flow_dir=os.path.join(args.flow_dir, args.normal_dir),
            split_dir=None, dataset_name=args.dataset, feat_type=args.feat_type, exclusion=exclusion_lst)
        normal_list = dict2list(normal_dict)
        normal_train_list, normal_test_list = normal_list[:800], normal_list[800:]
    elif args.dataset == 'ShanghaiTech':
        gt_dict = read_pickle_file(args.gt_dict_path)
        frame_dict = read_pickle_file(args.frame_dict_path)
        exclusion_lst = []

        train_classes, test_classes = [], ['abnormal']
        abnorm_train_dict, abnorm_test_dict = get_abnormal_feat_dict(
            rgb_dir=os.path.join(args.rgb_dir, 'abnormal'),
            flow_dir=os.path.join(args.flow_dir, 'abnormal'), split_num=None,
            split_dir=args.split_dir, dataset_name=args.dataset, feat_type=args.feat_type, exclusion=exclusion_lst)

        train_norm_feat_dict, test_norm_feat_dict = get_normal_feat_dict(
            rgb_dir=os.path.join(args.rgb_dir, args.normal_dir),
            flow_dir=os.path.join(args.flow_dir, args.normal_dir), split_dir=args.split_dir,
            dataset_name=args.dataset, feat_type=args.feat_type, exclusion=exclusion_lst)
        normal_train_list, normal_test_list = dict2list(train_norm_feat_dict), dict2list(test_norm_feat_dict)
    else:
        raise ValueError('Invalid argument: dataset')

    return abnorm_train_dict, abnorm_test_dict, normal_train_list, normal_test_list, train_classes, test_classes,\
           gt_dict, frame_dict


def get_class_split(idx):
    train_classes = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'RoadAccidents',
                 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
    test_classes = train_classes[idx-1]
    train_classes.remove(test_classes)
    return train_classes, [test_classes]


def read_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        ret_obj = pickle.load(f)
    return ret_obj


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_video_feats_from_splits(split_list, rgb_dir, flow_dir, dataset_name, feat_type, exclusion):
    video_feat_dict = {}

    for video_name_in_split_file in split_list:

        if dataset_name == 'UCF-Crime':
            if 'mp4' not in video_name_in_split_file or 'Normal' in video_name_in_split_file:
                continue
            video_name = video_name_in_split_file.split('/')[1].strip()[:-4]
            if video_name in exclusion:
                continue
            cls = video_name_in_split_file.split('/')[0].strip()
        else: # ShanghaiTech
            video_name = video_name_in_split_file.strip()
            if not os.path.exists(os.path.join(rgb_dir, video_name + '.npy')):
                continue
            cls = 'abnormal'

        if cls not in video_feat_dict:
            video_feat_dict[cls] = {}

        rgb_feat = np.load(os.path.join(rgb_dir, video_name_in_split_file.strip() + '.npy'))
        flow_feat = np.load(os.path.join(flow_dir, video_name_in_split_file.strip() + '.npy'))
        if feat_type == 'rgb':
            feats = [rgb_feat]
        elif feat_type == 'flow':
            feats = [flow_feat]
        elif feat_type == '2-stream':
            feats = [rgb_feat, flow_feat]
        else:
            raise ValueError('Invalid feat type argument')
        video_feat_dict[cls][video_name] = np.concatenate(feats, axis=1)

    return video_feat_dict


def get_abnormal_feat_dict(rgb_dir, flow_dir, split_num, split_dir, dataset_name, feat_type, exclusion):
    # return train/test data for abnormal videos
    # Ex: train_feat_dict[class_name] = {video_name1: segment_feature(dim: 32x1024), ...}, same as for test_feat_dict
    if dataset_name == 'UCF-Crime':
        train_txt_file = 'train_{}.txt'.format(format(split_num, '03d'))
        test_txt_file = 'test_{}.txt'.format(format(split_num, '03d'))
    else:
        train_txt_file = 'train.txt'
        test_txt_file = 'test.txt'
    train_path = os.path.join(split_dir, train_txt_file)
    test_path = os.path.join(split_dir, test_txt_file)
    train_split = open(train_path, 'r').readlines()
    test_split = open(test_path, 'r').readlines()

    train_abnorm_feat_dict = load_video_feats_from_splits(train_split, rgb_dir, flow_dir, dataset_name, feat_type,
                                                          exclusion)
    test_abnorm_feat_dict = load_video_feats_from_splits(test_split, rgb_dir, flow_dir, dataset_name, feat_type,
                                                         exclusion)

    return train_abnorm_feat_dict, test_abnorm_feat_dict


def get_normal_feat_dict(rgb_dir, flow_dir, split_dir, dataset_name, feat_type, exclusion):
    # return train/test data for normal videos
    # Ex: train_feat_dict[video_name] = [segment_feature(dim: 32x1024)], same as for test_feat_dict
    if dataset_name == 'UCF-Crime':
        all_norm_feat_dict = get_normal_feature_crime(rgb_dir=rgb_dir, flow_dir=flow_dir, feat_type=feat_type)
        return all_norm_feat_dict
    elif dataset_name == 'ShanghaiTech':
        train_norm_feat_dict, test_norm_feat_dict = get_normal_feature_shanghai(rgb_dir=rgb_dir, flow_dir=flow_dir,
                                                                                split_dir=split_dir,
                                                                                feat_type=feat_type)
        return train_norm_feat_dict, test_norm_feat_dict
    else:
        raise ValueError('Invalid feat type argument')


def get_normal_feature_crime(rgb_dir, flow_dir, feat_type):
    # Load all normal videos and first 800 videos for train(pretrain or meta-train),
    # last 150 videos for meta-test(fine-tuning, not for evaluation)
    all_norm_feat_dict = {}
    for npy_file in sorted(os.listdir(rgb_dir)):
        rgb_feat = np.load(os.path.join(rgb_dir, npy_file))
        flow_feat = np.load(os.path.join(flow_dir, npy_file))
        if feat_type == 'rgb':
            feat = [rgb_feat]
        elif feat_type == 'flow':
            feat = [flow_feat]
        elif feat_type == '2-stream':
            feat = [rgb_feat, flow_feat]

        video_name = npy_file[:-4]
        all_norm_feat_dict[video_name] = np.concatenate(feat, axis=1)
    return all_norm_feat_dict


def get_normal_feature_shanghai(rgb_dir, flow_dir, split_dir, feat_type):
    test_path = os.path.join(split_dir, 'test.txt')
    train_path = os.path.join(split_dir, 'train.txt')
    test_split = open(test_path, 'r').readlines()
    train_split = open(train_path, 'r').readlines()

    train_norm_feat_dict = {}
    test_norm_feat_dict = {}

    for video in train_split:
        video_name = video.strip()
        if not os.path.exists(os.path.join(rgb_dir, video_name) + '.npy') or \
                not os.path.exists(os.path.join(flow_dir, video_name) + '.npy'):
            continue

        rgb_feat = np.load(os.path.join(rgb_dir, video_name + '.npy'))
        flow_feat = np.load(os.path.join(flow_dir, video_name + '.npy'))
        if feat_type == 'rgb':
            feat = [rgb_feat]
        elif feat_type == 'flow':
            feat = [flow_feat]
        elif feat_type == '2-stream':
            feat = [rgb_feat, flow_feat]

        train_norm_feat_dict[video_name] = np.concatenate(feat, axis=1)

    for video in test_split:
        video_name = video.strip()
        if not os.path.exists(os.path.join(rgb_dir, video_name) + '.npy') or \
                not os.path.exists(os.path.join(flow_dir, video_name) + '.npy'):
            continue

        rgb_feat = np.load(os.path.join(rgb_dir, video_name + '.npy'))
        flow_feat = np.load(os.path.join(flow_dir, video_name + '.npy'))
        if feat_type == 'rgb':
            feat = [rgb_feat]
        elif feat_type == 'flow':
            feat = [flow_feat]
        elif feat_type == '2-stream':
            feat = [rgb_feat, flow_feat]

        test_norm_feat_dict[video_name] = np.concatenate(feat, axis=1)

    return train_norm_feat_dict, test_norm_feat_dict


def merge_dict(dict1, dict2):
    ret = {}
    for k, v in sorted(dict1.items()):
        ret[k] = v
    for k, v in sorted(dict2.items()):
        if k in ret:
            raise ValueError
        ret[k] = v

    return ret


def dict2list(dic):
    ret = []
    for _, v in sorted(dic.items()):
        if type(v) != dict:
            ret.append(v)
        else:
            lst = []
            for _, v2 in v.items():
                lst.append(v2)
            ret.append(np.array(lst))
    return ret


def dict_subset(dic, classes):
    ret = {}
    for k, v in sorted(dic.items()):
        if k in classes:
            ret[k] = v
    return ret


def get_class_splits(all_class_list, ratio=(7, 3, 3)):
    rand_indices = np.random.permutation(len(all_class_list))
    ind1 = rand_indices[:ratio[0]]
    ind2 = rand_indices[ratio[0]:ratio[0]+ratio[1]]
    ind3 = rand_indices[-ratio[2]:]
    all_class = np.array(all_class_list)
    return all_class[ind1], all_class[ind2], all_class[ind3]


def get_class_splits_from_file(cls_split_file):
    lines = open(cls_split_file, 'r').readlines()
    pretrain = lines[0].strip().split()
    metatrain = lines[1].strip().split()
    metatest = lines[2].strip().split()
    return pretrain, metatrain, metatest


def pretrain_set(classes, anomaly_train, anomaly_test):
    return dict_subset(anomaly_train, classes), dict_subset(anomaly_test, classes)


def split_an_train(all_an_train):
    pretrain = []
    metatrain = []
    for i, an_list in enumerate(all_an_train):
        mid = len(an_list) // 2
        if i == 0:
            pretrain = an_list[:mid]
            metatrain = an_list[mid:]
        else:
            pretrain = np.concatenate([pretrain, an_list[:mid]])
            metatrain = np.concatenate([metatrain, an_list[mid:]])

    return pretrain, metatrain


def split_dict(dic, num_valid_sample, cls, start_idx=0):
    trainset, validset = {}, {}
    num_train_sample = len(dic) - num_valid_sample
    keys = list(dic.keys())
    random.shuffle(keys)

    for i, video_name in enumerate(keys):
        feats = dic[video_name]
        if (i >= start_idx and i < start_idx + num_train_sample) or \
                (start_idx+num_train_sample>=len(dic) and \
                 i<num_train_sample+start_idx-len(dic)):
            trainset[video_name] = feats
        else:
            validset[video_name] = feats
    return {cls: trainset}, {cls: validset}


def get_cv_split(an_dict, num_fold_cv, num_valid_sample, cls):
    split_list = []
    start_indices = np.linspace(0, len(an_dict), num_fold_cv, False)
    for start_idx in start_indices:
        start_idx = int(start_idx)
        split_list.append(split_dict(an_dict, num_valid_sample, cls, start_idx))
    return split_list
