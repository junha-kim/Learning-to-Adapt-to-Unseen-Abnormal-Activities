import torch
from torch.utils.data import Dataset
import numpy as np


class PretrainDataset(Dataset):
    def __init__(self, abnormal_feats, normal_feats):
        super(PretrainDataset, self).__init__()
        self.abnormal = np.array(abnormal_feats)
        if len(self.abnormal.shape) >= 4 or len(self.abnormal.shape) == 1:
            # num_class x num_video x num_seg x feat_dim --> num_class * num_video x num_seg x feat_dim
            self.abnormal = np.concatenate(self.abnormal)
        self.normal = np.array(normal_feats)
        self.length = min(len(self.normal), len(self.abnormal))

    def shuffle(self):
        ab_indices = np.random.permutation(len(self.abnormal))
        no_indices = np.random.permutation(len(self.normal))
        self.abnormal = self.abnormal[ab_indices]
        self.normal = self.normal[no_indices]

    def __getitem__(self, item):
        return np.concatenate([self.abnormal[item], self.normal[item]]), np.array([1, 0])

    def __len__(self):
        return self.length


class EvalDataset(Dataset):
    def __init__(self, feat_dict, gt_dict, frame_dict):
        self.feats = []
        self.gts = []
        self.frames = []
        self.names = []
        for cls, dic in feat_dict.items():
            if type(dic) == dict:
                # for abnormal video
                for video_name, feats in dic.items():
                    gt = gt_dict[video_name]['gt']
                    if len(gt) == 0:
                        print(video_name)
                        continue
                    self.feats.append(feats)
                    for i in range(len(gt)):
                        gt[i] = int(gt[i])
                    self.gts.append(gt)
                    self.frames.append(int(frame_dict[video_name]))
                    self.names.append(video_name)
            else:
                # for normal video
                self.feats.append(dic) # 'dic' is normal features
                self.frames.append(int(frame_dict[cls])) # 'cls' is frame number
                self.gts.append([-1, -1])
                self.names.append(cls)

    def __getitem__(self, item):
        return self.feats[item], self.gts[item], self.frames[item], self.names[item]

    def __len__(self):
        return len(self.frames)

    def collate_fn(self, data):
        feats, gts, frames, names = list(zip(*data))
        feats, gts, frames, names = torch.Tensor(np.asarray(feats)), list(gts), list(frames), list(names)
        return feats, gts, frames, names


class MetaDataset(Dataset):
    def __init__(self, abnorm_feats, norm_feats, num_task):
        self.abnorm_feats = np.array(abnorm_feats)
        if len(self.abnorm_feats.shape) >= 4 or len(self.abnorm_feats.shape) == 1:
            self.abnorm_feats = np.concatenate(self.abnorm_feats)

        self.norm_feats = np.array(norm_feats)
        self.length = min(len(self.abnorm_feats), len(self.norm_feats))
        self.n_spt = 10
        self.n_qry = 30
        self.num_task = num_task
        self.build_episode()

    def build_episode(self):
        self.ab_spt = []
        self.no_spt = []
        self.ab_qry = []
        self.no_qry = []

        for i in range(self.num_task):
            ab_indices = np.random.permutation(len(self.abnorm_feats))
            no_indices = np.random.permutation(len(self.norm_feats))

            self.ab_spt.append(self.abnorm_feats[ab_indices[:self.n_spt]])
            self.no_spt.append(self.norm_feats[no_indices[:self.n_spt]])
            self.ab_qry.append(self.abnorm_feats[ab_indices[self.n_spt:self.n_spt+self.n_qry]])
            self.no_qry.append(self.norm_feats[no_indices[self.n_spt:self.n_spt+self.n_qry]])

    def __getitem__(self, item):
        return np.concatenate([self.ab_spt[item], self.no_spt[item]]), np.array([1] * self.n_spt + [0] * self.n_spt),\
               np.concatenate([self.ab_qry[item], self.no_qry[item]]), np.array([1] * self.n_qry + [0] * self.n_qry)

    def __len__(self):
        return self.num_task
