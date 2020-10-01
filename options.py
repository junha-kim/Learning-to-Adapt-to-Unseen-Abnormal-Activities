import argparse
import os

argparser = argparse.ArgumentParser()
# main
argparser.add_argument('--mode', type=str, help='pretrain / meta_train', default='pretrain')
argparser.add_argument('--exp', type=int, help='To distinguish between different experiments', default=0)
argparser.add_argument('--name', type=str, default=None)
argparser.add_argument('--seed', type=int, help='To distinguish different target class', default=1)
argparser.add_argument('--test_cls', type=str, help='One test class name', default='Explosion')
argparser.add_argument('--all_class', action='store_true', default=False)

# Baseline model
argparser.add_argument('--feat_dim', type=int, help='video feature dimension, 1024 for rgb/flow, 2048 for 2-stream',
                       default=2048)
argparser.add_argument('--drop_p', type=float, help='dropout rate', default=0.0)
argparser.add_argument('--batch_size', type=int, help='mini-batch size of one task', default=30)

# Meta train/test
argparser.add_argument('--kshot', type=int, help='k-shot',default=40)
argparser.add_argument('--outer_epoch', type=int, help='outer loop epoch number', default=3100)  # 100 or 200
argparser.add_argument('--inner_epoch', type=int, help='inner loop epoch number', default=5)
argparser.add_argument('--ft_max_epoch', type=int, default=300)
argparser.add_argument('--n_way', type=int, help='n way', default=1)
argparser.add_argument('--valid_step', type=int, default=10)

argparser.add_argument('--num_task', type=int, help='meta batch size, namely task num', default=15)
argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-5)
argparser.add_argument('--lr_lr', type=float, help='lr params learning rate', default=1e-5)
argparser.add_argument('--inner_lr', type=float, help='task-level inner update learning rate', default=1e-3)
argparser.add_argument('--ft_lr', type=float, help='task-level inner update learning rate', default=1e-3)
argparser.add_argument('--factor', type=float, default=0.9)
argparser.add_argument('--threshold', type=float, default=1e-4)
argparser.add_argument('--patience', type=int, default=30)
argparser.add_argument('--w_decay', type=float, default=0.01)
argparser.add_argument('--chpt', type=str, default=None)
# argparser.add_argument('--use_label', action='store_true', default=False)
argparser.add_argument('--save_chpt', action='store_true', default=False)
argparser.add_argument('--meta_sgd', action='store_true', default=False)
argparser.add_argument('--num_valid_sample', help='number of samples used for one validation',type=int, default=8)
argparser.add_argument('--num_fold_cv', help='number of different cross validation split',type=int, default=10)
argparser.add_argument('--chpt_format', type=str, default=None)
argparser.add_argument('--meta_dir', type=str, default='meta_model')
argparser.add_argument('--swa_start',type=int, default=0)
argparser.add_argument('--swa_end', type=int, default=300)
argparser.add_argument('--swa_step', type=int, default=1)
argparser.add_argument('--grad_clip', type=float, default=10.0)

argparser.add_argument('--sampling', action='store_true', default=False)

argparser.add_argument('--wandb', help='Whether to use wandb to log loss curve', action='store_true', default=False)

# Data loading config for UCF_Crime and ShanghaiTech
argparser.add_argument('--dataset', type=str, help='Name of dataset (UCF-Crime or ShanghaiTech)', default='UCF-Crime')
argparser.add_argument('--feat_type', type=str, help='Type of video feature (1. rgb, 2.flow, 3.2-stream)',
                       default='2-stream')

# config for UCF_Crime
argparser.add_argument('--data_root_dir', type=str, help='home dir of all dataset', default='data')
argparser.add_argument('--rgb_dir', type=str, help='directory containing all numpy rgb feature',
                       default='all_rgbs')
argparser.add_argument('--flow_dir', type=str, help='directory containing all numpy flow feature',
                       default='all_flows')
argparser.add_argument('--split_dir', type=str, default='splits')
argparser.add_argument('--normal_dir', type=str, help='Normal dir name in rgb_dir or flow_dir',
                       default='Normal_Videos_event')
argparser.add_argument('--frame_dict_path', type=str, help='frames dictionary file containing frames number of each video',
                       default='frames.pkl')
argparser.add_argument('--gt_dict_path', type=str, help='gt dictionary file containing gt info',
                       default='GT_anomaly.pkl')
argparser.add_argument('--exclusion_list_path', type=str, default='exclusion.pkl')
argparser.add_argument('--split_num', type=int, help='train/test split file index', default=4)

args = argparser.parse_args()

args.rgb_dir = os.path.join(args.data_root_dir, args.dataset, args.rgb_dir)
args.flow_dir = os.path.join(args.data_root_dir, args.dataset, args.flow_dir)
args.split_dir = os.path.join(args.data_root_dir, args.dataset, args.split_dir)
args.frame_dict_path = os.path.join(args.data_root_dir, args.dataset, args.frame_dict_path)
args.gt_dict_path = os.path.join(args.data_root_dir, args.dataset, args.gt_dict_path)
args.exclusion_list_path = os.path.join(args.data_root_dir, args.dataset, args.exclusion_list_path)

if args.feat_type == '2-stream':
    args.feat_dim = 2048
else:
    args.feat_dim = 1024
