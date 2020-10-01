from torch.utils.data import DataLoader
from meta import *
from utils import *
from train import *
from options import args
from learner import Learner
from loss import custom_objective
from dataset import *
import os
import wandb

if args.wandb:
    if args.name is None:
        wandb.init(project='WAD_32seg', name='seed{}_mode{}_exp{}'.format(args.seed, args.mode, args.exp))
    else:
        wandb.init(project='WAD_32seg', name=args.name)
    wandb.config.update(args)


if __name__ == '__main__':
    set_random_seed(args.seed)
    '''
    abnorm_train_dict: train split for all classes, abnorm_test_dict: test split for all classes
    For pretrain/meta-train, we extract D_base classes only
    For meta-test (fine-tuning + evaluation), we extract a D_novel class only

    Normal_test_list
    UCF_Crime: for training of meta-test (Normal is not used for evaluation)
    ShanghaiTech: not used
    '''
    abnorm_train_dict, abnorm_test_dict,\
    normal_train_list, normal_test_list,\
    train_classes, test_classes,\
    gt_dict, frame_dict = load_all_data(args)

    if args.mode == 'pretrain':
        if args.dataset == 'ShanghaiTech':
            raise ValueError('ShanghaiTech is only for evaluation')
        print('Pretrain mode!')
        abnorm_train_list = dict2list(dict_subset(abnorm_train_dict, train_classes))

        pretrain_dataset = PretrainDataset(abnorm_train_list, normal_train_list)
        abnorm_validation_dataset = EvalDataset(dict_subset(abnorm_test_dict, train_classes), gt_dict, frame_dict)

        trainloader = DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True)
        validloader = DataLoader(abnorm_validation_dataset, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=abnorm_validation_dataset.collate_fn)
        device = torch.device('cuda')

        model = Learner(input_dim=args.feat_dim, drop_p=args.drop_p).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.inner_lr, weight_decay=args.w_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience,
                                      threshold=args.threshold, threshold_mode='abs')
        criterion = custom_objective
        pretrain(model=model, trainloader=trainloader, validloader=validloader, optimizer=optimizer, criterion=criterion
                 , args=args, device=device, save_chpt=args.save_chpt, valid_step=args.valid_step, scheduler=scheduler)

    elif args.mode == 'eval':
        print('Meta-test (evaluation) mode!')

        abnorm_train_list = dict2list(dict_subset(abnorm_train_dict, test_classes))
        mtftdataset = PretrainDataset(abnorm_train_list, normal_test_list)
        ft_loader = DataLoader(mtftdataset, batch_size=mtftdataset.length, shuffle=True)
        cv_list = get_cv_split(abnorm_train_dict[test_classes[0]], args.num_fold_cv, args.num_valid_sample,
                                test_classes[0])
        cv_loader_list = []
        for train_dict, valid_dict in cv_list:
            if len(train_dict.keys()) == 0 or len(valid_dict.keys()) == 0:
                continue
            train_list = dict2list(train_dict)
            train_dataset = PretrainDataset(train_list, normal_train_list)
            train_loader = DataLoader(train_dataset, batch_size=train_dataset.length, shuffle=True)
            valid_dataset = EvalDataset(valid_dict, gt_dict, frame_dict)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                      collate_fn=valid_dataset.collate_fn)
            cv_loader_list.append((train_loader, valid_loader))

        eval_dataset = EvalDataset(dict_subset(abnorm_test_dict, test_classes), gt_dict, frame_dict)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=eval_dataset.collate_fn)

        device = torch.device('cuda')
        model = Learner(input_dim=args.feat_dim, drop_p=args.drop_p).to(device)
        criterion = custom_objective

        metatest(model, ft_loader, cv_loader_list, eval_loader, criterion, args, sampling=args.sampling)

    elif args.mode == 'meta_train':
        if args.dataset == 'ShanghaiTech':
            raise ValueError('ShanghaiTech is only for evaluation')
        print('Meta-train mode!')
        abnorm_train_list = dict2list(dict_subset(abnorm_train_dict, train_classes))

        chpt = None
        if args.chpt is not None:
            chpt = os.path.join('pretrain', args.chpt)
        meta_trainer = Meta(args, abnorm_train_list, normal_train_list, metric=calc_auc_per_video, chpt=chpt)

        metatrain(meta_trainer, args)

