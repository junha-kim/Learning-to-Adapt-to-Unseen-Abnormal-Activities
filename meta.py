import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from learner import Learner
from loss import custom_objective
from dataset import MetaDataset


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, abnorm_feats, norm_feats, metric, chpt=None):
        """

        :param args:
        """
        super(Meta, self).__init__()

        # meta train config
        self.inner_lr = args.inner_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.num_task = args.num_task
        self.batch_size = args.batch_size
        self.kshot = args.kshot
        self.epochs = args.inner_epoch
        self.w_decay = args.w_decay
        self.grad_clip = args.grad_clip

        self.inner_optim = 'sgd'
        # when inner_optim == 'adam'
        self.eps = 1e-8
        self.beta1 = 0.9
        self.beta2 = 0.999

        self.device = torch.device('cuda')
        self.net = Learner(args.feat_dim, drop_p=0).to(self.device)
        if chpt is not None:
            print('Load pretrained model!')
            self.net.load_state_dict(torch.load(chpt))

        self.loss_fn = custom_objective
        self.abnorm_feats = abnorm_feats
        self.norm_feats = norm_feats
        self.metric = metric

        self.meta_sgd = args.meta_sgd
        if self.meta_sgd:
            self.lr_params = self.get_lr_params()
            self.learner_optim = optim.Adam([{'params': self.net.parameters(), 'lr': args.meta_lr, 'weight_decay': args.w_decay},
                                             {'params':self.lr_params, 'lr': args.lr_lr}])
        else:
            self.learner_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr, weight_decay=args.w_decay)

    def get_init_moments(self):
        # implementation of adam when self.optim == 'adam'
        m = []
        v = []
        for param in self.net.parameters():
            m.append(torch.zeros_like(param, requires_grad=False))
            v.append(torch.zeros_like(param, requires_grad=False))
        return m, v

    def adam(self, args):#grad, params, m_prev, v_prev, t):
        grad, params, m_prev, v_prev, inner_lr, t = args[0], args[1], args[2], args[3], args[4], args[5]
        m = self.beta1 * m_prev + (1-self.beta1) * grad
        v = self.beta2 * v_prev + (1-self.beta2) * (grad ** 2)
        m_ = m/(1-self.beta1**t)
        v_ = v/(1-self.beta2**t)
        return params - inner_lr * m_ / (self.eps + torch.sqrt(v_)), m, v

    def sgd(self, args):
        grad, params, lr = args[0], args[1], args[2]
        return params - lr * grad

    def get_lr_params(self):
        # implementation of 'Meta-sgd'(== learnable lr) when self.meta_sgd is True
        lr_list = []
        for param in self.net.parameters():
            lr_list.append(nn.Parameter(self.inner_lr * torch.ones_like(param, requires_grad=True)))
        return lr_list

    def inner_loop(self, data):

        if self.inner_optim == 'adam':
            m, v = self.get_init_moments()
            t = 0
        x_spt = data[0].view(-1, data[0].size(-1)).to(self.device)
        y_spt = data[1].view(-1).to(self.device)
        x_qry = data[2].view(-1, data[2].size(-1)).to(self.device)
        y_qry = data[3].view(-1).to(self.device)
        fast_weights = self.net.parameters()
        for epoch in range(self.epochs):
            if self.inner_optim == 'adam':
                t += 1
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = self.net(x_spt, fast_weights)
            loss = self.loss_fn(logits.view(-1, 1), y_spt)
            for param in fast_weights:
                loss += torch.sum(param ** 2) * self.w_decay
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)#, create_graph=True)
            # 3. theta_pi = theta_pi - train_lr * grad
            if self.inner_optim == 'adam':
                if self.meta_sgd:
                    fast_weights, m, v = list(zip(*map(self.adam,
                                                    zip(grad, fast_weights, m, v, self.lr_params, [t]*len(grad))
                                                    )))
                else:
                    fast_weights, m, v = list(zip(*map(self.adam,
                                                    zip(grad, fast_weights, m, v, [self.inner_lr]*len(grad), [t]*len(grad))
                                                    )))
            else:
                if self.meta_sgd:
                    fast_weights = list(map(self.sgd, zip(grad, fast_weights, self.lr_params)))
                else:
                    fast_weights = list(map(self.sgd, zip(grad, fast_weights, [self.inner_lr]*len(grad))))

        logits_q = self.net(x_qry, fast_weights)
        loss_q = self.loss_fn(logits_q.view(-1, 1), y_qry)

        for param in fast_weights:
            loss_q += torch.sum(param ** 2) * self.w_decay

        return loss_q

    def outer_loop(self):

        total_loss = 0
        dataset = MetaDataset(self.abnorm_feats, self.norm_feats, self.num_task)
        loader = DataLoader(dataset, batch_size=1, num_workers=4)

        for data in loader:
            loss = self.inner_loop(data)
            total_loss += loss

        total_loss /= len(loader)

        # optimize theta parameters
        self.learner_optim.zero_grad()
        total_loss.backward()
        self.learner_optim.step()
        return total_loss.item()
