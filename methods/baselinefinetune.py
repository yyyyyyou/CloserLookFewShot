import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
from methods.meta_template import MetaTemplate
import math

class BaselineFinetune(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, loss_type = "softmax"):
        super(BaselineFinetune, self).__init__( model_func,  n_way, n_support)
        self.loss_type = loss_type

    def set_forward(self,x,is_feature = True):
        return self.set_forward_adaptation(x,is_feature); #Baseline always do adaptation

    def set_forward_adaptation(self,x,is_feature = True):
        assert is_feature == True, 'Baseline only support testing with feature'
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())

        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        elif self.loss_type == 'dist' or self.loss_type == 'st':
            linear_clf = backbone.distLinear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()

        batch_size = 4
        support_size = self.n_way* self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()
        scores = linear_clf(z_query)
        return scores

    def set_forward_loss(self,x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')

class BaselineFinetune_soft(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(BaselineFinetune_soft, self).__init__(model_func, n_way, n_support)
        self.k = 2
        self.gamma = 0.2
        self.fc = Parameter(torch.Tensor(self.feat_dim, self.n_way * self.k))
        self.weight = torch.zeros(self.n_way * self.k, self.n_way * self.k, dtype=torch.bool).cuda()
        for i in range(0, self.n_way):
            for j in range(0, self.k):
                self.weight[i*self.k+j, i*self.k+j+1:(i+1)*self.k] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))

    def set_forward(self, x, is_feature=True):
        return self.set_forward_adaptation(x, is_feature);  # Baseline always do adaptation

    def set_forward_adaptation(self, x, is_feature=True):
        assert is_feature == True, 'Baseline only support testing with feature'
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        y_support = Variable(y_support.cuda())

        set_optimizer = torch.optim.Adam([{'params':self.fc,'lr':0.01}],  eps=0.01, weight_decay=0.0001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()

        batch_size = 4
        support_size = self.n_way * self.n_support

        # Softtriple Finetune
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]

                centers = F.normalize(self.fc, p=2, dim=0)
                simInd = z_batch.matmul(centers)
                simStruc = simInd.reshape(-1, self.n_way, self.k)
                prob = F.softmax(simStruc * self.gamma, dim=2) # prob = 5*10
                simClass = torch.sum(prob * simStruc, dim=2) #dim(simClass) = 5

                loss = loss_function(simClass, y_batch)
                loss.backward()
                set_optimizer.step()

        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = z_query.matmul(centers)
        simStruc = simInd.reshape(-1, self.n_way, self.k)
        prob = F.softmax(simStruc * self.gamma, dim=2) # prob = 5*10
        simClass = torch.sum(prob * simStruc, dim=2) #dim(simClass) = 5
        return simClass

    def set_forward_loss(self, x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')
