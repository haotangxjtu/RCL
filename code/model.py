"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np

from torch.nn import Module
import torch.nn.functional as F

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)
 
 

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    # def getSparseEye(self,num):
    #     i = torch.LongTensor([[k for k in range(0,num)],[j for j in range(0,num)]])
    #     val = torch.FloatTensor([1]*num)
    #     return torch.sparse.FloatTensor(i,val)
    


    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        self.T=self.config['temperature']
        self.gama=self.config['gama']
        self.lamda=self.config['lamda']

       # self.selfLoop = self.getSparseEye(self.num_users+self.num_items)


        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
      
  
        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_5 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
             
        #初始化
        inip=1.0
        self.fuse_weight_1.data.fill_(inip/(self.n_layers+1))
        self.fuse_weight_2.data.fill_(inip/(self.n_layers+1))
        self.fuse_weight_3.data.fill_(inip/(self.n_layers+1))
        self.fuse_weight_4.data.fill_(inip/(self.n_layers+1)) 
        self.fuse_weight_5.data.fill_(inip/(self.n_layers+1))

 
         # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]#/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
 

    def drop_feature(self,x, drop_prob):
        drop_mask = torch.empty(
            (x.size(1), ),
            dtype=torch.float32,
            device=x.device).uniform_(0, 1) < drop_prob
        x = x.clone()
        x[:, drop_mask] = 0 
        return x


    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
     

    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
 
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        if  True :
            if self.config['dropout']:
                if self.training:
                 #   print("droping")
                    g_droped = self.__dropout(self.keep_prob)
                else:
                    g_droped = self.Graph        
            else:
                g_droped = self.Graph     

            for layer in range(self.n_layers):
                if self.A_split:
                    temp_emb = []
                    for f in range(len(g_droped)):
                        temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                    side_emb = torch.cat(temp_emb, dim=0)
                    all_emb = side_emb
                else:
                    all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)

            if self.config['methods']=="lrgccf":
                light_out=torch.cat(embs, 1)
                

            if self.config['methods']=="LightGCN":

                embs = torch.stack(embs, dim=1) 
          
                # light_out = torch.mean(embs, dim=1)
                if self.n_layers==1:
                    light_out= self.fuse_weight_1*embs[:,0,:] + self.fuse_weight_2*embs[:,1,:] 
                if self.n_layers==2:
                    light_out= self.fuse_weight_1*embs[:,0,:] + self.fuse_weight_2*embs[:,1,:]+ self.fuse_weight_3*embs[:,2,:] 
                if self.n_layers==3:
                    light_out= self.fuse_weight_1*embs[:,0,:] + self.fuse_weight_2*embs[:,1,:]+ self.fuse_weight_3*embs[:,2,:]+ self.fuse_weight_4*embs[:,3,:] 
                if self.n_layers==4:
                    light_out= self.fuse_weight_1*embs[:,0,:] + self.fuse_weight_2*embs[:,1,:]+ self.fuse_weight_3*embs[:,2,:]+ self.fuse_weight_4*embs[:,3,:]+ self.fuse_weight_5*embs[:,4,:]
  
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        return users, items 
    
    def topkused(self,light_out,top_k=48): 
        filter_value=0.0
        indices_to_remove = light_out < torch.topk(light_out, top_k)[0][..., -1, None]
        light_out[indices_to_remove] = filter_value 

        return light_out

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
   
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
         
        return loss, 0
        
      
  
    def topk_loss_gcl(self,users,pos_items, neg_items) :
        users_emb,pos_emb =  self.computer()  
 
        users_emb=   F.normalize(users_emb[users])  
        pos_emb=   F.normalize(pos_emb[pos_items])  
        T=self.T 
        sim_batch0=torch.mm(users_emb,pos_emb.t() ) 
        sim_batch= torch.exp(sim_batch0/T )  
        sim_batch=sim_batch*F.softplus(sim_batch0)**self.gama
        posself=sim_batch.diag()
        simsorted, indices = torch.sort(sim_batch)
      
        neg=  sim_batch.sum(dim=1)-posself
        topk=2
        lossRS= -torch.log(( posself+0.00001) /(neg+0.00001))-self.lamda*torch.log((simsorted[:,-topk:].sum(dim=1) +0.00001) /(neg+0.00001))
        loss=  lossRS.mean()

        return loss 

  

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma

 