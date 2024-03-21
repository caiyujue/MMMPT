import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import pdb
from collections import OrderedDict

class MultiheadAttention(nn.Module):
    def __init__(self, feature_dim=512, n_head=8, key_feature_dim=64):
        super(MultiheadAttention, self).__init__()
        self.Nh = n_head
        self.head = nn.ModuleList()
        for N in range(self.Nh):
            self.head.append(RelationUnit(feature_dim, key_feature_dim))
        # self.out_conv = nn.Linear(n_head*key_feature_dim, feature_dim)  # bias=False

    def forward(self, query=None, key=None, value=None):
        isFirst = True
        for N in range(self.Nh):
            if(isFirst):
                concat = self.head[N](query, key, value)
                isFirst = False
            else:
                concat = torch.cat((concat, self.head[N](query, key, value)), -1)
        # output = self.out_conv(concat)
        output = concat
        return output
    

class RelationUnit(nn.Module):
    def __init__(self, feature_dim=512, key_feature_dim=64):
        super(RelationUnit, self).__init__()
        self.temp = 100.  
        self.WK = nn.Linear(feature_dim, key_feature_dim)  # bias=False
        self.WV = nn.Linear(feature_dim, feature_dim)

        # Init weights
        for m in self.WK.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        
        for m in self.WV.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        

    def forward(self, query=None, key=None, value=None):
        ### NHW,B,C
        w_k = self.WK(key)
        w_k = F.normalize(w_k, p=2, dim=-1)
        w_k = w_k.permute(1,2,0) # Batch, Dim, Len_1

        w_q = self.WK(query)
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(1,0,2) # Batch, Len_2, Dim
        
        dot_prod = torch.bmm(w_q, w_k) # Batch, Len_2, Len_1  
        affinity = F.softmax(dot_prod*self.temp, dim=-1) 
        
        w_v = value.permute(1,0,2) # Batch, Len_1, Dim
        output = torch.bmm(affinity, w_v) # Batch, Len_2, Dim
        output = output.permute(1,0,2)

        return output

class MultiheadAttention_rgbt(nn.Module):
    def __init__(self, feature_dim=512, n_head=8, key_feature_dim=64):#512,1,128
        super(MultiheadAttention_rgbt, self).__init__()
        self.Nh = n_head#1
        self.head = nn.ModuleList()
        for N in range(self.Nh):
            self.head.append(RelationUnit_rgbt(feature_dim, key_feature_dim))

    def forward(self, 
                rgbquery=None, rgbkey=None, rgbvalue=None,
                tquery=None, tkey=None, tvalue=None,
                fusequery=None, fusekey=None, fusevalue=None,
                inmask=None,
                input_shape=None):  # mask=None, mask1= None
        isFirst = True
        for N in range(self.Nh):
            if(isFirst):  #1#,affinity
                rgb_concat,t_concat,rgb_concat1,t_concat1 = self.head[N](rgbquery, rgbkey, rgbvalue,
                                                   tquery, tkey, tvalue,
                                                   fusequery, fusekey, fusevalue,
                                                   inmask, input_shape)
                isFirst = False
            else:
                rgb_concat,t_concat,rgb_concat1,t_concat1 = torch.cat((rgb_concat, 
                                                self.head[N](rgbquery, rgbkey, rgbvalue,
                                                tquery, tkey, tvalue, 
                                                fusequery, fusekey, fusevalue,
                                                inmask, input_shape)), -1)
        return rgb_concat,t_concat,rgb_concat1,t_concat1

class RelationUnit_rgbt(nn.Module):
    def __init__(self, feature_dim=512, key_feature_dim=64 ):#512,128
        super(RelationUnit_rgbt, self).__init__()
        self.temp = 100.  
        self.feature_dim = feature_dim
        self.key_feature_dim = key_feature_dim

        self.WK1 = nn.Linear(feature_dim, key_feature_dim)  # bias=False
        self.WV1 = nn.Linear(feature_dim, feature_dim)
        self.GAP = nn.AdaptiveAvgPool1d(1)
        feat_dim=512
        hidden_dim = 256
        self.FF_layers = nn.Sequential(OrderedDict([                 
                ('fc1',   nn.Sequential(
                                        nn.Linear(feat_dim, hidden_dim),
                                        nn.ReLU())),
                ('fc2_RGB',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(hidden_dim, feat_dim),
                                        nn.ReLU())),
                ('fc2_T',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(hidden_dim, feat_dim),
                                        nn.ReLU()))
                                        ]))
     
        # Init weights
        for m in self.WK1.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.WV1.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

    def get_affinity(self,mask,w_q, w_k):
        mask_pos = mask
        mask_neg = 1 - mask_pos
        sim = torch.bmm(w_q, w_k) # b, Len_q, Len_k    input（p,m,n) * mat2(p,n,a) ->output(p,m,a)
        sim_pos = sim * mask_pos.view(mask_pos.shape[0], 1, -1)
        sim_neg = sim * mask_neg.view(mask_neg.shape[0], 1, -1)
        pos_map = torch.mean(torch.topk(sim_pos, 5, dim=-1).values, dim=-1)#b,lq
        neg_map = torch.mean(torch.topk(sim_neg, 5, dim=-1).values, dim=-1)#b,lq
        pred_ = torch.cat((torch.unsqueeze(pos_map, -1), torch.unsqueeze(neg_map, -1)), dim=-1)  #B,LQ,2
        pred_sm = F.softmax(pred_, dim=-1) #按行求，行和为1
        mask_sm = torch.unsqueeze(pred_sm[:, :, 0], dim=-1)#b,lq,1
        affinity = mask_sm.repeat(1, 1, w_k.shape[-1])  #b,lq,lk
        return affinity,pred_

    def get_fused(self,self_mask,cross_mask):
        b,l,dim = self_mask.shape
        self_mask = self_mask.permute(0,2,1).contiguous()
        cross_mask = cross_mask.permute(0,2,1).contiguous()
        feat_sum = self.GAP(self_mask+cross_mask)

        feat_sum = feat_sum.view(-1, feat_sum.shape[1])#b,512
        feat_sum = self.FF_layers.fc1(feat_sum)#b,256
        ########################################
        w_RGB = self.FF_layers.fc2_RGB(feat_sum)#b,512
        w_T = self.FF_layers.fc2_T(feat_sum)#b,512
        w = F.softmax(torch.cat([w_RGB, w_T], 0),dim=0)

        ##########################################
        feat = self_mask * w[:b,:].view(b,w.size()[1],1) + cross_mask * w[b:,:].view(b,w.size()[1],1)
        return feat

    def forward(self, rgbquery=None, rgbkey=None, rgbvalue=None, 
                tquery=None, tkey=None, tvalue=None, 
                fusequery=None, fusekey=None, fusevalue=None, 
                inmask=None, input_shape=None):
        
        ##########################################################
        rgb_q = self.WK1(rgbquery) ##([lenq, b, k_dim])
        rgb_q = F.normalize(rgb_q, p=2, dim=-1)#对输入的数据（tensor）进行指定维度的L2_norm运算
        rgb_q = rgb_q.permute(1,0,2).contiguous() # b, lenq, k_dim

        t_q = self.WK1(tquery) ##([lenq, b, k_dim])
        t_q = F.normalize(t_q, p=2, dim=-1)#对输入的数据（tensor）进行指定维度的L2_norm运算
        t_q = t_q.permute(1,0,2).contiguous() # b, lenq, k_dim
        ########################################################
        rgb_k = self.WK1(rgbkey)  #([lenk, b, k_dim])
        rgb_k = F.normalize(rgb_k, p=2, dim=-1)
        rgb_k = rgb_k.permute(1,2,0).contiguous() # b, k_dim, lenk

        t_k = self.WK1(tkey)  #([lenk, b, k_dim])
        t_k = F.normalize(t_k, p=2, dim=-1)
        t_k = t_k.permute(1,2,0).contiguous() # b, k_dim, lenk

        #######################################################
        fuse_k = self.WK1(torch.cat((rgbkey,fusekey),dim=0))
        fuse_k = F.normalize(fuse_k, p=2, dim=-1)
        fuse_k = fuse_k.permute(1,2,0).contiguous()

        fuse_k1 = self.WK1(torch.cat((tkey,fusekey),dim=0)) 
        fuse_k1 = F.normalize(fuse_k1, p=2, dim=-1)
        fuse_k1 = fuse_k1.permute(1,2,0).contiguous()
        #######################################################
        
        #rgb intra_modality_affinity #b,lenq,lenk
        rgbdot_prod = torch.bmm(rgb_q, fuse_k)
        rgbaffinity = F.softmax(rgbdot_prod*self.temp, dim=-1)
        #t intra_modality_affinity #b,lenq,lenk
        tdot_prod = torch.bmm(t_q, fuse_k1)
        taffinity = F.softmax(tdot_prod*self.temp, dim=-1)

        #intre_modality_correlation
        rgbtdot_prod = rgbdot_prod * tdot_prod  #b,lenq,lenk
        intre_affinity = F.softmax(rgbtdot_prod, dim=-1)
        
        #rgb_v = rgbvalue.permute(1,0,2).contiguous()
        rgb_v = torch.cat((rgbvalue,fusevalue),dim=0).permute(1,0,2).contiguous() # b, Len_v, Dim  ,  Len_v = len_k
        
        #t_v = tvalue.permute(1,0,2).contiguous()
        t_v = torch.cat((tvalue,fusevalue),dim=0).permute(1,0,2).contiguous()

        ### 自注意力图特征
        self_rgb = torch.bmm(rgbaffinity, rgb_v) # Batch, Len_q, Dim
        self_t = torch.bmm(taffinity, t_v)
        cross_rgb = torch.bmm(intre_affinity, t_v) # Batch, Len_q, Dim
        cross_t = torch.bmm(intre_affinity, rgb_v) # Batch, Len_q, Dim

        self_rgb = self_rgb.permute(1,0,2)  #Lenq,Batch,Dim
        self_t = self_t.permute(1,0,2)
        cross_rgb = cross_rgb.permute(1,0,2)  #Lenq,Batch,Dim
        cross_t = cross_t.permute(1,0,2)  #Lenq,Batch,Dim

        return  self_rgb, self_t,cross_rgb, cross_t,

class MultiheadAttention_fuse(nn.Module):
    def __init__(self, feature_dim=512, n_head=1, key_feature_dim=128):#512,1,128
        super(MultiheadAttention_fuse, self).__init__()
        self.Nh = n_head #1
        self.head = nn.ModuleList()
        for N in range(self.Nh):
            self.head.append(RelationUnit_fuse(feature_dim, key_feature_dim))

    def forward(self, query=None, key=None, value=None): 
        isFirst = True
        for N in range(self.Nh):
            if(isFirst): 
                concat = self.head[N](query, key, value)
                isFirst = False
            else:
                concat = torch.cat((concat, self.head[N](query, key, value)), -1)
        return concat

class RelationUnit_fuse(nn.Module):
    def __init__(self, feature_dim=512, key_feature_dim=64 ):#512,128
        super(RelationUnit_rgbt, self).__init__()
        self.temp = 100.  
        self.feature_dim = feature_dim
        self.key_feature_dim = key_feature_dim

        self.WK1 = nn.Linear(feature_dim, key_feature_dim)  # bias=False
        self.WV1 = nn.Linear(feature_dim, feature_dim)

        # Init weights
        for m in self.WK1.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.WV1.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, query=None, key=None, value=None):
        
        rgb_k = self.WK1(key)  #([lenk, b, k_dim])
        rgb_k = F.normalize(rgb_k, p=2, dim=-1)
        rgb_k = rgb_k.permute(1,2,0).contiguous() # b, k_dim, lenk

        rgb_q = self.WK1(query) ##([lenq, b, k_dim])
        rgb_q = F.normalize(rgb_q, p=2, dim=-1)#对输入的数据（tensor）进行指定维度的L2_norm运算
        rgb_q = rgb_q.permute(1,0,2).contiguous() # b, lenq, k_dim

        #rgb intra_modality_affinity #b,lenq,lenk
        rgbdot_prod = torch.bmm(rgb_q, rgb_k)
        rgbaffinity = F.softmax(rgbdot_prod*self.temp, dim=-1)

        rgb_v = value.permute(1,0,2).contiguous() # b, Len_v, Dim  ,  Len_v = len_k

        self_rgb = torch.bmm(rgbaffinity, rgb_v) # Batch, Len_q, Dim
        self_rgb = self_rgb.permute(1,0,2)  #Lenq,Batch,Dim

        return self_rgb
