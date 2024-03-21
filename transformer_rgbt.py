import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import math
import numpy as np
from typing import Optional, List
from torch import nn, Tensor
from .multihead_attention import MultiheadAttention,MultiheadAttention_rgbt
from ltr.models.layers.normalization import InstanceL2Norm
import pdb

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=1, num_layers=1, activation="relu"):
        super().__init__()
        multihead_attn = MultiheadAttention_rgbt(feature_dim=d_model, n_head=1, key_feature_dim=128)
        
        self.encoder = TransformerEncoder(multihead_attn=multihead_attn, FFN=None, 
                                          d_model=d_model, num_encoder_layers=num_layers)
        
        self.decoder = TransformerDecoder(multihead_attn=multihead_attn, FFN=None, 
                                          d_model=d_model, num_decoder_layers=num_layers)

    def forward(self,r_train_feat,r_test_feat,
                t_train_feat,t_test_feat,fuse_train_feat,fuse_test_feat,
                train_label):
        num_img_train = r_train_feat.shape[0]  #n,b,c,h,w
        num_img_test = r_test_feat.shape[0] #n,b,c,h,w
        
        ## encoder  nhw,b,c
        encoded_rgbmemory, encoded_tmemory = self.encoder(r_train_feat,t_train_feat,fuse_train_feat)

        train_label = F.interpolate(train_label, size=(r_train_feat.shape[-2:]), mode='bilinear', align_corners=True)
        enrgb_feat = None
        ent_feat=None
        dergb_feat=None
        det_feat=None
        ## decoder
        for i in range(num_img_train):
            encur_rgbfeat,encur_tfeat,test_rgbmask,test_tmask = self.decoder(
                                                r_train_feat[i,...].unsqueeze(0), t_train_feat[i,...].unsqueeze(0),
                                                fuse_train_feat[i,...].unsqueeze(0),
                                                rgb_memory=encoded_rgbmemory, t_memory=encoded_tmemory, 
                                                pos=train_label)
            if i == 0:
                enrgb_feat = encur_rgbfeat
                ent_feat = encur_tfeat
                show_rgbmask = test_rgbmask
                show_tmask = test_tmask
            else:
                enrgb_feat = torch.cat((enrgb_feat, encur_rgbfeat), 0)
                ent_feat = torch.cat((ent_feat, encur_tfeat), 0)
                show_rgbmask = torch.cat((show_rgbmask, test_rgbmask), 0)
                show_tmask = torch.cat((show_tmask, test_tmask), 0)
        
        for i in range(num_img_test):
            decur_rgbfeat,decur_tfeat,_,_ = self.decoder(
                                                        r_test_feat[i,...].unsqueeze(0), t_test_feat[i,...].unsqueeze(0),
                                                        fuse_test_feat[i,...].unsqueeze(0),
                                                        rgb_memory=encoded_rgbmemory, t_memory=encoded_tmemory,  
                                                        pos=train_label)
            if i == 0:
                dergb_feat = decur_rgbfeat
                det_feat = decur_tfeat
                
            else:
                dergb_feat = torch.cat((dergb_feat, decur_rgbfeat), 0)
                det_feat = torch.cat((det_feat, decur_tfeat), 0)
                
        return enrgb_feat,ent_feat,dergb_feat,det_feat#,show_rgbmask,show_tmask

class TransformerEncoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model):
        super().__init__()
        self.self_attn = multihead_attn
        # Implementation of Feedforward model
        norm_scale = math.sqrt(1.0 / (d_model * 4 * 4))
        self.norm = torch.nn.InstanceNorm2d(d_model) #InstanceL2Norm(scale=norm_scale)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def instance_norm(self, src, input_shape):
        num_imgs, batch, dim, h, w = input_shape
        # Normlization
        src = src.reshape(num_imgs, h, w, batch, dim).permute(0,3,4,1,2) #3,b,512,22,22
        src = src.reshape(-1, dim, h, w) #3b,512,22,22
        src = self.norm(src)  #3b,512,22,22
        # reshape back
        src = src.reshape(num_imgs, batch, dim, -1).permute(0,3,1,2)  #3,484,b,512
        src = src.reshape(-1, batch, dim)  #nhw,b,dim
        return src

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,rgb_src,t_src,fuse_src,input_shape):
        rgb_query = rgb_key = rgb_value = rgb_src  #nhw,b,dim
        t_query = t_key = t_value = t_src  #nhw,b,dim
        fuse_query = fuse_key = fuse_value = fuse_src
        
        ##### self-attention #######
        self_rgb,self_t,cross_rgb,cross_t = self.self_attn(rgbquery=rgb_query, rgbkey=rgb_key, rgbvalue=rgb_value,
                                                           tquery=t_query, tkey=t_key, tvalue=t_value,
                                                           fusequery=fuse_query, fusekey=fuse_key, fusevalue=fuse_value,
                                                           input_shape=input_shape)

        src_rgb = rgb_src + self.dropout1(cross_rgb)#cross_rgb#
        ### src_rgb = self.instance_norm(src_rgb, input_shape)
        src_t = t_src + self.dropout2(cross_t)#cross_t#

        return src_rgb, src_t

class TransformerEncoder(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model=512, num_encoder_layers=1, activation="relu"):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(multihead_attn, FFN, d_model)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)

    def forward(self, rgb_src, t_src, fuse_src): 
        '''
        src: 模板特征，       n,b,dim,h,w
        '''
        assert rgb_src.dim() == 5, 'Expect 5 dimensional inputs'
        assert t_src.dim() == 5, 'Expect 5 dimensional inputs'
        assert fuse_src.dim() == 5, 'Expect 5 dimensional inputs'
        
        src_shape = rgb_src.shape 
        num_imgs, batch, dim, h, w = rgb_src.shape
        
        ########### get nhw,b,dim shape ###############
        rgb_src = rgb_src.reshape(num_imgs, batch, dim, -1).permute(0,3,1,2)  #n,hw,b,dim
        rgb_src = rgb_src.reshape(-1, batch, dim)  #nhw,b,dim
        t_src = t_src.reshape(num_imgs, batch, dim, -1).permute(0,3,1,2)
        t_src = t_src.reshape(-1, batch, dim)  
        fuse_src = fuse_src.reshape(num_imgs, batch, dim, -1).permute(0,3,1,2)  
        fuse_src = fuse_src.reshape(-1, batch, dim)  
        
        output_rgb = rgb_src   #nhw,b,dim
        output_t = t_src
        output_fuse = fuse_src
        
        for layer in self.layers:
            output_rgb,output_t = layer(output_rgb,output_t,output_fuse,input_shape=src_shape)
        
        return output_rgb, output_t

class TransformerDecoderLayer(nn.Module):
    def __init__(self, multihead_attn_rgbt, FFN, d_model):
        super().__init__()
        self.self_attn1 = multihead_attn_rgbt
        self.cross_attn = MultiheadAttention(feature_dim=d_model, n_head=1, key_feature_dim=128)

        norm_scale = math.sqrt(1.0 / (d_model * 4 * 4))
        self.norm = torch.nn.InstanceNorm2d(d_model) 
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
        dropout = 0.1
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        activation="relu"
        self.linear1 = nn.Linear(d_model, 2048)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(2048, d_model)
        self.activation = _get_activation_fn(activation)

        self.linear11 = nn.Linear(d_model, 2048)
        self.dropout11 = nn.Dropout(dropout)
        self.linear21 = nn.Linear(2048, d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor * pos

    def instance_norm(self, src, input_shape):
        num_imgs, batch, dim, h, w = input_shape
        # Normlization
        src = src.reshape(num_imgs, h, w, batch, dim).permute(0,3,4,1,2)#3,b,512,22,22
        src = src.reshape(-1, dim, h, w)#3b,512,22,22
        src = self.norm(src)  #3b,512,22,22
        # reshape back
        src = src.reshape(num_imgs, batch, dim, -1).permute(0,3,1,2)  #3,484,b,512
        src = src.reshape(-1, batch, dim)  #nhw,b,dim
        return src

    def forward(self,rgb_tgt,t_tgt,fuse_tgt,
                rgb_memory, t_memory, 
                input_shape, 
                pos: Optional[Tensor] = None 
                       ):
        '''
        tgt: nhw,b,dim
        memory:nhw,b,dim
        pos:nhw,b,dim
        '''
        ################## self-attention ###########################
        rgb_query = rgb_key = rgb_value = rgb_tgt #b,dim,h,w
        t_query = t_key = t_value = t_tgt
        fuse_query = fuse_key = fuse_value = fuse_tgt
        
        self_rgb, self_t, cross_rgb, cross_t = self.self_attn1(rgbquery=rgb_query, rgbkey=rgb_key, rgbvalue=rgb_value,
                                                              tquery=t_query, tkey=t_key, tvalue=t_value,
                                                              fusequery=fuse_query, fusekey=fuse_key, fusevalue=fuse_value,
                                                              inmask=None, input_shape=input_shape)

        rgb_tgt = rgb_tgt + self.dropout1(cross_rgb)
        t_tgt = t_tgt + self.dropout2(cross_t)
        
        #############################################################
        rgb_mask=None
        t_mask = None

        ################ mask Transformation. query to template #############
        rgb_mask = self.cross_attn(query=rgb_tgt, key=rgb_memory, value=pos)
        t_mask = self.cross_attn(query=t_tgt, key=t_memory, value=pos)

        rgb_tgt2 = rgb_tgt * rgb_mask
        # rgb_tgt2 = self.instance_norm(rgb_tgt2, input_shape)
        t_tgt2 = t_tgt * t_mask

        # # ############## Feature Transformation. ########################################
        # rgb_tgt3 = self.cross_attn(query=rgb_tgt, key=rgb_memory, value=rgb_memory*pos)
        # t_tgt3 = self.cross_attn(query=t_tgt, key=t_memory, value=t_memory*pos)#input_shape=memory_shape

        # rgb_tgt4 = rgb_tgt + rgb_tgt3
        # t_tgt4 = t_tgt + t_tgt3
        
        # rgb_tgt = rgb_tgt2 + rgb_tgt4
        # t_tgt = t_tgt2 + t_tgt4

        ###############################################################################
        rgb_tgt2 = self.linear2(self.dropout(self.activation(self.linear1(rgb_tgt2))))
        t_tgt2 = self.linear21(self.dropout11(self.activation(self.linear11(t_tgt2))))

        rgb_tgt = rgb_tgt + self.dropout3(rgb_tgt2)
        t_tgt = t_tgt + self.dropout4(t_tgt2) 
        # rgb_tgt = self.norm1(rgb_tgt)
        # t_tgt = self.norm2(t_tgt)
        
        return rgb_tgt,t_tgt,rgb_mask,t_mask

class TransformerDecoder(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model=512, num_decoder_layers=6, activation="relu"):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(multihead_attn, FFN, d_model)
        self.layers = _get_clones(decoder_layer, num_decoder_layers)

    def forward(self, rgb_tgt, t_tgt, fuse_tgt,
                rgb_memory, t_memory,
                pos: Optional[Tensor] = None):
        '''
        tgt:1,b,dim,h,w
        memory:n,b,dim,h,w
        pos:b,n,h,w
        '''
        assert rgb_tgt.dim() == 5, 'Expect 5 dimensional inputs'
        assert t_tgt.dim() == 5, 'Expect 5 dimensional inputs'
        assert fuse_tgt.dim() == 5, 'Expect 5 dimensional inputs'
        
        tgt_shape = rgb_tgt.shape  #1,b,dim,h,w
        num_imgs,batch, dim, h, w = rgb_tgt.shape
        
        if pos is not None:#1
            num_pos1,batch1,h1,w1 = pos.shape  #b,n,22,22
            pos = pos.view(num_pos1,batch1, 1, -1).permute(0,3,1,2)  #3,484,b,1
            pos = pos.reshape(-1, batch1, 1)#1452,b,1
            pos = pos.repeat(1, 1, dim)  #nhw,b,dim
        
        rgb_tgt = rgb_tgt.view(num_imgs, batch, dim, -1).permute(0,3,1,2)  #1,484,b,512
        rgb_tgt = rgb_tgt.reshape(-1, batch, dim)#hw,b,dim
        t_tgt = t_tgt.view(num_imgs, batch, dim, -1).permute(0,3,1,2)  #1,484,b,512
        t_tgt = t_tgt.reshape(-1, batch, dim)#hw,b,dim
        fuse_tgt = fuse_tgt.view(num_imgs, batch, dim, -1).permute(0,3,1,2)  #1,484,b,512
        fuse_tgt = fuse_tgt.reshape(-1, batch, dim)#hw,b,dim
        
        output_rgb = rgb_tgt
        output_t = t_tgt
        output_fuse = fuse_tgt

        for layer in self.layers:
            output_rgb,output_t,rgb_mask,t_mask = layer(output_rgb,output_t,output_fuse,
                                                        rgb_memory,t_memory,
                                                        input_shape=tgt_shape, 
                                                        pos=pos)
        
        output_rgbfeat = output_rgb.reshape(num_imgs, h, w, batch, dim).permute(0,3,4,1,2)
        output_rgbfeat = output_rgbfeat.reshape(-1, dim, h, w)

        output_tfeat = output_t.reshape(num_imgs, h, w, batch, dim).permute(0,3,4,1,2)
        output_tfeat = output_tfeat.reshape(-1, dim, h, w)

        output_rgbmask = rgb_mask.reshape(num_imgs, h, w, batch, dim).permute(0,3,4,1,2)
        output_rgbmask = output_rgbmask.reshape(-1, dim, h, w)

        output_tmask = t_mask.reshape(num_imgs, h, w, batch, dim).permute(0,3,4,1,2)
        output_tmask = output_tmask.reshape(-1, dim, h, w)
        
        return output_rgbfeat, output_tfeat, output_rgbmask, output_tmask


def _get_clones(module, N):
    # return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    return nn.ModuleList([module for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


