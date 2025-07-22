from model import longclip, longclip_adapter
import torch
import torch.nn as nn
from model.model_longclip import LayerNorm, QuickGELU
import re
from collections import OrderedDict
from typing import Tuple, Union

def split_sentence(text):

    pattern = r'[^.!?]+[.!?]'
    matches = re.findall(pattern, text)
    sentences = [match.strip() for match in matches]

    return sentences

class TokenizerSelector:
    def __new__(cls, name, clippath=None):
        if name == "longclip":
            return longclip.tokenize
        elif name == "longclipadapter":
            return longclip.tokenize
        else:
            raise ValueError("Invalid name parameter")

class ClipSelector:
    def __new__(cls, name, clippath=None, adapter_type = "DC", checkpoint_path=None):
        if name == "longclip":
            return longclip.load(clippath,device='cpu')[0]
        elif name == "longclipadapter":
            model = longclip_adapter.load_adapter_model(clippath,device='cpu',adapter_type=adapter_type)
            return model
        else:
            raise ValueError("Invalid name parameter")

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        #self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) 
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1) 
                nn.init.constant_(m.bias, 0)    


    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def get_attn(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        lnx = self.ln_1(x)
        attnx, attn = self.attn(lnx, lnx, lnx, attn_mask=self.attn_mask)
        x = x + attnx
        x = x + self.mlp(self.ln_2(x))
        return x, attn

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class CrossAttentionTextKVModel(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim,num_head):
        super(CrossAttentionTextKVModel, self).__init__()
        self.text_fc_k = nn.Linear(text_dim, hidden_dim)
        self.text_fc_v = nn.Linear(text_dim, hidden_dim)
        self.image_fc = nn.Linear(image_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_head)
        self.ln_1 = LayerNorm(hidden_dim)
        self.ln_2 = LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(hidden_dim, hidden_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(hidden_dim * 4, hidden_dim))
        ]))
        self.ln_3 = LayerNorm(hidden_dim)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) 
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)  
                nn.init.constant_(m.bias, 0)    

    def forward(self, text_features, image_features, text_mask=None):
        origin_feature = image_features
        text_features = self.ln_1(text_features)
        image_features = self.ln_2(image_features)
        text_features_k = self.text_fc_k(text_features)
        text_features_v = self.text_fc_v(text_features)
        image_features = self.image_fc(image_features)

        if text_mask is not None:
            attn_output, _ = self.attention(
                query=image_features.permute(1, 0, 2),
                key=text_features_k.permute(1, 0, 2),
                value=text_features_v.permute(1, 0, 2),
                key_padding_mask=text_mask
            )
        else:
            attn_output, _ = self.attention(
                query=image_features.permute(1, 0, 2),
                key=text_features_k.permute(1, 0, 2),
                value=text_features_v.permute(1, 0, 2)
            )
        origin_feature = origin_feature + attn_output.permute(1, 0, 2)
        origin_feature = origin_feature + self.mlp(self.ln_3(origin_feature))
        return origin_feature
    
class CrossAttentionTextKVBlockformal(nn.Module):
    def __init__(self,
                 text_dim,
                 image_dim,
                 hidden_dim=512,
                 num_head=8,
                 ):
        super(CrossAttentionTextKVBlockformal, self).__init__()
        
        self.CrossBlock =  CrossAttentionTextKVModel(text_dim=text_dim, image_dim=image_dim, hidden_dim=hidden_dim,num_head=num_head)
        self.SelfAttenBlock = SelfAttentionBlock(image_dim,num_head)

    def get_attn(self,padded_detail_features,fusion_feature,detail_mask=None):
        x = self.CrossBlock(padded_detail_features,fusion_feature,detail_mask)
        x = x.permute(1, 0, 2)
        x, attn = self.SelfAttenBlock.get_attn(x)
        x = x.permute(1, 0, 2)
        return x, attn

    def forward(self,padded_detail_features,fusion_feature,detail_mask=None):
        x = self.CrossBlock(padded_detail_features,fusion_feature,detail_mask)
        x = x.permute(1, 0, 2)
        x = self.SelfAttenBlock(x)
        x = x.permute(1, 0, 2)
        return x

class C2cmodel(nn.Module): 
    def __init__(self,
                clipmodel,
                tokenizer,
                 ):
        super(C2cmodel, self).__init__()
        self.clipmodel = clipmodel
        self.tokenizer = tokenizer
        projdim = self.clipmodel.token_embedding.weight.shape[1]
        self.Eiters = 0
        self.max_sentences = 9
        self.logit_scale = self.clipmodel.logit_scale.exp()
        self.detailproj = nn.ModuleList([
            nn.Identity()
            for i in range(3)])
        self.imgfusSelfblock = SelfAttentionBlock(512,16)
        self.detailfusSelfblock = SelfAttentionBlock(512,16)
        self.crossblockS = nn.ModuleList([
            CrossAttentionTextKVBlockformal(text_dim=projdim, image_dim=projdim, hidden_dim=projdim,num_head=16)
            for i in range(1)])
        self.crossblockW = nn.ModuleList([
            CrossAttentionTextKVBlockformal(text_dim=projdim, image_dim=projdim, hidden_dim=projdim,num_head=16)
            for i in range(1)])
        self.img_norm = LayerNorm(512)
        self.fusion_proj = nn.Linear(768,512)
        self.sentence_norm = LayerNorm(512)
        self.word_norm = LayerNorm(512)
        self.gate = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        scale = 512 ** -0.5
        self.imgproj = nn.Parameter(torch.zeros(512,512),requires_grad=True)

        for name, param in self.clipmodel.named_parameters():
            if not ('gate1' in name or  'adapter1' in name or 'gate2' in name or  'adapter2' in name):
                param.requires_grad = False

        for name,param in self.named_parameters():
            print( name,param.requires_grad)

    def encode_sentences(self,sentences,device='cuda'):
        sentence_inputs = self.tokenizer(sentences).to(device)
        sentence_feature = self.clipmodel.encode_text(sentence_inputs)
        return sentence_feature.to(device)
    
    def encode_sentences_batch(self, sentences, device='cuda', max_sentences=None):
        flat_sentences = []
        truncated_sentences = []
        for sample_sents in sentences:
            num_sents = len(sample_sents)
            if max_sentences is not None and num_sents > max_sentences:
                truncated_sample_sentences=sample_sents[:max_sentences]
            else:
                truncated_sample_sentences = sample_sents
            truncated_sentences.append(truncated_sample_sentences)
            flat_sentences.extend(truncated_sample_sentences)
        sentence_inputs = self.tokenizer(flat_sentences).to(device) 
        sentence_features = self.clipmodel.encode_text(sentence_inputs) 
        start_idx = 0
        all_features = []
        for sample_sents in truncated_sentences:
            num_sents = len(sample_sents)
            sample_features = sentence_features[start_idx:start_idx+num_sents]
            all_features.append(sample_features)
            start_idx += num_sents
        return all_features
    
    def merge_sentences(self,detaillist,sentencelist):
        for i in range(len(detaillist)):
            sentencelist[i].insert(0, detaillist[i])
        return sentencelist
    
    def pad_and_mask(self, samples, max_sentences,device='cuda'):
        batch_size = len(samples)
        input_dim = samples[0].size(1)
        padded_samples = torch.zeros((batch_size, max_sentences, input_dim),device=device)
        mask = torch.zeros((batch_size, max_sentences), dtype=torch.bool,device=device)
        for i, sample in enumerate(samples):
            sample=sample.to(device)
            length = sample.size(0)
            padded_samples[i, :length, :] = sample
            mask[i, :length] = 1

        return padded_samples, mask

    def get_attn(self,image, detail_text, word, idx=11, device='cuda'):
        self.clipmodel = self.clipmodel.to(device)
        image_features,image_hidden_token = self.clipmodel.encode_image_full(image,noproj=True)
        word_token = self.tokenizer(word).to(device)
        word_features,_ = self.clipmodel.encode_text_full_andfeature(word_token,noproj=True)
        detail_text_split = [split_sentence(input_text) for input_text in detail_text]
        detail_text_merge = self.merge_sentences(detail_text, detail_text_split)
        detail_features_list = self.encode_sentences_batch(detail_text_merge, device=device,
                                                           max_sentences=self.max_sentences)
        padded_detail_features, detail_mask = self.pad_and_mask(detail_features_list, self.max_sentences, device=device)
        all_detail_features = padded_detail_features[:, 0, :]
        for blk in self.detailproj:
            padded_detail_features = blk(padded_detail_features)
        padded_detail_features = self.sentence_norm(padded_detail_features)
        word_features = self.word_norm(word_features)
        fusion_feature = image_hidden_token
        fusion_feature = self.fusion_proj(fusion_feature)
        fusion_feature = self.img_norm(fusion_feature)
        padded_detail_features = self.detailfusSelfblock(padded_detail_features.permute(1, 0, 2)).permute(1, 0, 2)
        fusion_feature, last_attn = self.imgfusSelfblock.get_attn(fusion_feature.permute(1, 0, 2))
        fusion_feature = fusion_feature.permute(1, 0, 2)
        for blk in self.crossblockS:
            fusion_feature, last_attn1 = blk.get_attn(padded_detail_features, fusion_feature, ~detail_mask)
            sentence_out_feature = fusion_feature[:, 0, :].squeeze(1)
        for blk in self.crossblockW:
            fusion_feature, last_attn1 = blk.get_attn(word_features, fusion_feature)
        fusionimage_features = fusion_feature[:, 0, :].squeeze(1)
        fusionimage_features = fusionimage_features @ self.imgproj
        return self.clipmodel.get_img_attn(image,idx), last_attn1

    def encode_img(self,image, detail_text, word):
        device = next(self.parameters()).device
        self.clipmodel = self.clipmodel.to(device)
        image_features,image_hidden_token = self.clipmodel.encode_image_full(image,noproj=True)
        word_token = self.tokenizer(word).to(device)
        word_features,_ = self.clipmodel.encode_text_full_andfeature(word_token,noproj=True)
        detail_text_split = [split_sentence(input_text) for input_text in detail_text]
        detail_text_merge = self.merge_sentences(detail_text, detail_text_split)
        detail_features_list = self.encode_sentences_batch(detail_text_merge, device=device,
                                                           max_sentences=self.max_sentences)
        padded_detail_features, detail_mask = self.pad_and_mask(detail_features_list, self.max_sentences, device=device)
        all_detail_features = padded_detail_features[:, 0, :]
        for blk in self.detailproj:
            padded_detail_features = blk(padded_detail_features)
        padded_detail_features = self.sentence_norm(padded_detail_features)
        word_features = self.word_norm(word_features)
        fusion_feature = image_hidden_token
        fusion_feature = self.fusion_proj(fusion_feature)
        fusion_feature = self.img_norm(fusion_feature)
        padded_detail_features = self.detailfusSelfblock(padded_detail_features.permute(1, 0, 2)).permute(1, 0, 2)
        fusion_feature, last_attn = self.imgfusSelfblock.get_attn(fusion_feature.permute(1, 0, 2))
        fusion_feature = fusion_feature.permute(1, 0, 2)
        for blk in self.crossblockS:
            fusion_feature= blk(padded_detail_features, fusion_feature, ~detail_mask)
            sentence_out_feature = fusion_feature[:, 0, :].squeeze(1)
        for blk in self.crossblockW:
            fusion_feature = blk(word_features, fusion_feature)
        fusionimage_features = fusion_feature[:, 0, :].squeeze(1)
        fusionimage_features = fusionimage_features @ self.imgproj
        alpha = torch.sigmoid(self.gate)
        fusionimage_features = alpha * image_features + (1 - alpha) * fusionimage_features
        return fusionimage_features

    def forward(self, image, text, detail_text, word, is_train=False,device='cuda',stage=True):
        self.clipmodel = self.clipmodel.to(device)
        image_features,image_hidden_token = self.clipmodel.encode_image_full(image,noproj=True)
        text_token = self.tokenizer(text).to(device)
        text_features = self.clipmodel.encode_text(text_token)
        word_token = self.tokenizer(word).to(device)
        word_features,_ = self.clipmodel.encode_text_full_andfeature(word_token,noproj=True)
        if  True:
            detail_text_split = [split_sentence(input_text) for input_text in detail_text]
            detail_text_merge = self.merge_sentences(detail_text,detail_text_split)       
            detail_features_list = self.encode_sentences_batch(detail_text_merge,device=device,max_sentences=self.max_sentences)
            padded_detail_features, detail_mask = self.pad_and_mask(detail_features_list, self.max_sentences,device=device)
            all_detail_features = padded_detail_features[:,0,:]
            for blk in self.detailproj:
                padded_detail_features = blk(padded_detail_features)
            padded_detail_features = self.sentence_norm(padded_detail_features)
            word_features = self.word_norm(word_features)
            fusion_feature = image_hidden_token
            fusion_feature = self.fusion_proj(fusion_feature)
            fusion_feature = self.img_norm(fusion_feature)
            padded_detail_features = self.detailfusSelfblock(padded_detail_features.permute(1, 0, 2)).permute(1, 0, 2)
            fusion_feature = self.imgfusSelfblock(fusion_feature.permute(1, 0, 2)).permute(1, 0, 2)
            for blk in self.crossblockS:
                fusion_feature = blk(padded_detail_features,fusion_feature,~detail_mask)
                sentence_out_feature = fusion_feature[:,0,:].squeeze(1)
            for blk in self.crossblockW:
                fusion_feature = blk(word_features,fusion_feature)
            fusionimage_features = fusion_feature[:,0,:].squeeze(1)
            fusionimage_features = fusionimage_features @ self.imgproj
            alpha = torch.sigmoid(self.gate)
            fusionimage_features = alpha*image_features + (1-alpha)*fusionimage_features

        else:
            image_features=image_features

        logit_scale = self.clipmodel.logit_scale.exp()
        logits_per_image =   fusionimage_features @ text_features.t()

        logits_per_text = logits_per_image.t()
        if not is_train:
            return logits_per_image, logits_per_text
        else:
            return logits_per_image, logits_per_text,all_detail_features,text_features