#!/usr/bin/env python
# COPYRIGHT 2020. Fred Fung. Boston University.
"""
Baseline Siamese model for vehicle retrieval task on CityFlow-NL
"""
import torch
import torch.nn.functional as F
from model import ft_net_SE, ft_net 
from transformers import AutoTokenizer, AutoModel
from DeBERTa import deberta

class SiameseBaselineModel(torch.nn.Module):
    def __init__(self, model_cfg, init_model):
        super().__init__()
        self.model_cfg = model_cfg
        self.resnet50 = ft_net_SE( class_num = 2498, droprate=0.2, stride=1, pool='gem', circle =True, init_model = init_model)
        #self.resnet50 = ft_net( class_num = 2498, droprate=0.2, stride=1, pool='avg+max',circle =True)
        #self.bert_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.bert_model = AutoModel.from_pretrained("roberta-base")
        self.logit_scale = torch.nn.Parameter(torch.ones(()), requires_grad=True)
        #self.lang_fc = torch.nn.Linear(768, 1024)
        self.lang_fc = torch.nn.Linear(768, 4096)
        if model_cfg.deberta:
            self.bert_model = deberta.DeBERTa(pre_trained='base') 
            self.bert_model.apply_state()
    def forward(self, input_ids, attention_mask, crops):
        if self.model_cfg.deberta:
            outputs = self.bert_model(input_ids)[-1]
        else:
            outputs = self.bert_model(input_ids, attention_mask = attention_mask)
        lang_embeds = torch.mean(outputs.last_hidden_state, dim=1)
        lang_embeds = self.lang_fc(lang_embeds) # 2048
        predict_class_l, lang_embeds = self.resnet50.classifier(lang_embeds) # 3028, 512
        predict_class_v, visual_embeds = self.resnet50(crops) # 3028, 512
        #print(visual_embeds.shape)
        #print(lang_embeds.shape)
        #d = F.pairwise_distance(visual_embeds, lang_embeds)
        #similarity = torch.exp(-d)
        return visual_embeds, lang_embeds, predict_class_v, predict_class_l

    def compute_lang_embed(self, input_ids, attention_mask):
        with torch.no_grad():
            if self.model_cfg.deberta:
                outputs = self.bert_model(input_ids)[-1]
            else:
                outputs = self.bert_model(input_ids, attention_mask = attention_mask)
            lang_embeds = torch.mean(outputs.last_hidden_state, dim=1)
            lang_embeds = self.lang_fc(lang_embeds)
            _, lang_embeds = self.resnet50.classifier(lang_embeds) # 3028, 512
        return lang_embeds

    def compute_similarity_on_frame(self, track, lang_embeds):
        with torch.no_grad():
            crops = track["crops"][0].cuda()
            _, visual_embeds = self.resnet50(crops)
            similarity = 0.
            for lang_embed in lang_embeds:
                d = F.pairwise_distance(visual_embeds, lang_embed)
                similarity += torch.mean(torch.exp(-d))
            similarity = similarity / len(lang_embeds)
        return similarity
