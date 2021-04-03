#!/usr/bin/env python
# COPYRIGHT 2020. Fred Fung. Boston University.
"""
Baseline Siamese model for vehicle retrieval task on CityFlow-NL
"""
import torch
import torch.nn.functional as F
from model import ft_net_SE, ft_net, NetVLAD, weights_init_kaiming
from transformers import AutoTokenizer, AutoModel
from DeBERTa import deberta

class SiameseBaselineModel(torch.nn.Module):
    def __init__(self, model_cfg, init_model=None):
        super().__init__()
        self.model_cfg = model_cfg
        self.netvlad = model_cfg.netvlad
        self.resnet50 = ft_net_SE( class_num = 2498, droprate=0.2, stride=1, pool='gem', circle =True, init_model = init_model, netvlad = False)
        #self.resnet50 = ft_net( class_num = 2498, droprate=0.2, stride=1, pool='avg+max',circle =True)
        #self.bert_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.bert_model = AutoModel.from_pretrained("roberta-base")
        self.logit_scale1 = torch.nn.Parameter(torch.ones(()), requires_grad=True)
        self.logit_scale2 = torch.nn.Parameter(torch.ones(()), requires_grad=True)
        #self.lang_fc = torch.nn.Linear(768, 1024)
        if model_cfg.netvlad:
            self.lang_fc = torch.nn.Sequential(*[
                    NetVLAD(dim=768),
                    torch.nn.Linear(9*768, 4096) ])
            self.visual_fc = torch.nn.Sequential(*[
                    torch.nn.Conv2d(2048, 768, kernel_size=(1,1)),
                    NetVLAD(dim=768),
                    torch.nn.Linear(9*768, 4096) ])
        else:
            self.lang_fc = torch.nn.Sequential(*[
                      torch.nn.BatchNorm1d(768*2), 
                      torch.nn.Linear(768*2, 4096) ]) 
            self.visual_fc = torch.nn.Sequential(*[
                      torch.nn.BatchNorm1d(4096),
                      torch.nn.Linear(4096, 4096) ])
            self.lang_fc.apply(weights_init_kaiming)
            self.visual_fc.apply(weights_init_kaiming)

        self.motion = model_cfg.motion
        if model_cfg.motion:
            self.resnet50_m = ft_net( class_num = 2498, droprate=0.2, stride=1, pool='gem', circle =True, init_model = None, netvlad = model_cfg.netvlad)
            self.resnet50_m.classifier = self.resnet50.classifier #share classifier
            self.car_fc = torch.nn.Linear(512, 512)
            self.bg_fc = torch.nn.Linear(512, 512)
            self.combine_v = torch.nn.Linear(1024, 512)
        if model_cfg.deberta:
            self.bert_model = deberta.DeBERTa(pre_trained='base') 
            self.bert_model.apply_state()

    def forward(self, input_ids, attention_mask, crops, motion=None):
        # base feature
        if self.model_cfg.fixed: #fix learned model
            self.bert_model.eval()
            self.resnet50.model.eval()
            with torch.no_grad():
                if self.model_cfg.deberta:
                    outputs = self.bert_model(input_ids)[-1]
                else:
                    outputs = self.bert_model(input_ids, attention_mask = attention_mask)
                visual_embeds = self.resnet50.model.features(crops) 
        else:
            if self.model_cfg.deberta:
                outputs = self.bert_model(input_ids)[-1]
            else:
                outputs = self.bert_model(input_ids, attention_mask = attention_mask)
            visual_embeds = self.resnet50.model.features(crops) # N, 2048, h,w 

        # embedding
        if self.netvlad:
            lang_embeds = outputs.last_hidden_state
            lang_embeds = lang_embeds.transpose(1,2).contiguous().unsqueeze(-1) # N, 768, length, 1
        else:
            l1 = torch.mean(outputs.last_hidden_state, dim=1)
            l2, _ = torch.max(outputs.last_hidden_state, dim=1)
            lang_embeds = torch.cat((l1,l2), dim=1)
            x1 = self.resnet50.model.avg_pool2(visual_embeds)
            x2 = self.resnet50.model.max_pool2(visual_embeds)
            visual_embeds = torch.cat((x1,x2), dim = 1) #4096

        lang_embeds = self.lang_fc(lang_embeds) # learned
        visual_embeds = self.visual_fc(visual_embeds) # learned

        # shared classifier
        predict_class_l, lang_embeds = self.resnet50.classifier(lang_embeds) # learned
        predict_class_v, visual_embeds = self.resnet50.classifier(visual_embeds) # learned

        if self.motion:
            predict_class_m, motion_embeds = self.resnet50_m(crops) # 3028, 512
            predict_class_v = predict_class_m + predict_class_v
            visual_embeds = self.car_fc(visual_embeds) + self.bg_fc(motion_embeds) 
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
            if self.netvlad:
                lang_embeds = outputs.last_hidden_state
                lang_embeds = lang_embeds.transpose(1,2).contiguous().unsqueeze(-1)
            else:
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
