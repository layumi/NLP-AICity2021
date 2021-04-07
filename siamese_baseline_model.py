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
from resnet_t2v import tsm_resnet50
from model import GeM

class SiameseBaselineModel(torch.nn.Module):
    def __init__(self, model_cfg, init_model=None):
        super().__init__()
        self.model_cfg = model_cfg
        self.nseg = model_cfg.nseg
        self.netvlad = model_cfg.netvlad
        if self.nseg>1:
            rstride = 2
        else: 
            rstride = 1
        self.resnet50 = ft_net_SE( class_num = 2498, droprate=model_cfg.droprate, stride = rstride, pool='gem', circle =True, init_model = init_model, netvlad = False)
        #self.bert_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        if self.nseg>1:
            self.gem = GeM(dim =4096)
        self.bert_model = AutoModel.from_pretrained("roberta-base")
        # remove pooler to save memory
        #self.bert_model.pooler = torch.nn.Sequential()
        self.logit_scale1 = torch.nn.Parameter(torch.ones(()), requires_grad=True)
        self.logit_scale2 = torch.nn.Parameter(torch.ones(()), requires_grad=True)
        #self.lang_fc = torch.nn.Linear(768, 1024)
        if model_cfg.netvlad:
            self.lang_fc = torch.nn.Sequential(*[
                    NetVLAD(dim=768),
                    torch.nn.Linear(9*768, 4096) ])
            #self.visual_fc = torch.nn.Sequential(*[
            #        torch.nn.Conv2d(2048, 768, kernel_size=(1,1)),
            #        NetVLAD(dim=768),
            #        torch.nn.Linear(9*768, 4096) ])
        else:
            self.lang_fc = torch.nn.Sequential(*[
                      torch.nn.BatchNorm1d(768*2), 
                      torch.nn.Linear(768*2, 4096) ])

        #if model_cfg.motion:
        #    self.visual_fc = torch.nn.Sequential(*[
        #            torch.nn.Conv2d(2048, 768, kernel_size=(1,1)),
        #            NetVLAD(dim=768),
        #            torch.nn.Linear(9*768, 4096) ])
        #else: 
        self.visual_fc = torch.nn.Sequential(*[
                      torch.nn.BatchNorm1d(4096),
                      torch.nn.Linear(4096, 4096) ])
        self.lang_fc.apply(weights_init_kaiming)
        self.visual_fc.apply(weights_init_kaiming)

        self.motion = model_cfg.motion
        if model_cfg.motion:
            self.resnet50_m = tsm_resnet50(pretrained= True, num_segments = 4)
        if model_cfg.deberta:
            self.bert_model = deberta.DeBERTa(pre_trained='base') 
            self.bert_model.apply_state()

    def forward(self, input_ids, attention_mask, crops, motion=None):
        # base feature
        if self.model_cfg.fixed: #fix learned model
            self.bert_model.eval()
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
            visual_embeds = self.resnet50.model.features(crops) # N*nseg, 2048, h,w 

        # embedding
        if self.netvlad:
            lang_embeds = outputs.last_hidden_state # N, length, 768
            #visual_embeds = visual_embeds
        else:
            l1 = torch.mean(outputs.last_hidden_state, dim=1)
            l2, _ = torch.max(outputs.last_hidden_state, dim=1)
            lang_embeds = torch.cat((l1,l2), dim=1)

        x1 = self.resnet50.model.avg_pool2(visual_embeds)
        x2 = self.resnet50.model.max_pool2(visual_embeds)
        visual_embeds = torch.cat((x1,x2), dim = 1) ## N*nseg, 4096
        ### nseg -> 1
        if self.nseg>1:
            visual_embeds = visual_embeds.view(-1, self.nseg, 4096).transpose(1,2).unsqueeze(-1)  # N, 4096, nseg, 1 
            visual_embeds = self.gem(visual_embeds)

        if self.motion:
            motion = motion.view(-1, 3, self.model_cfg.CROP_SIZE, self.model_cfg.CROP_SIZE)
            motion_embeds = self.resnet50_m(motion) # 3028, 512
            visual_embeds = visual_embeds + motion_embeds  #40), dim = 1) #4096
            predict_class_m, _ = self.resnet50.classifier(motion_embeds)

        lang_embeds = self.lang_fc(lang_embeds) # learned
        visual_embeds = self.visual_fc(visual_embeds) # learned

        # shared classifier
        predict_class_l, lang_embeds = self.resnet50.classifier(lang_embeds) # learned
        predict_class_v, visual_embeds = self.resnet50.classifier(visual_embeds) # learned

        if self.motion:
            return visual_embeds, lang_embeds, predict_class_v, predict_class_l, predict_class_m
            
        return visual_embeds, lang_embeds, predict_class_v, predict_class_l

    def compute_visual_embed(self, crops, motion=None):
        with torch.no_grad():
            visual_embeds = self.resnet50.model.features(crops)
            x1 = self.resnet50.model.avg_pool2(visual_embeds)
            x2 = self.resnet50.model.max_pool2(visual_embeds)
            visual_embeds = torch.cat((x1,x2), dim = 1) #4096
            if self.motion:
                motion = motion.view(-1, 3, self.model_cfg.CROP_SIZE, self.model_cfg.CROP_SIZE)
                motion_embeds = self.resnet50_m(motion) # 3028, 512
                visual_embeds = visual_embeds + motion_embeds  #40), dim = 1) #4096

            visual_embeds = self.visual_fc(visual_embeds) # learned
            _, visual_embeds = self.resnet50.classifier(visual_embeds) # learned
        return visual_embeds        

    def compute_lang_embed(self, input_ids, attention_mask):
        with torch.no_grad():
            if self.model_cfg.deberta:
                outputs = self.bert_model(input_ids)[-1]
            else:
                outputs = self.bert_model(input_ids, attention_mask = attention_mask)
            if self.netvlad:
                lang_embeds = outputs.last_hidden_state
            else:
                l1 = torch.mean(outputs.last_hidden_state, dim=1)
                l2, _ = torch.max(outputs.last_hidden_state, dim=1)
                lang_embeds = torch.cat((l1,l2), dim=1)

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
