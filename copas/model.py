"""
CoPAS model adapted for shoulder MRI classification.

Original: https://github.com/zqiuak/CoPAS (Nature Communications 2024)
Changes from original:
  - Hardcoded 12 -> self.class_num (supports 7 shoulder diseases)
  - All architecture (Co-Plane Attention, Cross-Modal Attention,
    Correlation Mining) preserved exactly as original
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from copas.resnet3d import generate_model as resnet3D


def ini_weights(module_list):
    for m in module_list:
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Res_3D_Encoder(nn.Module):
    def __init__(self, kargs, **kwargs):
        super().__init__()
        layer = kargs.model_depth
        if layer == 50:
            self.model = resnet3D(kargs)
            self.feature_channel = 2048
        elif layer == 18:
            self.model = resnet3D(kargs)
            self.feature_channel = 512

    def forward(self, x, squeeze_to_vector=False, pool="max"):
        assert x.dim() == 5
        if pool == "avg":
            pool_func = F.adaptive_avg_pool3d
        else:
            pool_func = F.adaptive_max_pool3d
        x = self.model(x)
        if squeeze_to_vector:
            x = pool_func(x, 1)
            x = torch.flatten(x, start_dim=1)
        else:
            x = x.transpose(1, 2)
            x = pool_func(x, (self.feature_channel, 1, 1))
            x = torch.flatten(x, start_dim=2)
        return x


class Co_Plane_Att(nn.Module):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.emb_dim = embed_dim
        self.mq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mk1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mk2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mv1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mv2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        ini_weights(self.modules())

    def forward(self, main_f, co_f1, co_f2):
        res = main_f
        q = self.mq(main_f)
        k1 = self.mk1(co_f1).permute(0, 2, 1)
        k2 = self.mk2(co_f2).permute(0, 2, 1)
        v1 = self.mv1(co_f1)
        v2 = self.mv2(co_f2)
        att1 = torch.matmul(q, k1) / np.sqrt(self.emb_dim)
        att1 = torch.softmax(att1, -1)
        att2 = torch.matmul(q, k2) / np.sqrt(self.emb_dim)
        att2 = torch.softmax(att2, -1)
        out1 = torch.matmul(att1, v1)
        out2 = torch.matmul(att2, v2)
        self.attmap1 = att1.detach().cpu()
        self.attmap2 = att2.detach().cpu()
        f = self.norm(0.5 * (out1 + out2) + res)
        f = f.transpose(1, 2)
        f = F.adaptive_max_pool1d(f, 1)
        f = torch.flatten(f, start_dim=1)
        return f


class Cross_Modal_Att(nn.Module):
    def __init__(self, feature_channel, **kwargs):
        super().__init__(**kwargs)
        self.transform_matrix = nn.Linear(2 * feature_channel, feature_channel)
        self.norm = nn.BatchNorm1d(num_features=feature_channel)
        ini_weights([self.transform_matrix, self.norm])

    def forward(self, pdw_f, aux_f):
        assert aux_f.dim() == pdw_f.dim()
        add_f = pdw_f + aux_f
        sub_f = torch.cat((pdw_f, aux_f), dim=1)
        att_f = self.transform_matrix(sub_f)
        att_f = torch.relu(att_f)
        att_f = torch.softmax(att_f, -1)
        f = add_f * att_f
        return f


class Branch_Classifier(nn.Module):
    def __init__(self, classnum, feature_channel, dropout_rate):
        super().__init__()
        self.classifiers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_channel, classnum))
        ini_weights(self.classifiers)

    def forward(self, f):
        if f.dim() == 3:
            f = f.squeeze(2)
        return self.classifiers(f)


class CoPAS_Shoulder(nn.Module):
    """CoPAS model for shoulder MRI multi-label classification.

    Architecture (identical to original CoPAS):
      - 3 PD encoders (sag, cor, axi) with Co-Plane Attention
      - 2 auxiliary encoders (T1WI, T2WI) with Cross-Modal Attention
      - Correlation Mining fusion head
      - Per-branch + final classifiers

    Input: list of 5 tensors [sag_PD, cor_PD, axi_PD, sag_T1WI, cor_T2WI]
           each shaped [B, 1, Z, H, W]
    Output: (final_pred, sag_pred, cor_pred, axi_pred), each [B, class_num]
    """

    def __init__(self, kargs):
        super().__init__()
        self.kargs = kargs
        self.class_num = kargs.ClassNum
        self.branch = kargs.active_branch
        self.dropout_rate = 0.05
        self.backbone = kargs.backbone

        # Device: all modules init on CPU, use .cuda() + DataParallel externally
        self.device_list = ["cpu"] * 3

        self.__make_encoder__()
        self.__make_co_plane_att__()
        self.__make_cross_modal_att__()
        self.__make_classifier__()

        n_cls = self.class_num
        self.mining_conv = nn.Conv3d(1, n_cls, (n_cls, n_cls, n_cls))
        ini_weights([self.mining_conv, self.multi_view_classifier])

        model_param = sum(np.prod(v.size()) for _, v in self.named_parameters()) / 1e6
        print('CoPAS model param = %.2f MB' % model_param)

    def __make_encoder__(self):
        encoder_func = Res_3D_Encoder
        if self.branch[0]:
            self.sag_enc = encoder_func(self.kargs).to(self.device_list[0])
            if not self.kargs.no_cross_modal:
                self.t2w_enc = encoder_func(self.kargs).to(self.device_list[0])
        if self.branch[1]:
            self.cor_enc = encoder_func(self.kargs).to(self.device_list[1])
            if not self.kargs.no_cross_modal:
                self.t1w_enc = encoder_func(self.kargs).to(self.device_list[0])
        if self.branch[2]:
            self.axi_enc = encoder_func(self.kargs).to(self.device_list[2])

    def __make_co_plane_att__(self):
        if self.kargs.no_co_att:
            return
        emb_dim = self.kargs.emb_dim
        if self.branch[0]:
            self.sag_att = Co_Plane_Att(emb_dim).to(self.device_list[0])
        if self.branch[1]:
            self.cor_att = Co_Plane_Att(emb_dim).to(self.device_list[1])
        if self.branch[2]:
            self.axi_att = Co_Plane_Att(emb_dim).to(self.device_list[2])

    def __make_cross_modal_att__(self):
        if self.kargs.no_cross_modal:
            return
        if self.branch[0]:
            self.sag_cross_att = Cross_Modal_Att(
                self.sag_enc.feature_channel).to(self.device_list[0])
        if self.branch[1]:
            self.cor_cross_att = Cross_Modal_Att(
                self.cor_enc.feature_channel).to(self.device_list[1])

    def __make_classifier__(self):
        n_cls = self.class_num
        if self.branch[0]:
            self.sag_classifier = Branch_Classifier(
                n_cls, self.sag_enc.feature_channel, self.dropout_rate).to(self.device_list[0])
        if self.branch[1]:
            self.cor_classifier = Branch_Classifier(
                n_cls, self.cor_enc.feature_channel, self.dropout_rate).to(self.device_list[1])
        if self.branch[2]:
            self.axi_classifier = Branch_Classifier(
                n_cls, self.axi_enc.feature_channel, self.dropout_rate).to(self.device_list[2])

        if self.kargs.no_corr_mining:
            self.multi_view_classifier = nn.Sequential(
                nn.Linear(3 * n_cls, n_cls)).to(self.device_list[2])
        else:
            self.multi_view_classifier = nn.Sequential(
                nn.Linear(n_cls, n_cls)).to(self.device_list[2])

    # ---- Branch forward ----
    def __sag_branch__(self, input, device=None):
        sag_img, cor_img, axi_img, t2_img, _ = input

        if self.kargs.no_co_att:
            pdw_f = self.sag_enc(sag_img, squeeze_to_vector=True)
        else:
            sag_cor = cor_img.transpose(2, 4)
            sag_axi = torch.rot90(axi_img.transpose(2, 4), k=1, dims=[3, 4]).detach()
            main_f = self.sag_enc(sag_img)
            with torch.no_grad():
                co_f1 = self.sag_enc(sag_cor)
                co_f2 = self.sag_enc(sag_axi)
            pdw_f = self.sag_att(main_f, co_f1, co_f2)

        if self.kargs.no_cross_modal:
            pred = self.sag_classifier(pdw_f)
        else:
            aux_f = self.t2w_enc(t2_img, squeeze_to_vector=True)
            cross_m_f = self.sag_cross_att(pdw_f, aux_f)
            pred = self.sag_classifier(cross_m_f)
        return pred

    def __cor_branch__(self, input, device=None):
        sag_img, cor_img, axi_img, _, t1_img = input

        if self.kargs.no_co_att:
            pdw_f = self.cor_enc(cor_img, squeeze_to_vector=True)
        else:
            cor_sag = sag_img.transpose(2, 4)
            cor_axi = torch.rot90(axi_img.transpose(2, 3), k=2, dims=[3, 4])
            main_f = self.cor_enc(cor_img)
            with torch.no_grad():
                co_f1 = self.cor_enc(cor_sag)
                co_f2 = self.cor_enc(cor_axi)
            pdw_f = self.cor_att(main_f, co_f1, co_f2)

        if self.kargs.no_cross_modal:
            pred = self.cor_classifier(pdw_f)
        else:
            aux_f = self.t1w_enc(t1_img, squeeze_to_vector=True)
            cross_m_f = self.cor_cross_att(pdw_f, aux_f)
            pred = self.cor_classifier(cross_m_f)
        return pred

    def __axi_branch__(self, input, device=None):
        sag_img, cor_img, axi_img, _, _ = input

        if self.kargs.no_co_att:
            pdw_f = self.axi_enc(axi_img, squeeze_to_vector=True)
        else:
            axi_cor = cor_img.transpose(2, 3)
            axi_sag = sag_img.transpose(2, 4)
            main_f = self.axi_enc(axi_img)
            with torch.no_grad():
                co_f1 = self.axi_enc(axi_cor)
                co_f2 = self.axi_enc(axi_sag)
            pdw_f = self.axi_att(main_f, co_f1, co_f2)

        pred = self.axi_classifier(pdw_f)
        return pred

    # ---- Correlation Mining fusion ----
    def __discovery__(self, sag_pred, cor_pred, axi_pred, device=None):
        with torch.no_grad():
            sag_pred = torch.sigmoid(sag_pred)
            cor_pred = torch.sigmoid(cor_pred)
            axi_pred = torch.sigmoid(axi_pred)

        if self.kargs.no_corr_mining:
            pred_matrix = torch.cat((sag_pred, cor_pred, axi_pred), dim=1)
            return self.multi_view_classifier(pred_matrix)

        union_prob = sag_pred * cor_pred * axi_pred
        sag_t = sag_pred.unsqueeze(2).unsqueeze(2)
        cor_t = cor_pred.unsqueeze(2).unsqueeze(1)
        axi_t = axi_pred.unsqueeze(1).unsqueeze(1)
        pred_matrix = (sag_t * cor_t * axi_t).unsqueeze(1)
        fin_att = torch.flatten(self.mining_conv(pred_matrix), start_dim=1)
        fin_pred = union_prob * fin_att
        return fin_pred

    # ---- Forward ----
    def forward(self, input):
        # Support stacked tensor [B, 5, C, Z, H, W] for DataParallel
        if isinstance(input, torch.Tensor):
            input = [input[:, i] for i in range(input.shape[1])]
        if self.branch[0]:
            sag_pred = self.__sag_branch__(input)
            final_pred = sag_pred
        if self.branch[1]:
            cor_pred = self.__cor_branch__(input)
            final_pred = cor_pred
        if self.branch[2]:
            axi_pred = self.__axi_branch__(input)
            final_pred = axi_pred
        if sum(self.branch) == 1:
            return final_pred, final_pred, final_pred, final_pred
        if sum(self.branch) == 3:
            final_pred = self.__discovery__(sag_pred, cor_pred, axi_pred)
            return final_pred, sag_pred, cor_pred, axi_pred

    # ---- Loss ----
    def criterion(self, pred, label, act_task=-1, final=False):
        task_weights = [1.0] * self.class_num
        pos_weights = torch.tensor(self.kargs.pos_weights, device=label.device)

        final_pred, sag_pred, cor_pred, axi_pred = pred
        sag_loss = 0.0
        cor_loss = 0.0
        axi_loss = 0.0
        final_loss = 0.0
        lossfunc = F.binary_cross_entropy_with_logits
        lossfunc_final = Focal_Loss_with_logits

        for i in range(self.class_num):
            if i == act_task or act_task == -1:
                task_weight = task_weights[i]
            else:
                task_weight = 0.01
            pos_wei = pos_weights[i]
            subject_label = label[:, i:i + 1]

            if self.branch[0]:
                sag_loss += task_weight * lossfunc(
                    sag_pred[:, i:i + 1], subject_label, pos_weight=pos_wei)
            if self.branch[1]:
                cor_loss += task_weight * lossfunc(
                    cor_pred[:, i:i + 1], subject_label, pos_weight=pos_wei)
            if self.branch[2]:
                axi_loss += task_weight * lossfunc(
                    axi_pred[:, i:i + 1], subject_label, pos_weight=pos_wei)
            if final:
                final_loss += task_weight * lossfunc_final(
                    final_pred[:, i:i + 1], subject_label, pos_weight=pos_wei)

        if sum(self.branch) == 1:
            loss = [sag_loss, cor_loss, axi_loss][self.branch.index(1)]
        elif final:
            loss = self.kargs.alpha * (sag_loss + cor_loss + axi_loss) + final_loss
        else:
            loss = sag_loss + cor_loss + axi_loss

        return loss, loss.item()

    # ---- Save/Load ----
    def save_or_load_encoder_para(self, mode="save", path=""):
        if mode == "save":
            act_func = self.__save_para_
        elif mode == "load":
            act_func = self.__load_para_
        else:
            raise Exception("wrong mode")

        if self.branch[0]:
            act_func(self.sag_enc, 'sag_enc', path=path)
            act_func(self.sag_classifier, "sag_cls", path=path)
            if not self.kargs.no_cross_modal:
                act_func(self.t2w_enc, 't2w_enc', path=path)
        if self.branch[1]:
            act_func(self.cor_enc, 'cor_enc', path=path)
            act_func(self.cor_classifier, "cor_cls", path=path)
            if not self.kargs.no_cross_modal:
                act_func(self.t1w_enc, 't1w_enc', path=path)
        if self.branch[2]:
            act_func(self.axi_enc, 'axi_enc', path=path)
            act_func(self.axi_classifier, "axi_cls", path=path)

    def __save_para_(self, model, name, path=""):
        if path:
            torch.save(model.state_dict(),
                       os.path.join(path, "%s_%s_para.pkl" % (model.__class__.__name__, name)))

    def __load_para_(self, model, name, path=""):
        save_path = path or self.kargs.pretrain_folder
        model.load_state_dict(torch.load(
            os.path.join(save_path, "%s_%s_para.pkl" % (model.__class__.__name__, name))))


def Focal_Loss_with_logits(pred, label, pos_weight=None, gamma=2, reduction='mean'):
    """Binary focal loss — no external dependency.

    Args:
        pred:       logits, same shape as label (e.g. [B, 1])
        label:      binary targets (0/1), float
        pos_weight: scalar tensor, positive-class weight
        gamma:      focusing parameter (default 2)
        reduction:  'mean' | 'sum' | 'none'
    """
    # numerically-stable BCE per element
    ce = F.binary_cross_entropy_with_logits(pred, label, reduction='none')

    # p_t = probability assigned to the correct class
    p = torch.sigmoid(pred)
    p_t = p * label + (1 - p) * (1 - label)
    focal_weight = (1 - p_t) ** gamma

    # class-balanced alpha (matches original CoPAS weighting)
    if pos_weight is not None:
        alpha = pos_weight / (1 + pos_weight)          # weight for positive
        alpha_t = label * alpha + (1 - label) * (1 - alpha)  # weight for negative
        loss = alpha_t * focal_weight * ce
    else:
        loss = focal_weight * ce

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss
