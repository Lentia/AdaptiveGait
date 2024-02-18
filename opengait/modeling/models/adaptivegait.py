import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks


class AFFB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, halving, **kwargs):
        super(AFFB, self).__init__()
        self.halving = halving
        self.conv_w = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=False, **kwargs)
        self.conv_p = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=False, **kwargs)
        self.w = nn.Parameter(torch.ones(1) * 11)

    def forward(self, x):
        gob_feat = self.conv_w(x)
        if self.halving == 0:
            z = self.conv_p(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            z = x.split(split_size, 3)
            z = torch.cat([self.conv_p(_) for _ in z], 3)
        # print(z.size())
        # print(gob_feat.size())

        feat = F.leaky_relu(self.w * gob_feat) + F.leaky_relu(z)
        return feat


class FEM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, halving, **kwargs):
        super(FEM, self).__init__()
        self.halving = halving
        self.conv_ge1 = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=False, **kwargs)
        self.conv_ge2 = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=False, **kwargs)
        self.conv_le1 = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=False, **kwargs)
        self.conv_le2 = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=False, **kwargs)
        self.w = nn.Parameter(torch.ones(1) * 6)

    def forward(self, x):
        # local feature expander
        if self.halving == 0:
            y_le1 = self.conv_le1(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            y_le1 = x.split(split_size, 3)
            y_le1 = torch.cat([self.conv_le1(_) for _ in y_le1], 3)
        y_le2 = self.conv_le2(x)
        y_le = F.leaky_relu(torch.cat([y_le2, y_le1], dim=3))

        # global feature expander
        y_ge1 = self.conv_ge1(x)
        y_ge2 = self.conv_ge2(x)
        y_ge = F.leaky_relu(torch.cat([y_ge1, y_ge2], dim=3))

        y = F.leaky_relu(torch.cat([self.w * y_ge, y_le], dim=1))
        return y



class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)


class AdaptiveGait(BaseModel):
    """
        GaitGL: Gait Recognition via Effective Global-Local Feature Representation and Local Temporal Aggregation
        Arxiv : https://arxiv.org/pdf/2011.01461.pdf
    """

    def __init__(self, *args, **kargs):
        super(AdaptiveGait, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):
        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']
        dataset_name = self.cfgs['data_cfg']['dataset_name']

        if dataset_name in ['OUMVLP', 'GREW']:
            # For OUMVLP and GREW
            self.conv3d = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
                BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
            )
            self.LTA = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )

            self.AFFB1 = nn.Sequential(
                AFFB(in_c[0], in_c[1], halving=2, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                AFFB(in_c[1], in_c[1], halving=2, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.MaxPool0 = nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.AFFB2 = nn.Sequential(
                AFFB(in_c[1], in_c[2], halving=2, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                AFFB(in_c[2], in_c[2], halving=2, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.FEM = nn.Sequential(
                AFFB(in_c[2], in_c[3], halving=2, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                FEM(in_c[3], in_c[3], halving=2, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
        else:
            # For CASIA-B or other unstated datasets.
            self.conv3d = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )
            self.LTA = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )

            self.AFFB1 = AFFB(in_c[0], in_c[1], halving=2, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1) )
            self.MaxPool0 = nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.AFFB2 = AFFB(in_c[1], in_c[2], halving=3, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            self.FEM = FEM(in_c[2], in_c[2], halving=4, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = GeMHPP()

        self.Head0 = SeparateFCs(64, in_c[-1]*2, in_c[-1]*2)

        if 'SeparateBNNecks' in model_cfg.keys():
            self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
            self.Bn_head = False
        else:
            self.Bn = nn.BatchNorm1d(in_c[-1]*2)
            self.Head1 = SeparateFCs(64, in_c[-1]*2, class_num)
            self.Bn_head = True

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        outs = self.conv3d(sils)
        outs = self.LTA(outs)

        outs = self.AFFB1(outs)
        outs = self.MaxPool0(outs)

        outs = self.AFFB2(outs)
        outs = self.FEM(outs)  # [n, c, s, h, w]

        outs = self.TP(outs, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs = self.HPP(outs)  # [n, c, p]

        gait = self.Head0(outs)  # [n, c, p]

        if self.Bn_head:  # Original GaitGL Head
            bnft = self.Bn(gait)  # [n, c, p]
            logi = self.Head1(bnft)  # [n, c, p]
            embed = bnft
        else:  # BNNechk as Head
            bnft, logi = self.BNNecks(gait)  # [n, c, p]
            embed = gait

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval