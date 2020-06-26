import torch.nn as nn
import torch
from OCRForClothes.HRNet.OCR import SpatialGather_Module, SpatialOCR_Module
from OCRForClothes.HRNet.batchnorm import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x, ocr=True, segSize=False):
        result = self.encoder(x)
        if segSize:
            self.decoder.use_softmax = True

        out_aux = self.decoder(result, ocr, segSize)
        self.decoder.use_softmax = False
        return out_aux

class Resnet_OCR(nn.Module):
    def __init__(self, encoder, decoder, size):
        super(Resnet_OCR, self).__init__()
        self.EncoderDecoder = EncoderDecoder(encoder, decoder)
        self.size = size
        self.use_softmax = False
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(59, 512,
                      kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )

        self.ocr_gather_head = SpatialGather_Module(59)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=512,
                                                 key_channels=256,
                                                 out_channels=512,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )

        self.cls_head = nn.Conv2d(512, 59, kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, x, ocr=True, segSize=False):
        if segSize:
            self.use_softmax = True
        out_aux = self.EncoderDecoder(x, ocr, segSize)
        feats = self.conv3x3_ocr(out_aux)
        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)
        
        out_aux_seg = []
        out = self.cls_head(feats)


        if self.use_softmax:  # is True during inference
            out = nn.functional.interpolate(
                out, size=segSize, mode='bilinear', align_corners=False)
            out = nn.functional.softmax(out, dim=1)
            out_aux = nn.functional.interpolate(
                out_aux, size=segSize, mode='bilinear', align_corners=False)
            out_aux = nn.functional.softmax(out_aux, dim=1)
        else:
            out = nn.functional.log_softmax(out, dim=1)
            out_aux = nn.functional.log_softmax(out_aux, dim=1)

        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)
        self.use_softmax = False
        return out_aux_seg


class Baseline(nn.Module):
    def __init__(self, encoder, decoder, size):
            super(Baseline, self).__init__()
            self.EncoderDecoder = EncoderDecoder(encoder, decoder)
            self.size = size
            self.use_softmax = False   
    def forward(self, x, ocr=True, segSize=False):
        if segSize:
            self.use_softmax = True
        out_aux = self.EncoderDecoder(x, ocr, segSize)

        if self.use_softmax:  # is True during inference
            out_aux = nn.functional.interpolate(
                out_aux, size=segSize, mode='bilinear', align_corners=False)
            out_aux = nn.functional.softmax(out_aux, dim=1)
        else:
            out_aux = nn.functional.log_softmax(out_aux, dim=1)
        
        return out_aux
