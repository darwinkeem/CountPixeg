import torch
import torch.nn as nn
import torch.nn.functional as F
#import config as C

#args = C.parse_args()
#for attr, value in args.__dict__.items():
#    setattr(C, attr, value)

class Convolution_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.tran_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.double33conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.tran_channels(x)
        residual = x
        x = self.double33conv(x)

        x = x + residual

        return self.ReLU(x)

class Up_cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                                nn.BatchNorm2d(out_channels))
        self.ReLU =nn.ReLU()

    def forward(self, x1, x2):
        x2 = self.up(x2)
        w_out = x1.shape[-1]
        h_out = x1.shape[-2]
        x2 = F.interpolate(x2, size=[h_out, w_out], mode='bilinear')

        return self.ReLU(x1 + x2)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                  Convolution_Block(in_channels, out_channels))

    def forward(self, x):
        return self.down(x)

class SpatialGather_Module(nn.Module):
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, channels, height, width = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, channels, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # bat x hw x c
        probs = F.softmax(self.scale * probs, dim=2) # bat x k x hw
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3) # bat x k x c
        return ocr_context

class ObjectAttentionBlock(nn.Module):
    def __init__(self, in_channels, key_channels, scale=1):
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
            nn.Conv2d(self.key_channels, self.key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU()
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(self.key_channels),
            #nn.ReLU(),
            #nn.Conv2d(self.key_channels, self.key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(self.key_channels),
            #nn.ReLU()
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(self.key_channels),
            #nn.ReLU()
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(self.key_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)

        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)

        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim =1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)

        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)

        return context

class ObjectAttentionBlock2D(ObjectAttentionBlock):
    def __init__(self, in_channels, key_channels, scale = 1):
        super(ObjectAttentionBlock2D, self).__init__(in_channels, key_channels, scale)

class SpatialOCR_Module(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.1):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels, scale)

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output

class OCR(nn.Module):
    def __init__(self, last_input_channel, n_classes):
        super(OCR, self).__init__()
        self.n_classes = n_classes
        self.last_input_channel = last_input_channel
        self.out = Out(self.last_input_channel, self.n_classes)
        self.ocr_gather_head = SpatialGather_Module(self.n_classes)

        ocr_mid_channels = int(self.last_input_channel / 2)
        ocr_key_channels = int(ocr_mid_channels / 2)
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(self.last_input_channel, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU()
        )

        self.ocr_distri_head = SpatialOCR_Module(ocr_mid_channels,
                                                 ocr_key_channels,
                                                 ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05)

        self.cls_head = nn.Conv2d(ocr_mid_channels, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(self.last_input_channel, self.last_input_channel,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.last_input_channel),
            nn.ReLU(),
            nn.Conv2d(self.last_input_channel, n_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, feats):
        out_aux_seg = []

        out_aux = self.out(feats)

        feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, out_aux)

        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux_seg.append(out)
        out_aux_seg.append(out_aux)

        return out_aux_seg

class Out(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.out = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.out(x)

class UPPNet(nn.Module):
    def __init__(self, n_ch, n_classes, ocr=True, check=True):
        super(UPPNet, self).__init__()
        self.n_classes= n_classes
        self.check = check
        channels = [64, 128, 256, 512, 1024]
        self.input_layer = Convolution_Block(n_ch, channels[0])
        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])
        self.down4 = Down(channels[3], channels[4])

        self.Up0_1 = Up_cat(channels[1], channels[0])
        self.Conv0_1 = Convolution_Block(channels[0], channels[0])

        self.Up1_1 = Up_cat(channels[2], channels[1])
        self.Conv1_1 = Convolution_Block(channels[1], channels[1])

        self.Up0_2 = Up_cat(channels[1], channels[0])
        self.Conv0_2 = Convolution_Block(channels[0]*2, channels[0])

        self.Up2_1 = Up_cat(channels[3], channels[2])
        self.Conv2_1 = Convolution_Block(channels[2], channels[2])

        self.Up1_2 = Up_cat(channels[2], channels[1])
        self.Conv1_2 = Convolution_Block(channels[1]*2, channels[1])

        self.Up0_3 = Up_cat(channels[1], channels[0])
        self.Conv0_3 = Convolution_Block(channels[0]*3, channels[0])

        self.Up3_1 = Up_cat(channels[4], channels[3])
        self.Conv3_1 = Convolution_Block(channels[3], channels[3])

        self.Up2_2 = Up_cat(channels[3], channels[2])
        self.Conv2_2 = Convolution_Block(channels[2]*2, channels[2])

        self.Up1_3 = Up_cat(channels[2], channels[1])
        self.Conv1_3 = Convolution_Block(channels[1]*3, channels[1])

        self.Up0_4 = Up_cat(channels[1], channels[0])
        self.Conv0_4 = Convolution_Block(channels[0]*4, channels[0])

        if ocr is True:
            self.out = OCR(channels[0], n_classes)
        else:
            self.out = Out(channels[0], n_classes)
        if self.check is True:
            self.last_layer1 = nn.Conv2d(channels[0], n_classes, kernel_size=1, stride=1, padding=0)
            self.last_layer2 = nn.Conv2d(channels[0], n_classes, kernel_size=1, stride=1, padding=0)
            self.last_layer3 = nn.Conv2d(channels[0], n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.input_layer(x)
        x1_0 = self.down1(x)
        x2_0 = self.down2(x1_0)
        x3_0 = self.down3(x2_0)
        x4_0 = self.down4(x3_0)

        x0_1 = self.Up0_1(x, x1_0)
        x0_1 = self.Conv0_1(x0_1)
        x1_1 = self.Up1_1(x1_0, x2_0)
        x1_1 = self.Conv1_1(x1_1)
        x2_1 = self.Up2_1(x2_0, x3_0)
        x2_1 = self.Conv2_1(x2_1)
        x3_1 = self.Up3_1(x3_0, x4_0)
        x3_1 = self.Conv3_1(x3_1)

        x0_2 = self.Up0_2(x0_1, x1_1)
        x0_2 = self.Conv0_2(torch.cat([x0_2, x], dim=1))
        x1_2 = self.Up1_2(x1_1, x2_1)
        x1_2 = self.Conv1_2(torch.cat([x1_2, x1_0], dim=1))
        x2_2 = self.Up2_2(x2_1, x3_1)
        x2_2 = self.Conv2_2(torch.cat([x2_2, x2_0], dim=1))

        x0_3 = self.Up0_3(x0_2, x1_2)
        x0_3 = self.Conv0_3(torch.cat([x0_3, x0_1, x], dim=1))
        x1_3 = self.Up1_3(x1_2, x2_2)
        x1_3 = self.Conv1_3(torch.cat([x1_3, x1_1, x1_0], dim=1))

        x0_4 = self.Up0_4(x0_3, x1_3)
        x0_4 = self.Conv0_4(torch.cat([x0_4, x0_2, x0_1, x], dim=1))

        out_p = self.out(x0_4)

        if self.check is True:
            check1 = self.last_layer1(x0_3)
            check2 = self.last_layer1(x0_2)
            check3 = self.last_layer1(x0_1)
            if not isinstance(out_p, list):
                out = []
                out.append(out_p)
            else:
                out = out_p
            out.append(check1)
            out.append(check2)
            out.append(check3)
        else:
            out = out_p

        return out



