import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F
from localconntect import LocallyConnected1d
import os
## Arcface , Resnet https://github.com/ronghuaiyang/arcface-pytorch

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.PReLU(),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNetFace(nn.Module):
    def __init__(self, block, layers, use_se=True):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNetFace, self).__init__()
        # self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.bn4 = nn.BatchNorm2d(512)
        # self.dropout = nn.Dropout()
        # self.fc5 = nn.Linear(512 * 16 * 16, 512)
        # self.bn5 = nn.BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.bn4(x)
        # x = self.dropout(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc5(x)
        # x = self.bn5(x)

        return x


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()

        # self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
        #                        bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # self.avgpool = nn.AvgPool2d(8, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc5 = nn.Linear(512 * 4 * 4, 512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = nn.AvgPool2d(kernel_size=x.size()[2:])(x)
        # x = self.avgpool(x)


        
        # x = x.view(x.size(0), -1)
        # x = self.fc5(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    return model


def resnet_face18(use_se=True, **kwargs):
    model = ResNetFace(IRBlock, [2, 2, 2, 2], use_se=use_se, **kwargs)
    return model




class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()

        # self.tree_connect_1 = nn.Sequential(
        #     LocallyConnected1d(512,512,1,9,1),
        #     nn.ReLU()
        # )

        # self.tree_connect_2 = nn.Sequential(
        #     LocallyConnected1d(512,512,1,8,1),
        #     nn.ReLU()
        # )
        # self.bn1 = nn.BatchNorm1d(512, affine=False)
        # self.bn2 = nn.BatchNorm1d(512, affine=False)
        # self.tree_connect_0 = nn.Sequential(
        #     LocallyConnected2d(512,128,16,1,1),
        #     nn.ReLU()
        # )

        self.tree_connect_1 = nn.Sequential(
            LocallyConnected1d(512,512,256,1,1),
            nn.ReLU()
        )

        self.tree_connect_2 = nn.Sequential(
            LocallyConnected1d(256,14,512,1,1),
            nn.ReLU()
        )

        # self.in1 = nn.InstanceNorm1d(32, affine=False)
        # self.in2 = nn.InstanceNorm1d(16, affine=False)

        # self.fc = nn.Linear(512, 512)
        self.in3 = nn.InstanceNorm1d(14, affine=True)


    def forward(self, input):

        input = input.view(input.size(0), input.size(1),  -1)
        output = self.tree_connect_1(input)
        # output = self.in1(output)
        output = output.permute(0,2,1)
        output = self.tree_connect_2(output)
        # output = self.in2(output)
        # output = F.normalize(output)
        # output = output.view(output.size(0), 1,  -1)
        # output = self.bn1(output)

        # output = self.fc(output)
        output = self.in3(output)
        

        # output = output.permute(0,2,1)
        # output = self.tree_connect_2(output)
        # output = self.bn2(output)

        return output


class Semantic_Fusion_Network(nn.Module):
    def __init__(self):
        super(Semantic_Fusion_Network, self).__init__()
        self.arcfacenet = resnet_face18()
        self.masknet = resnet18()
        self.fusion = ACE(512,512)
        self.regre = Regressor()

        self.encoder = Zencoder(9,512)
        # # self.local_connect_1 = LocallyConnected1d(512,512,1,9,1)
        # # self.local_connect_2 = LocallyConnected1d(512,512,1,5,1)
        # # self.local_connect_3 = LocallyConnected1d(512,512,1,5,1)
        # self.local_connect_4 = LocallyConnected1d(512,512,1,16,1)
        

    def forward(self, face_image, mask_image, style_image):

        face_feature = self.arcfacenet(face_image)
        mask_feature = self.masknet(mask_image)

        style_code = self.encoder(style_image,face_image)
        # fusion_result = self.fusion(face_feature,mask_feature,style_code)
        fusion_result = self.fusion(mask_feature,face_feature,style_code)


        # fusion_result = fusion_result.view(fusion_result.shape[0],fusion_result.shape[1],-1)
        # # fusion_result = self.local_connect_1(fusion_result)
        # # fusion_result = self.local_connect_2(fusion_result)
        # # fusion_result = self.local_connect_3(fusion_result)
        # direction = self.local_connect_4(fusion_result)

        direction = self.regre(fusion_result)
        # direction = torch.sigmoid(direction)

        return direction




class Zencoder(torch.nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=2, norm_layer=nn.InstanceNorm2d):
        super(Zencoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(1), nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0),
                 norm_layer(ngf), nn.LeakyReLU(0.2, False)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, False)]

        ### upsample
        for i in range(1):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.LeakyReLU(0.2, False)]

        model += [nn.ReflectionPad2d(1), nn.Conv2d(256, output_nc, kernel_size=3, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)


    def forward(self, input, segmap):

        codes = self.model(input)

        segmap = F.interpolate(segmap, size=codes.size()[2:], mode='nearest')

        # print(segmap.shape)
        # print(codes.shape)


        b_size = codes.shape[0]
        # h_size = codes.shape[2]
        # w_size = codes.shape[3]
        f_size = codes.shape[1]

        s_size = segmap.shape[1]

        codes_vector = torch.zeros((b_size, s_size, f_size), dtype=codes.dtype, device=codes.device)

        # xxxxx = segmap.bool()

        for i in range(b_size):
            for j in range(s_size):
                component_mask_area = torch.sum(segmap.bool()[i, j])

                if component_mask_area > 0:
                    codes_component_feature = codes[i].masked_select(segmap.bool()[i, j]).reshape(f_size,  component_mask_area).mean(1)
                    codes_vector[i][j] = codes_component_feature

                    # codes_avg[i].masked_scatter_(segmap.bool()[i, j], codes_component_mu)

        return codes_vector


class ACE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.Spade = SPADE(norm_nc, label_nc)
        self.style_length = 512
        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise_var = nn.Parameter(torch.zeros(norm_nc), requires_grad=True)


        ks = 5
        pw = ks // 2

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        self.create_gamma_beta_fc_layers()

        self.conv_gamma = nn.Conv2d(self.style_length, norm_nc, kernel_size=ks, padding=pw)
        self.conv_beta = nn.Conv2d(self.style_length, norm_nc, kernel_size=ks, padding=pw)




    def forward(self, x, segmap, style_codes):

        # added_noise = (torch.randn(x.shape[0], x.shape[3], x.shape[2], 1).cuda() * self.noise_var).transpose(1, 3)
        # normalized = self.param_free_norm(x + added_noise)

        normalized = self.param_free_norm(x)

        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')



        [b_size, f_size, h_size, w_size] = normalized.shape
        middle_avg = torch.zeros((b_size, self.style_length, h_size, w_size), device=normalized.device)

        # style_codes = style_codes.permute(0,2,1)

        for i in range(b_size):
            for j in range(style_codes.shape[1]):
                component_mask_area = torch.sum(segmap.bool()[i, j])

                if component_mask_area > 0:


                    middle_mu = F.relu(self.__getattr__('fc_mu' + str(j))(style_codes[i][j]))
                    component_mu = middle_mu.reshape(self.style_length, 1).expand(self.style_length, component_mask_area)

                    middle_avg[i].masked_scatter_(segmap.bool()[i, j], component_mu)


        gamma_avg = self.conv_gamma(middle_avg)
        beta_avg = self.conv_beta(middle_avg)


        gamma_spade, beta_spade = self.Spade(segmap)

        gamma_alpha = torch.sigmoid(self.blending_gamma)
        beta_alpha = torch.sigmoid(self.blending_beta)

        gamma_final = gamma_alpha * gamma_avg + (1 - gamma_alpha) * gamma_spade
        beta_final = beta_alpha * beta_avg + (1 - beta_alpha) * beta_spade
        out = normalized * (1 + gamma_final) + beta_final


        return out

    def create_gamma_beta_fc_layers(self):


        ###################  These codes should be replaced with torch.nn.ModuleList

        style_length = self.style_length

        self.fc_mu0 = nn.Linear(style_length, style_length)
        self.fc_mu1 = nn.Linear(style_length, style_length)
        self.fc_mu2 = nn.Linear(style_length, style_length)
        self.fc_mu3 = nn.Linear(style_length, style_length)
        self.fc_mu4 = nn.Linear(style_length, style_length)
        self.fc_mu5 = nn.Linear(style_length, style_length)
        self.fc_mu6 = nn.Linear(style_length, style_length)
        self.fc_mu7 = nn.Linear(style_length, style_length)
        self.fc_mu8 = nn.Linear(style_length, style_length)
        self.fc_mu9 = nn.Linear(style_length, style_length)
        self.fc_mu10 = nn.Linear(style_length, style_length)
        self.fc_mu11 = nn.Linear(style_length, style_length)
        self.fc_mu12 = nn.Linear(style_length, style_length)
        self.fc_mu13 = nn.Linear(style_length, style_length)
        self.fc_mu14 = nn.Linear(style_length, style_length)
        self.fc_mu15 = nn.Linear(style_length, style_length)
        self.fc_mu16 = nn.Linear(style_length, style_length)
        self.fc_mu17 = nn.Linear(style_length, style_length)
        self.fc_mu18 = nn.Linear(style_length, style_length)



class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()


        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 512

        ks = 5
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )

        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, segmap):

        inputmap = segmap

        actv = self.mlp_shared(inputmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return gamma, beta


