import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.transforms import ToTensor

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, net_type, num_in_channels=3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.net_type = net_type
        self.num_in_channels = num_in_channels
        self.num_classes = num_classes

        if ('normal' in self.net_type) or ('hybrid_nor' in self.net_type) or ('synergy_nor' in self.net_type) or ('synergy_all' in self.net_type):
           self.conv1_normal = nn.Conv2d(self.num_in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
           self.bn1_normal = nn.BatchNorm2d(64)
           self.layer1_normal = self._make_layer(block, 64, num_blocks[0], stride=1)
           self.layer2_normal = self._make_layer(block, 128, num_blocks[1], stride=2)
           self.layer3_normal = self._make_layer(block, 256, num_blocks[2], stride=2)
           self.layer4_normal = self._make_layer(block, 512, num_blocks[3], stride=2)

        if ('negative' in self.net_type) or ('hybrid_neg' in self.net_type) or ('synergy_neg' in self.net_type) or ('synergy_all' in self.net_type):
           self.conv1_negative = nn.Conv2d(self.num_in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
           self.bn1_negative = nn.BatchNorm2d(64)
           self.layer1_negative = self._make_layer(block, 64, num_blocks[0], stride=1)
           self.layer2_negative = self._make_layer(block, 128, num_blocks[1], stride=2)
           self.layer3_negative = self._make_layer(block, 256, num_blocks[2], stride=2)
           self.layer4_negative = self._make_layer(block, 512, num_blocks[3], stride=2)

        if ('normal' in self.net_type) or ('synergy_nor' in self.net_type) or ('synergy_all' in self.net_type):      
           self.linear_normal = nn.Linear(512*block.expansion, self.num_classes)
        if ('hybrid_nor' in self.net_type) or ('synergy_nor' in self.net_type) or ('synergy_all' in self.net_type):
           self.linear_normal_n = nn.Linear(512*block.expansion,self. num_classes)
        if ('hybrid_neg' in self.net_type) or ('synergy_neg' in self.net_type) or ('synergy_all' in self.net_type):
           self.linear_negative = nn.Linear(512*block.expansion, self.num_classes)
        if ('negative' in self.net_type) or ('synergy_neg' in self.net_type) or ('synergy_all' in self.net_type):
           self.linear_negative_n = nn.Linear(512*block.expansion, self.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        # conv block:

        if ('normal' in self.net_type) or ('hybrid_nor' in self.net_type) or ('synergy_nor' in self.net_type) or ('synergy_all' in self.net_type):
           out_normal = F.relu(self.bn1_normal(self.conv1_normal(x)))
           out_normal = self.layer1_normal(out_normal)
           out_normal = self.layer2_normal(out_normal)
           out_normal = self.layer3_normal(out_normal)
           out_normal = self.layer4_normal(out_normal)
           out_normal = F.adaptive_avg_pool2d(out_normal, (1, 1))
           out_normal = out_normal.view(out_normal.size(0), -1)

        if ('negative' in self.net_type) or ('hybrid_neg' in self.net_type) or ('synergy_neg' in self.net_type) or ('synergy_all' in self.net_type):
           out_negative = F.relu(self.bn1_negative(self.conv1_negative(x)))
           out_negative = self.layer1_negative(out_negative)
           out_negative = self.layer2_negative(out_negative)
           out_negative = self.layer3_negative(out_negative)
           out_negative = self.layer4_negative(out_negative)
           out_negative = F.adaptive_avg_pool2d(out_negative, (1, 1))
           out_negative = out_negative.view(out_negative.size(0), -1)

        # fc block:

        if 'synergy_nor' in self.net_type:
            out_normal_normal = self.linear_normal(out_normal)
            out_normal_negative = self.linear_normal_n(1 - out_normal)

            out = out_normal_normal + out_normal_negative
        
        else:
            if 'synergy_neg' in self.net_type:
                out_negative_normal = self.linear_negative(out_negative)
                out_negative_negative = self.linear_negative_n(1 - out_negative)

                out = out_negative_normal + out_negative_negative

            else:
                if 'synergy_all' in self.net_type:
                    out_normal_normal = self.linear_normal(out_normal)
                    out_normal_negative = self.linear_normal_n(1 - out_normal)
                    out_negative_normal = self.linear_negative(out_negative)
                    out_negative_negative = self.linear_negative_n(1 - out_negative)

                    out = out_normal_normal + out_normal_negative + out_negative_normal + out_negative_negative

                else:
                    if 'negative' in self.net_type:
                         out = 1 - out_negative
                         out = self.linear_negative_n(out)

                    else:
                         if 'normal' in self.net_type:
                             out = out_normal
                             out = self.linear_normal(out)

                         else:
                             if 'hybrid_neg' in self.net_type:
                                 out = out_negative
                                 out = self.linear_negative(out)

                             else:
                                 out = 1 - out_normal
                                 out = self.linear_normal_n(out)
                
        return out

    def freeze(self):
        msg = f'[{self.net_type}] Freezing: '
        for name, param in self.named_parameters(recurse=True):
            if not 'linear' in name:
                msg += f"{name} "
                param = param.requires_grad_(False)
                assert param.requires_grad == False
        print(msg)


    def get_conv_layers_normal(self):
        print(f"[{self.net_type}] Returning copies conv1, bn1, layers 1 to 4")
        return copy.deepcopy([self.conv1_normal, self.bn1_normal, self.layer1_normal, self.layer2_normal, self.layer3_normal, self.layer4_normal])

    def get_conv_layers_negative(self):
        print(f"[{self.net_type}] Returning copies conv1, bn1, layers 1 to 4")
        return copy.deepcopy([self.conv1_negative, self.bn1_negative, self.layer1_negative, self.layer2_negative, self.layer3_negative, self.layer4_negative])

    def set_conv_layers_normal(self, layers):
        print(f"[{self.net_type}] Setting conv1, bn1, layers 1 to 4")
        self.conv1_normal, self.bn1_normal, self.layer1_normal, self.layer2_normal, self.layer3_normal, self.layer4_normal = layers
   
    def set_conv_layers_negative(self, layers):
        print(f"[{self.net_type}] Setting conv1, bn1, layers 1 to 4")
        self.conv1_negative, self.bn1_negative, self.layer1_negative, self.layer2_negative, self.layer3_negative, self.layer4_negative = layers

    def set_linear_normal(self, linear):
        self.linear_normal = linear

    def set_linear_normal_n(self, linear):
        self.linear_normal_n = linear

    def set_linear_negative(self, linear_negative):
        self.linear_negative = linear_negative

    def set_linear_negative_n(self, linear_negative):
        self.linear_negative_n = linear_negative

    def get_net_type(self):
        return self.net_type

class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def ResNet18(net_type, num_in_channels, num_classes):
    return ResNet(BasicBlock, [2,2,2,2], net_type, num_in_channels, num_classes)

