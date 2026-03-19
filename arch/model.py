import numpy as np
import torch
from torch import nn
import torchvision
from arch.VariationalBottleneck import VariationalBottleneck
from arch.BayesianLayer import BayesianLinear
import torch.nn.functional as F

class VBMLP(nn.Module):
    def __init__(self, num_classes, data_shape, width=1024,):
        super().__init__()
        self.width = width
        self.num_classes = num_classes
        self.data_shape = data_shape

        self.flat = nn.Flatten()
        self.l1 = nn.Linear(np.prod(self.data_shape), width)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, self.num_classes)

        self.flagVB = 0

        self.vb = VariationalBottleneck(in_shape=(width,))
        self.eps = torch.randn(size=(1, 256))
        self.learned_eps = torch.nn.Parameter(torch.randn(size=(1, 256)))

    def forward(self, x):
        x = self.flat(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)

        # client randomly samples a feature
        if self.flagVB == 0:
            eps = None
        # the attacker has no knowledge of sampled feature, use a randomly generated one instead
        elif self.flagVB == 1:
            eps = self.eps
        # our proposed attack uses the jointly-optimized one instead
        else:
            eps = self.learned_eps

        x = self.vb(x, eps)
        x = self.l3(x)
        return x

    def loss(self):
        return self.vb.loss()

    def resample(self):
        self.eps = torch.randn(size=(1, 256))
        self.learned_eps.copy_(torch.randn(size=(1, 256)))

class VBResNet18(torchvision.models.ResNet):
    def __init__(self, block=torchvision.models.resnet.BasicBlock, layers=[2,2,2,2], num_classes=10, 
                 zero_init_residual=False, groups=1, base_width=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # nn.Module
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (VariationalBottleneck)):
                nn.init.kaiming_normal_(m.weight)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, torchvision.models.resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, torchvision.models.resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.flagVB = 0
        self.vb = VariationalBottleneck(in_shape=(512 * block.expansion,))
        self.eps = torch.randn(size=(1, 256))
        self.learned_eps = torch.nn.Parameter(torch.randn(size=(1, 256)))
        # self.multiplier = torch.nn.Parameter(torch.randn(512 * block.expansion,))
        
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        # x_ext = x
        # x_cor = torch.mul(self.multiplier, x_ext)
        # x = x_cor

        if self.flagVB == 0:
            eps = None
        elif self.flagVB == 1:
            eps = self.eps
        elif self.flagVB == 2:
            eps = self.learned_eps
        x = self.vb(x,eps)

        x = self.fc(x)
        return x
    
    def forward(self, x):
        return self._forward_impl(x) 
    
    def loss(self):
        # return 0
        return self.vb.loss()
    
    def resample(self):
        self.eps = torch.randn(size=(1, 256))
        self.learned_eps.copy_(torch.randn(size=(1, 256)))

class VBResNet34(torchvision.models.ResNet):
    def __init__(self, block=torchvision.models.resnet.BasicBlock, layers=[3,4,6,3], num_classes=10, 
                 zero_init_residual=False, groups=1, base_width=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # nn.Module
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (VariationalBottleneck)):
                nn.init.kaiming_normal_(m.weight)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, torchvision.models.resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, torchvision.models.resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.flagVB = 0
        self.vb = VariationalBottleneck(in_shape=(512 * block.expansion,))
        self.eps = torch.randn(size=(1, 256))
        self.learned_eps = torch.nn.Parameter(torch.randn(size=(1, 256)))
        # self.multiplier = torch.nn.Parameter(torch.randn(512 * block.expansion,))
        
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        # x_ext = x
        # x_cor = torch.mul(self.multiplier, x_ext)
        # x = x_cor

        if self.flagVB == 0:
            eps = None
        elif self.flagVB == 1:
            eps = self.eps
        elif self.flagVB == 2:
            eps = self.learned_eps
        x = self.vb(x,eps)

        x = self.fc(x)
        return x
    
    def forward(self, x):
        return self._forward_impl(x) 
    
    def loss(self):
        # return 0
        return self.vb.loss()
    
    def resample(self):
        self.eps = torch.randn(size=(1, 256))
        self.learned_eps.copy_(torch.randn(size=(1, 256)))

class BLConvNet4(nn.Module):

    def __init__(self, num_channels=3, num_classes=10):
        super(BLConvNet4, self).__init__()
    
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
                
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.fc1 = nn.Linear(512*16*16, 256) # for 256
        self.fc1 = nn.Linear(663552, 256) # for 576
        # self.fc1 = nn.Linear(663552,256)
        self.fc = nn.Linear(256, num_classes)

        self.bl_channels = 256
        self.flagVB = 0
        self.bl = BayesianLinear(self.bl_channels, self.bl_channels)
        self.eps = torch.randn(size=(self.bl_channels,self.bl_channels))
        self.epsb = torch.randn(size=(1,self.bl_channels))
        self.learned_eps = torch.nn.Parameter(torch.randn(size=(self.bl_channels,self.bl_channels)))
        self.learned_epsb = torch.nn.Parameter(torch.randn(size=(1,self.bl_channels)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.maxpool(x)
        
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.maxpool(x)
        
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.maxpool(x)
        
        x = x.view(x.size(0),-1)
        
        x = self.fc1(x)
                
        if self.flagVB == 0:
            eps = epsb = None
        elif self.flagVB == 1:
            eps = self.eps
            epsb = self.epsb
        elif self.flagVB == 2:
            eps = self.learned_eps
            epsb = self.learned_epsb
        x = self.bl(x,eps,epsb)

        x = self.fc(x)
        return x
    
    def loss(self):
        return self.bl.loss()
    
    def resample(self):
        self.eps = torch.randn(size=(self.bl_channels,self.bl_channels))
        self.epsb = torch.randn(size=(1,self.bl_channels))
        self.learned_eps = torch.nn.Parameter(torch.randn(size=(self.bl_channels,self.bl_channels)))
        self.learned_epsb = torch.nn.Parameter(torch.randn(size=(1,self.bl_channels)))

class BLConvNet8(nn.Module):
    """ConvNetBN."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super().__init__()

        self.conv1 = torch.nn.Conv2d(num_channels, width, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(width)

        self.conv2 = torch.nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(width)
            
        self.pool1 = torch.nn.MaxPool2d(2, stride=2)
        
        self.conv3 = torch.nn.Conv2d(width, 2*width, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(2*width)
            
        self.conv4 = torch.nn.Conv2d(2*width, 2*width, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(2*width)
            
        self.pool2 = torch.nn.MaxPool2d(2, stride=2) 
            
        self.conv5 = torch.nn.Conv2d(2*width, 4*width, kernel_size=3, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(4*width)

        self.conv6 = torch.nn.Conv2d(4*width, 4*width, kernel_size=3, padding=1)
        self.bn6 = torch.nn.BatchNorm2d(4*width)
            
        self.pool3 = torch.nn.MaxPool2d(2, stride=2)
            
        self.conv7 = torch.nn.Conv2d(4*width, 8*width, kernel_size=3, padding=1)
        self.bn7 = torch.nn.BatchNorm2d(8*width)

        self.conv8 = torch.nn.Conv2d(8*width, 8*width, kernel_size=3, padding=1)
        self.bn8 = torch.nn.BatchNorm2d(8*width)
            
        self.pool4 = torch.nn.MaxPool2d(2, stride=2)
            
        self.flatten = torch.nn.Flatten()
        
        self.fc = nn.Linear(256*8*width, 8*width)
        self.fc = nn.Linear(331776, 8*width)

        self.bl_channels = 8*width
        self.num_classes = num_classes
        self.flagVB = 0
        self.eps = torch.randn(size=(num_classes,self.bl_channels))
        self.epsb = torch.randn(size=(1,num_classes))
        self.learned_eps = torch.nn.Parameter(torch.randn(size=(num_classes,self.bl_channels)))
        self.learned_epsb = torch.nn.Parameter(torch.randn(size=(1,num_classes)))

        self.bl = BayesianLinear(8*width, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool1(x)
        
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool2(x)

        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.conv6(x)
        x = F.relu(self.bn6(x))
        x = self.pool3(x)
        
        x = self.conv7(x)
        x = F.relu(self.bn7(x))
        x = self.conv8(x)
        x = F.relu(self.bn8(x))
        x = self.pool4(x)

        x = self.flatten(x)               
        x = F.relu(self.fc(x))        
        
        if self.flagVB == 0:
            eps = None
            epsb = None
        elif self.flagVB == 1:
            eps = self.eps
            epsb = self.epsb
        elif self.flagVB == 2:
            eps = self.learned_eps
            epsb = self.learned_epsb
        x = self.bl(x,eps,epsb)

        return x
    
    def loss(self):
        return self.bl.loss()
    
    def resample(self):
        self.eps = torch.randn(size=(self.num_classes,self.bl_channels))
        self.epsb = torch.randn(size=(1,self.num_classes))
        self.learned_eps = torch.nn.Parameter(torch.randn(size=(self.num_classes,self.bl_channels)))
        self.learned_epsb = torch.nn.Parameter(torch.randn(size=(1,self.num_classes)))

def convert_relu_to_sigmoid(model):
    for child_name, child in model.named_children(): 
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.Sigmoid())
        else:
            convert_relu_to_sigmoid(child)
