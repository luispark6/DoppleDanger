import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200']
using_ckpt = False


class BlurPool(nn.Module):
    """Anti-aliasing blur pooling to prevent checkerboard artifacts"""
    def __init__(self, channels, stride=2, kernel_size=3):
        super(BlurPool, self).__init__()
        self.stride = stride
        self.channels = channels
        
        if stride == 1:
            self.blur = nn.Identity()
        else:
            # Create anti-aliasing blur kernel
            if kernel_size == 3:
                kernel = torch.tensor([1, 2, 1], dtype=torch.float32)
            elif kernel_size == 5:
                kernel = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
            else:  # Default to 3x3
                kernel = torch.tensor([1, 2, 1], dtype=torch.float32)
            
            # Make it 2D
            kernel = kernel[None, None, :] * kernel[None, :, None]
            kernel = kernel / kernel.sum()
            
            # Repeat for all channels (depthwise)
            kernel = kernel.repeat(channels, 1, 1, 1)
            
            # Register as buffer so it moves with model to GPU
            self.register_buffer('kernel', kernel)
            
            # Create the blur layer
            padding = kernel_size // 2
            self.blur = nn.Conv2d(channels, channels, kernel_size, 
                                stride=stride, padding=padding, 
                                groups=channels, bias=False)
            
            # Set the weights and freeze them
            self.blur.weight.data = kernel
            self.blur.weight.requires_grad = False

    def forward(self, x):
        return self.blur(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=True,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=True)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        
        # Use BlurPool for strided convolutions to prevent checkerboard
        if stride > 1:
            self.conv2 = conv3x3(planes, planes, stride=1)  # Remove stride from conv
            self.blur_pool = BlurPool(planes, stride=stride)
        else:
            self.conv2 = conv3x3(planes, planes, stride=stride)
            self.blur_pool = None
            
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward_impl(self, x):
        identity = x
        
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        
        # Apply blur pooling if needed
        if self.blur_pool is not None:
            out = self.blur_pool(out)
            
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        return out        

    def forward(self, x):
        if self.training and using_ckpt:
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)


class IResNet(nn.Module):
    fc_scale = 14 * 14
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, 
                 fp16=False, use_blur_pool=True):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.use_blur_pool = use_blur_pool
        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.groups = groups
        self.base_width = width_per_group
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.use_blur_pool and stride > 1:
                # Anti-aliased downsample: learn features first, then blur+downsample
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride=1),  # No stride
                    nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
                    BlurPool(planes * block.expansion, stride=stride)  # Anti-aliased downsampling
                )
            else:
                # Traditional downsample
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
                )
        
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        
        x = self.fc(x.float() if self.fp16 else x)
        # x = self.features(x)  # Uncomment if you need feature normalization
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)
