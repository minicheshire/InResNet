import torch
import torch.nn as nn
import torch.nn.functional as functional
import math
from torch.autograd import Variable

from random import random    

'''

Notes:
    Lines to modify when altering between In-ResNet and \lambda-In-ResNet:
        #263, #301, #334 
        
'''        

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock,self).__init__()
        self.bn1=nn.BatchNorm2d(in_planes)
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,padding=1,bias=False)
        self.relu=nn.ReLU(inplace=True) 
        self.stride=stride
        self.in_planes=in_planes
        self.planes=planes
    def forward(self,x):
        out=self.bn1(x)
        out=self.relu(out)
        out=self.conv1(out)
        out=self.bn2(out)
        out=self.relu(out)
        out=self.conv2(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__( self, in_planes, planes, stride=1):
        super(Bottleneck,self).__init__()
        self.bn1=nn.BatchNorm2d(in_planes)
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn3=nn.BatchNorm2d(planes)
        self.conv3=nn.Conv2d(planes,planes*4,kernel_size=1,bias=False)
        self.relu=nn.ReLU(inplace=True)

        self.in_planes = in_planes
        self.planes=planes

    def forward(self,x):

        out=self.bn1(x)
        out=self.relu(out)
        out=self.conv1(out)
        
        out=self.bn2(out)
        out=self.relu(out)
        out=self.conv2(out)
        
        out=self.bn3(out)
        out=self.relu(out)
        out=self.conv3(out)

        return out

class Downsample(nn.Module):
    def __init__(self,in_planes,out_planes,stride=2):
        super(Downsample,self).__init__()
        self.downsample=nn.Sequential(
                        nn.BatchNorm2d(in_planes),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_planes, out_planes,
                                kernel_size=1, stride=stride, bias=False)
                        )
    def forward(self,x):
        x=self.downsample(x)
        return x

class ResNet_N(nn.Module):

    def __init__(self,block,layers,noise_level=0.001,pretrain=True,num_classes=100):
        self.in_planes=16
        self.planes=[16,32,64]
        self.strides=[1,2,2]
        super(ResNet_N,self).__init__()
        self.noise_level=noise_level
        self.block=block
        self.conv1=nn.Conv2d(3,16,kernel_size=3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.relu=nn.ReLU(inplace=True)
        self.pretrain=pretrain
        
        blocks=[]
        for i in range(3):
            blocks.append(block(self.in_planes,self.planes[i],self.strides[i]))
            self.in_planes=self.planes[i]*block.expansion
            for j in range(1,layers[i]):
                blocks.append(block(self.in_planes,self.planes[i]))
        self.blocks=nn.ModuleList(blocks)
        self.downsample1=Downsample(16,64,stride=1)
        #self.downsample1=nn.Conv2d(16, 64,
        #                    kernel_size=1, stride=1, bias=False)
        self.downsample21=Downsample(16*block.expansion,32*block.expansion)
        self.downsample22=Downsample(16*block.expansion,32*block.expansion)
        self.downsample31=Downsample(32*block.expansion,64*block.expansion)
        self.downsample32=Downsample(32*block.expansion,64*block.expansion)

        self.bn=nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def change_state(self):
        self.pretrain=not self.pretrain

    def forward(self,x):
        x=self.conv1(x)
        #x=self.bn1(x)
        #x=self.relu(x)
        
        if self.block.expansion==4:
            residual=self.downsample1(x)
        else:
            residual=x
        
        x=self.blocks[0](x)+residual
        if self.training:
            x+=Variable(torch.FloatTensor(x.size()).cuda().normal_(0,self.noise_level),requires_grad=False) 
        for i,b in enumerate(self.blocks):
            if i==0:
                continue
            residual=x
            
            if b.in_planes != b.planes * b.expansion :
                if b.planes==32:
                    residual=self.downsample21(x)
                    
                elif b.planes==64:
                    residual=self.downsample31(x)
                    
            
            
            x=b(x)+residual               
           
            if self.training:
                x+=Variable(torch.FloatTensor(x.size()).cuda().uniform_(0,self.noise_level),requires_grad=False) 
            
        
        x=self.bn(x)
        x=self.relu(x)
        x=self.avgpool(x)
        x=x.view(x.size(0), -1)
        x=self.fc(x) 
        return x    
    

class ResNet(nn.Module):

    def __init__(self,block,layers,noise_level=0.001,pretrain=True,num_classes=10):
        self.in_planes=16
        self.planes=[16,32,64]
        self.strides=[1,2,2]
        super(ResNet,self).__init__()
        self.noise_level=noise_level
        self.block=block
        self.conv1=nn.Conv2d(3,16,kernel_size=3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.relu=nn.ReLU(inplace=True)
        self.pretrain=pretrain
        
        blocks=[]
        for i in range(3):
            blocks.append(block(self.in_planes,self.planes[i],self.strides[i]))
            self.in_planes=self.planes[i]*block.expansion
            for j in range(1,layers[i]):
                blocks.append(block(self.in_planes,self.planes[i]))
        self.blocks=nn.ModuleList(blocks)
        self.downsample1=Downsample(16,64,stride=1)
        #self.downsample1=nn.Conv2d(16, 64,
        #                    kernel_size=1, stride=1, bias=False)
        self.downsample21=Downsample(16*block.expansion,32*block.expansion)
        self.downsample22=Downsample(16*block.expansion,32*block.expansion)
        self.downsample31=Downsample(32*block.expansion,64*block.expansion)
        self.downsample32=Downsample(32*block.expansion,64*block.expansion)

        self.bn=nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def change_state(self):
        self.pretrain=not self.pretrain
    


    def forward(self,x):
        x=self.conv1(x)
        #x=self.bn1(x)
        #x=self.relu(x)
        
        if self.block.expansion==4:
            residual=self.downsample1(x)
        else:
            residual=x
        
        x=self.blocks[0](x)+residual
        for i,b in enumerate(self.blocks):
            if i==0:
                continue
            residual=x
            
            if b.in_planes != b.planes * b.expansion :
                if b.planes==32:
                    residual=self.downsample21(x)
                    
                elif b.planes==64:
                    residual=self.downsample31(x)
           
            x=b(x)+residual               
            
        
        x=self.bn(x)
        x=self.relu(x)
        x=self.avgpool(x)
        x=x.view(x.size(0), -1)
        x=self.fc(x) 
        return x    

class InResNet(nn.Module):

    def __init__(self,block,layers,pretrain=False,num_classes=10,stochastic_depth=False,PL=0.5,noise_level=0.001,noise=False):
        self.in_planes=16
        self.planes=[16,32,64]
        self.strides=[1,2,2]
        super(InResNet,self).__init__()
        self.noise=noise
        self.block=block
        self.conv1=nn.Conv2d(3,16,kernel_size=3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.relu=nn.ReLU(inplace=True)
        self.pretrain=pretrain
        self.ks=nn.ParameterList([nn.Parameter(torch.Tensor(1).uniform_(0.75, 0.8))for i in range(layers[0]+layers[1]+layers[2])])
#        self.ks=nn.ParameterList([nn.Parameter(torch.Tensor(1).uniform_(0.2, 0.25))for i in range(layers[0]+layers[1]+layers[2])]) # Use this line for \lambda-In-ResNet; for 164-layer experiments, use [0.8, 0.9] for In-ResNet or [0.1, 0.2] for \lambda-In-ResNet
        self.stochastic_depth=stochastic_depth
        blocks=[]
        n=layers[0]+layers[1]+layers[2]
        
        if not self.stochastic_depth:
            for i in range(3):
                blocks.append(block(self.in_planes,self.planes[i],self.strides[i]))
                self.in_planes=self.planes[i]*block.expansion
                for j in range(1,layers[i]):
                    blocks.append(block(self.in_planes,self.planes[i]))
        else:
            death_rates=[i/(n-1)*(1-PL) for i in range(n)]
            print(death_rates)
            for i in range(3):
                blocks.append(block(self.in_planes,self.planes[i],self.strides[i],death_rate=death_rates[i*layers[0]]))
                self.in_planes=self.planes[i]*block.expansion
                for j in range(1,layers[i]):
                    blocks.append(block(self.in_planes,self.planes[i],death_rate=death_rates[i*layers[0]+j]))
        self.blocks=nn.ModuleList(blocks)
        self.downsample1=Downsample(16,64,stride=1)
        self.downsample21=Downsample(16*block.expansion,32*block.expansion)
        self.downsample31=Downsample(32*block.expansion,64*block.expansion)

        self.bn=nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def print_ks(self):
        ha = []
        for i in self.ks: ha.append((functional.relu(1-i)).item()) #Change functional.relu(1-i) into functional.relu(i) for \lambda-In-ResNet experiments 
        return str(ha)
                
    def change_state(self):
        self.pretrain=not self.pretrain

    def forward(self,x):
        x=self.conv1(x)
        
        if self.block.expansion==4:
            residual=self.downsample1(x)
        else:
            residual=x
        x=self.blocks[0](x)+residual
        last_residual=residual
        for i,b in enumerate(self.blocks):
            if i==0:
                continue
            residual=x
                
            if b.in_planes != b.planes * b.expansion :
                if b.planes==32:
                    residual=self.downsample21(x)
                elif b.planes==64:
                    residual=self.downsample31(x)

                x=b(x)
                x+=residual
                
            elif self.pretrain:
                x=b(x)+residual                
            else:
                x=b(x)+(1-functional.relu(1-self.ks[i])).expand_as(residual)*residual
#                x=(1+functional.relu(self.ks[i])).expand_as(b(x))*b(x)+(1-functional.relu(self.ks[i])).expand_as(residual)*residual #Use this line for \lambda-In-ResNet
            
            last_residual=residual
        
        x=self.bn(x)
        x=self.relu(x)
        x=self.avgpool(x)
        x=x.view(x.size(0), -1)
        x=self.fc(x) 
        return x

def ResNet_20(**kwargs) :
    return ResNet(BasicBlock,[3,3,3],**kwargs)

def ResNet_32(**kwargs) :
    return ResNet(BasicBlock,[5,5,5],**kwargs)    

def ResNet_44(**kwargs) :
    return ResNet(BasicBlock,[7,7,7],**kwargs)

def ResNet_56(**kwargs) :
    return ResNet(BasicBlock,[9,9,9],**kwargs)

def ResNet_110(**kwargs) :
    return ResNet(BasicBlock,[18,18,18],**kwargs)    

def ResNet_164(**kwargs) :
    return ResNet(Bottleneck,[18,18,18],**kwargs)

def InResNet20(**kwargs) :
    return InResNet(BasicBlock,[3,3,3],**kwargs)

def InResNet32(**kwargs) :
    return InResNet(BasicBlock,[5,5,5],**kwargs)

def InResNet44(**kwargs) :
    return InResNet(BasicBlock,[7,7,7],**kwargs)

def InResNet56(**kwargs) :
    return InResNet(BasicBlock,[9,9,9],**kwargs)

def InResNet110(**kwargs) :
    return InResNet(BasicBlock,[18,18,18],**kwargs)

def InResNet164(**kwargs):
    return InResNet(Bottleneck,[18,18,18],**kwargs)

