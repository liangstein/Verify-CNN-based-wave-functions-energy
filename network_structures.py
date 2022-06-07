import torch
import torch.nn as nn
L=12
site_number=int(L**2)

class Net_J1J2(nn.Module):
    def __init__(self):
        super(Net_J1J2,self).__init__()
        self.conv1=nn.Conv2d(1,64,5,padding=0)
        self.deconv1 = nn.ConvTranspose1d(64, 64, 2, stride=2, padding=0)
        self.conv2=nn.Conv2d(64,32,5,padding=0)
        self.deconv2 = nn.ConvTranspose1d(32, 32, 2, stride=2, padding=0)
        self.conv3=nn.Conv2d(32,32,3,padding=0)
        self.deconv3 = nn.ConvTranspose1d(32, 32, 2, stride=2, padding=0)
        self.conv4=nn.Conv2d(32,32,3,padding=0)
        self.deconv4 = nn.ConvTranspose1d(32, 32, 2, stride=2, padding=0)
        self.conv5=nn.Conv2d(32,32,3,padding=0)
        self.deconv5 = nn.ConvTranspose1d(32, 32, 2, stride=2, padding=0)
        self.conv6=nn.Conv2d(32,32,3,padding=0)
        self.deconv6 = nn.ConvTranspose1d(32, 1, 2, stride=2, padding=0)
        self.maxpool1=nn.MaxPool1d(2)
    def nn(self, x1):
        x1 = self.conv1(x1)
        x1 = x1.view(-1, 64, site_number)
        x1 = self.maxpool1(x1)
        x1 = self.deconv1(x1)
        x1 = x1.view(-1, 64, L, L)
        x1 = x1.repeat((1, 1, 3, 3))[:, :, L - 2:2 * L + 2, L - 2:2 * L + 2]
        x1 = self.conv2(x1)
        x1 = x1.view(-1, 32, site_number)
        x1 = self.maxpool1(x1)
        x1 = self.deconv2(x1)
        x1 = x1.view(-1, 32, L, L)
        x1 = x1.repeat((1, 1, 3, 3))[:, :, L - 1:2 * L + 1, L - 1:2 * L + 1]
        x1 = self.conv3(x1)
        x1 = x1.view(-1, 32, site_number)
        x1 = self.maxpool1(x1)
        x1 = self.deconv3(x1)
        x1 = x1.view(-1, 32, L, L)
        x1 = x1.repeat((1, 1, 3, 3))[:, :, L - 1:2 * L + 1, L - 1:2 * L + 1]
        x1 = self.conv4(x1)
        x1 = x1.view(-1, 32, site_number)
        x1 = self.maxpool1(x1)
        x1 = self.deconv4(x1)
        x1 = x1.view(-1, 32, L, L)
        x1 = x1.repeat((1, 1, 3, 3))[:, :, L - 1:2 * L + 1, L - 1:2 * L + 1]
        x1 = self.conv5(x1)
        x1 = x1.view(-1, 32, site_number)
        x1 = self.maxpool1(x1)
        x1 = self.deconv5(x1)
        x1 = x1.view(-1, 32, L, L)
        x1 = x1.repeat((1, 1, 3, 3))[:, :, L - 1:2 * L + 1, L - 1:2 * L + 1]
        x1 = self.conv6(x1)
        x1 = x1.view(-1, 32, site_number)
        x1 = self.maxpool1(x1)
        x1 = self.deconv6(x1)
        x1 = x1.view(-1, int(L ** 2))*10
        x1 = torch.prod(x1,1)
        return x1
    def forward(self,x1,x2,x3,x4):
        y = self.nn(x1)+self.nn(x2)+self.nn(x3)+self.nn(x4)
        return y

class Net_tJ(nn.Module):
    def __init__(self):
        super(Net_tJ, self).__init__()
        self.embed = nn.Embedding(3,2)#last number is embedding size
        self.conv1 = nn.Conv2d(2, 192, 5, padding=0)
        self.conv2 = nn.Conv1d(192, 48, 5, padding=0)
        self.conv3 = nn.Conv1d(48, 48, 5, padding=0)
        self.conv4 = nn.Conv1d(48, 48, 5, padding=0)
        self.conv5 = nn.Conv1d(48, 48, 5, padding=0)
        self.conv6 = nn.Conv1d(48, 48, 5, padding=0)
        self.conv7 = nn.Conv1d(48, 48, 5, padding=0)
        #self.deconv = nn.ConvTranspose1d(128, 1, 2, stride=2, padding=0)
        self.maxpool = nn.MaxPool1d(2,stride=1)
        self.conv_final = nn.Conv1d(48, 1, 1, padding=0)
    def nn(self,x):
        #x shape is [B,L+K-1,L+K-1]
        bs=len(x)
        x=self.embed(x)#[B,L+K-1,L+K-1,E]
        x=x.permute(0,3,1,2)#[B,E,L+K-1,L+K-1]
        x=self.conv1(x) #[B,C,L,L]
        x=x.view(bs,192,-1) #[B,C,L^2]
        x=self.maxpool(x) #[B,C,L^2-1]
        x=x.repeat((1,1,2))[:,:,:site_number+4]
        x=self.conv2(x) #[B,C,L^2]
        x=self.maxpool(x) #[B,C,L^2-1]
        x=x.repeat((1,1,2))[:,:,:site_number+4]
        x=self.conv3(x) #[B,C,L^2]
        x=self.maxpool(x) #[B,C,L^2-1]
        x=x.repeat((1,1,2))[:,:,:site_number+4]
        x=self.conv4(x) #[B,1,L^2]
        x=self.maxpool(x) #[B,C,L^2-1]
        x=x.repeat((1,1,2))[:,:,:site_number+4]
        x=self.conv5(x) #[B,C,L^2]
        x=self.maxpool(x) #[B,C,L^2-1]
        x=x.repeat((1,1,2))[:,:,:site_number+4]
        x=self.conv6(x) #[B,C,L^2]
        x=self.maxpool(x) #[B,C,L^2-1]
        x=x.repeat((1,1,2))[:,:,:site_number+4]
        x=self.conv7(x) #[B,C,L^2]
        x=self.maxpool(x) #[B,C,L^2-1]
        x=x.repeat((1,1,2))[:,:,:site_number]
        x=self.conv_final(x) #[B,1,L^2]
        x=x.view(bs,-1)#*10
        x=torch.prod(x,1)
        return x
    def forward(self, x):
        x=self.nn(x)
        return x
