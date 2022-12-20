import torch
import torch.nn as nn

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.conv1=nn.Conv2d(48, 512, 11, stride=4, padding=1)  # b, 16, 10, 10
        self.act1=nn.Tanh()
        self.p1=nn.MaxPool2d(2,stride=2,return_indices=True)
        self.conv2=nn.Conv2d(512, 256, 3, stride=2,padding=1)
        self.act2=nn.Tanh()
        self.p2 = nn.MaxPool2d(2, stride=2,return_indices=True)
        self.conv3=nn.Conv2d(256,128,3,stride=1,padding=1)
        self.act3=nn.Tanh()

        self.d_conv1=nn.ConvTranspose2d(128, 256, 2, stride=2) # b, 16, 5, 5
        self.d_act1=nn.Tanh()
        self.d_conv2 = nn.ConvTranspose2d(256, 512, 4, stride=4)  # b, 16, 5, 5
        self.d_act2 = nn.Tanh()
        self.d_conv3 = nn.ConvTranspose2d(512, 48, 4, stride=4,padding=1)  # b, 16, 5, 5
        self.d_act3 = nn.Tanh()
        self.d_conv4 = nn.ConvTranspose2d(48, 48, 11, stride=1)  # b, 16, 5, 5
        self.d_act4 = nn.Tanh()


    def forward(self, x):
        x=self.conv1(x)
        print('conv1:',x.shape)
        x=self.act1(x)
        outsize1=x.size()
        x,indices1=self.p1(x)
        print("p1:",x.shape)
        x = self.act2(self.conv2(x))
        print("conv2:",x.shape)
        outsize2=x.size()
        x,indices2 = self.p2(x)
        print("p2:",x.shape)
        x = self.act3(self.conv3(x))
        print("conv3:",x.shape)

        x = self.d_act1(self.d_conv1(x))
        print("deconv1:",x.shape)
        # print((indices2.shape))
        # x = self.d_p1(x,indices2,output_size=outsize2)
        # print("dp1:",x.shape)
        x = self.d_act2(self.d_conv2(x))
        print("deconv2:",x.shape)
        # x = self.d_p2(x,indices1,output_size=outsize1)
        # print('dp2:',x.shape)
        # print('indice2:',indices1.shape)
        x = self.d_act3(self.d_conv3(x))
        x = self.d_act4(self.d_conv4(x))
        print(x.shape)

model=autoencoder()
model(torch.zeros(1,48,520,360))
