import torch
import torch.nn as nn

from cnn_vis.archs.simple_cnn import SimpleCNN

class SimpleDecoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = SimpleCNN()
        self.num_cpt = 84
        
        self.cpt_to_low_res_img = nn.Linear(self.num_cpt, 7*7)
        self.rconv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.rconv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.rconv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
        self.up1 = nn.Upsample(14)
        self.up2 = nn.Upsample(28)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.classifier.forward_1(x)
        cpt = self.sigmoid(out)
        cls_out = self.classifier.forward_2(cpt)
        
        recon = self.cpt_to_low_res_img(cpt).view(-1, 1, 7, 7)
        recon = self.rconv1(recon)
        recon = self.up1(recon)
        recon = self.rconv2(recon)
        recon = self.up2(recon)
        recon = self.rconv3(recon)
        
        return cls_out, recon, cpt
        
        
        
        

class SimpleDecodeCNNOrg(nn.Module):
    def __init__(self, n_convs=16, n_h_layers=2, avg_pool_dim=7, n_concept=7):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.cpt_linear = nn.Linear(64 * 7 * 7, n_concept)
        self.cls_linear = nn.Linear(n_concept, 10)
        
        self.cpt_to_low_res_img = nn.Linear(n_concept, 7*7)
        self.rconv1 = nn.Conv2d(1, n_convs, kernel_size=3, padding=1)
        self.rconv2 = nn.Conv2d(n_convs, n_convs, kernel_size=3, padding=1)
        self.rconv3 = nn.Conv2d(n_convs, 1, kernel_size=3, padding=1)
        
        self.up1 = nn.Upsample(14)
        self.up2 = nn.Upsample(28)
        
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        encodes = []
        decodes = []
        
        conv_out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(conv_out))
        out = self.max_pool(out)
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.max_pool(out)
        
        encodes.append(out)
        
        cpt = self.sigmoid(self.cpt_linear(torch.flatten(out, 1)))
        out = self.cls_linear(cpt)
        
        recon = self.cpt_to_low_res_img(cpt).view(-1, 1, 7, 7)
        recon = self.rconv1(recon)
        decodes.append(recon)
        recon = self.up1(recon)
        recon = self.rconv2(recon)
        recon = self.up2(recon)
        recon = self.rconv3(recon)
        
        return conv_out, out, recon, cpt, encodes, decodes