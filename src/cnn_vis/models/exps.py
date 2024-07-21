from pathlib import Path

import torch
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt

from cnn_vis.archs.simple_cnn import SimpleCNN
from cnn_vis.archs.simple_decode_cnn import SimpleDecoderCNN

class Exp001:
    def __init__(self):
        self.arch = SimpleCNN()
        
    def load_data(self):
        DATASET_ROOT = Path("D:/Datasets/mnist")
        train_dset = torchvision.datasets.MNIST(
            root=DATASET_ROOT,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )
        
        test_dset = torchvision.datasets.MNIST(
            root=DATASET_ROOT,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )
        
        return train_dset, test_dset
        
    def train(self):
        
        optimizer = torch.optim.Adam(self.arch.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()
        recon_loss_fn = torch.nn.L1Loss()
        
        train_dset, test_dset = self.load_data()
        
        dloader = torch.utils.data.DataLoader(train_dset, batch_size=16, num_workers=4)
        test_dloader = torch.utils.data.DataLoader(test_dset, batch_size=16, num_workers=4)
        epochs = 2
        losses = []
        tr_acc_stats = [0, 0]
        for epoch in range(epochs):
            self.arch.train()
            for idx, (img, lbl) in enumerate(dloader):
                out = self.arch(img)
                loss = loss_fn(out, lbl)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                tr_acc_stats[0] += (torch.max(out, 1)[1] == lbl).sum().item()
                tr_acc_stats[1] += len(out)
                
                losses.append(loss.item())

                fig, ax = plt.subplots(1, 1)
                ax.plot(losses, label="class-loss")
                ax.legend()
                plt.savefig("output/loss_curve.png")
                plt.close()
                
                if idx % 100 == 0:
                    batch_accuracy = tr_acc_stats[0] / tr_acc_stats[1]
                    torch.save(self.arch.state_dict(), "output/exp001.pt")
                    out_str = f"Iter {idx}\n"
                    out_str += f"Loss: {loss.item():2f}\n"
                    out_str += f"Batch Accuracy: {batch_accuracy:.2%}\n"
                    print(out_str)
                    tr_acc_stats = [0, 0]
                    
            for idx, (img, lbl) in enumerate(test_dloader):
                self.arch.eval()
                acc_stats = [0, 0]
                with torch.no_grad():
                    out = self.arch(img)
                    acc_stats[0] += (torch.max(out, 1)[1] == lbl).sum().item()
                    acc_stats[1] += len(out)
            print(f"Testing Accuracy: {(acc_stats[0]/acc_stats[1]):.2%}")  

class Exp002:
    def __init__(self):
        self.arch = SimpleDecoderCNN()
        
    def load_data(self):
        DATASET_ROOT = Path("D:/Datasets/mnist")
        train_dset = torchvision.datasets.MNIST(
            root=DATASET_ROOT,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )
        
        test_dset = torchvision.datasets.MNIST(
            root=DATASET_ROOT,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )
        
        return train_dset, test_dset
        
    def train(self):
        
        optimizer = torch.optim.Adam(self.arch.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()
        recon_loss_fn = torch.nn.L1Loss()
        
        train_dset, test_dset = self.load_data()
        
        dloader = torch.utils.data.DataLoader(train_dset, batch_size=16, num_workers=4)
        test_dloader = torch.utils.data.DataLoader(test_dset, batch_size=16, num_workers=4)
        epochs = 2
        cls_losses = []
        recon_losses = []
        tr_acc_stats = [0, 0]
        for epoch in range(epochs):
            self.arch.train()
            for idx, (img, lbl) in enumerate(dloader):
                cls_out, recon, cpt = self.arch(img)
                cls_loss = loss_fn(cls_out, lbl)
                recon_loss = recon_loss_fn(img, recon)
                
                total_loss = cls_loss + recon_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                preds = torch.max(cls_out, 1)[1]
                tr_acc_stats[0] += (preds == lbl).sum().item()
                tr_acc_stats[1] += len(cls_out)
                
                cls_losses.append(cls_loss.item())
                recon_losses.append(recon_loss.item())

                fig, ax = plt.subplots(1, 1)
                ax.plot(cls_losses, label="class-loss")
                ax.plot(recon_losses, label="recon-loss")
                ax.legend()
                plt.savefig("output/loss_curve.png")
                plt.close()
                
                if idx % 50 == 0:
                    batch_accuracy = tr_acc_stats[0] / tr_acc_stats[1]
                    torch.save(self.arch.state_dict(), "output/exp002.pt")
                    out_str = f"Iter {idx}\n"
                    out_str += f"CLS Loss: {cls_loss.item():2f}\n"
                    out_str += f"RECON Loss: {recon_loss.item():2f}\n"
                    out_str += f"Batch Accuracy: {batch_accuracy:.2%}\n"
                    print(out_str)
                    tr_acc_stats = [0, 0]
                    
                    fig, axs = plt.subplots(1, 2)
                    axs[0].imshow(img[0, 0].detach(), cmap="gray")
                    axs[1].imshow(recon[0, 0].detach(), cmap="gray")
                    axs[0].set_title('Original')
                    axs[1].set_title(f'Recon: ({preds[0].item()})')
                    axs[0].axis('off')
                    axs[1].axis('off')
                    plt.savefig("output/train_recon.png")
                    plt.close()
                    
                    
            for idx, (img, lbl) in enumerate(test_dloader):
                self.arch.eval()
                acc_stats = [0, 0]
                with torch.no_grad():
                    cls_out, recon, cpt = self.arch(img)
                    acc_stats[0] += (torch.max(cls_out, 1)[1] == lbl).sum().item()
                    acc_stats[1] += len(cls_out)
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(img[0, 0].detach(), cmap="gray")
            axs[1].imshow(recon[0, 0].detach(), cmap="gray")
            axs[0].set_title('Original')
            axs[1].set_title('Recon')
            axs[0].axis('off')
            axs[1].axis('off')
            plt.savefig("output/test_recon.png")
            plt.close()
            print(f"Testing Accuracy: {(acc_stats[0]/acc_stats[1]):.2%}")  