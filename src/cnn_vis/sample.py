from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision import transforms

import torch.nn.functional as F

import matplotlib.pyplot as plt

from cnn_vis.archs.simple_decode_cnn import SimpleDecodeCNN

def run():
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

    n_convs = 10
    model = SimpleDecodeCNN(n_convs, n_h_layers=2, n_concept=50)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    recon_loss_fn = torch.nn.L1Loss()
    dloader = torch.utils.data.DataLoader(train_dset, batch_size=16, num_workers=4)
    epochs = 2
    losses = []
    recon_losses = []
    inter_recon_losses = []
    cpt_losses = []
    for epoch in range(epochs):
        for idx, (img, lbl) in enumerate(dloader):
            to_pil = torchvision.transforms.ToPILImage()
            conv_out, out, recon, cpt, encodes, decodes = model(img)
            loss = loss_fn(out, lbl)
            recon_loss = recon_loss_fn(img, recon)
            inter_recon_loss = torch.tensor(0)
            #for enc, dec in zip(encodes, decodes):
            #    inter_recon_loss += recon_loss_fn(enc, dec)
            
            total_loss = loss + recon_loss + inter_recon_loss
            # print(out)
            # print(lbl)

            weights = model.conv1.weight
            fig, axs = plt.subplots(2, n_convs + 1)
            axs[0, 0].imshow(img[0, 0], cmap="gray")
            axs[1, 0].imshow(recon[0, 0].detach().numpy(), cmap="gray")
            axs[0, 0].axis("off")
            axs[1, 0].axis("off")
            for i in range(n_convs):
                # print(conv_out.shape)
                axs[0, i + 1].imshow(to_pil(conv_out[0, i]), cmap="gray")
                axs[1, i + 1].imshow(to_pil(weights[i, 0]), cmap="gray")
                axs[0, i + 1].axis("off")
                axs[1, i + 1].axis("off")

            #plt.tight_layout()
            plt.savefig("test.png")
            plt.close()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            
            conv_out, out, recon, cpt_recon, encodes, decodes = model(recon.detach())
            
            cpt_loss = recon_loss_fn(cpt.detach(), cpt_recon)
            optimizer.zero_grad()
            cpt_loss.backward()
            optimizer.step()

            out_str = f"Iter {idx}\n"
            out_str += f"Loss: {loss.item():2f}\n"
            out_str += f"Recon Loss: {recon_loss.item():2f}\n"
            out_str += f"Recon inter Loss: {inter_recon_loss.item():2f}\n"
            out_str += f"Cpt Loss: {cpt_loss.item():2f}\n"
            print(out_str)
            #print(f"Iter {idx} | Loss: {loss.item():2f} | Recon Loss: {recon_loss.item():2f}")
            losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            inter_recon_losses.append(inter_recon_loss.item())
            cpt_losses.append(cpt_loss.item())

            fig, ax = plt.subplots(1, 1)
            ax.plot(losses, label="cls loss")
            ax.plot(recon_losses, label="recon loss")
            ax.plot(inter_recon_losses, label="inter_recon_loss")
            ax.plot(cpt_losses, label="cpt_loss")
            ax.legend()
            plt.savefig("loss_curve.png")
            plt.close()
            
            if idx % 100 == 0:
                torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    run()
