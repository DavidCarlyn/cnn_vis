from pathlib import Path

import torch
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt

from cnn_vis.archs.simple_decode_cnn import SimpleDecodeCNN

def infer(model_path):
    model = SimpleDecodeCNN(10, n_h_layers=2, n_concept=50)
    model.load_state_dict(torch.load(model_path))
    
    DATASET_ROOT = Path("D:/Datasets/mnist")
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
    
    to_pil = transforms.ToPILImage()
    references = {}
    for idx, (img, lbl) in enumerate(test_dset):
        if lbl in references: continue
        conv_out, out, recon, cpt, _, _ = model(img.unsqueeze(0))
        
        _, pred = torch.max(out, 1)
        references[lbl] = (img[0], recon[0, 0].detach().numpy(), pred[0].item())
        
    fig, axs = plt.subplots(10, 2, figsize=(6, 18))
    for i in range(10):
        img, recon, pred = references[i]
        axs[i, 0].imshow(img, cmap='gray')
        axs[i, 1].imshow(recon, cmap='gray')
        axs[i, 1].set_title(f"GT: {i} | Pred: {pred}")
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')
    plt.savefig("infer.png")

if __name__ == "__main__":
    model_path = Path(r"C:/Users/David Carlyn/Code/cnn_vis/model.pt")
    infer(model_path)