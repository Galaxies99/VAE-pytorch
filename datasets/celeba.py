from torchvision.datasets import CelebA
from torchvision import transforms

def CelebADataset(root, split, img_size, center_crop, download = False):
    return CelebA(
        root = root,
        split = split,
        transform = transform(img_size, center_crop),
        download = download
    )

def transform(img_size, center_crop):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(center_crop),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda X: 2 * X - 1.)
    ])
