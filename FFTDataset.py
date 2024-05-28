from math import sqrt
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
import random
from PIL import ImageDraw
from utils.digits_process_dataset import load_mnist


def get_post_transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
 
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=[0.6,1.3], contrast=[0.5,1.5], saturation=[0.5,1.5], hue=[-0.02,0.02]),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return img_transform

def get_pre_transform(image_size=224, crop=False, jitter=0):
    if crop:
        img_transform = [transforms.RandomResizedCrop(image_size, scale=[0.8, 1.0])]
    else:
        img_transform = [transforms.Resize((image_size, image_size))]
    img_transform += [transforms.RandomHorizontalFlip(), lambda x: np.asarray(x)]
    img_transform = transforms.Compose(img_transform)
    return img_transform


def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0): 
    lam = np.random.uniform(0, alpha)

    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12

class FourierDGDataset(Dataset):
    def __init__(self, names, labels, transformer=None, from_domain=None, alpha=1.0):
        
        self.labels = labels 
        self.transformer = transformer 
        self.post_transform = get_post_transform() 
        self.from_domain = from_domain 
        self.alpha = alpha 
        
        self.flat_names = []
        self.flat_labels = [] 
        self.flat_domains = [] 
        for i in range(len(names)):
            self.flat_names += names[i]
            self.flat_labels += labels[i]
            self.flat_domains += [i] * len(names[i])
        assert len(self.flat_names) == len(self.flat_labels) 
        assert len(self.flat_names) == len(self.flat_domains) 

    def __len__(self):
        return len(self.flat_names)

    def __getitem__(self, index): 
        img_name = self.flat_names[index]
        label = self.flat_labels[index]
        domain = self.flat_domains[index]
        img = Image.open(img_name).convert('RGB')
        img_o = self.transformer(img) 

        img_s, label_s, domain_s = self.sample_image(domain) 
        img_s2o, img_o2s = colorful_spectrum_mix(img_o, img_s, alpha=self.alpha)
        img_o, img_s = self.post_transform(img_o), self.post_transform(img_s)
        img_s2o, img_o2s = self.post_transform(img_s2o), self.post_transform(img_o2s)
        img = [img_o, img_s, img_s2o, img_o2s]
        label = [label, label_s, label, label_s]
        domain = [domain, domain_s, domain, domain_s]
        return img, label, domain

    def sample_image(self, domain): 
        if self.from_domain == 'all': 
            domain_idx = random.randint(0, len(self.names)-1)
        elif self.from_domain == 'inter': 
            domains = list(range(len(self.names)))
            domains.remove(domain)
            domain_idx = random.sample(domains, 1)[0]
        elif self.from_domain == 'intra': 
            domain_idx = domain
        else:
            raise ValueError("Not implemented")
        img_idx = random.randint(0, len(self.names[domain_idx])-1)
        imgn_ame_sampled = self.names[domain_idx][img_idx]
        img_sampled = Image.open(imgn_ame_sampled).convert('RGB')
        label_sampled = self.labels[domain_idx][img_idx]
        return self.transformer(img_sampled), label_sampled, domain_idx


def createBackgroundPic(width=256,height=256):  
    image = Image.new('RGB', (width, height), 'black')
    
    draw = ImageDraw.Draw(image)

    for i in range(0, width):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        line_width = random.randint(1, width)
        draw.line((i, 0, i, height),width=line_width, fill=color)
        i=i+line_width

    p_num = random.randint(0, 100)
    for i in range(p_num):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        draw.point((x, y), (255, 255, 255))
    
    for i in range(20):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.ellipse([x, y, x + 4, y + 4], fill=color)

    for i in range(3):
        x1 = random.randint(0, width - 1)
        y1 = random.randint(0, height - 1)
        x2 = random.randint(0, width - 1)
        y2 = random.randint(0, height - 1)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.line([(x1, y1), (x2, y2)], fill=color, width=2)
    
    return image

    

    




