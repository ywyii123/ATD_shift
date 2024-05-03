from utils import util_image
import lpips
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import PIL

def read_images(path):
    images = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.png'):
                path = os.path.join(root, file)
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
    return np.array(images)

def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.array(image)

def cal_lpips(sr_path, gt_path):
    trans = transforms.ToTensor()
    im0_path_list = []
    im1_path_list = []
    for root, dirs, files in os.walk(sr_path):
        for file in files:
            if file.endswith('.png'):
                path = os.path.join(root, file)
                im0_path_list.append(path)
    for root, dirs, files in os.walk(gt_path):
        for file in files:
            if file.endswith('.png'):
                path = os.path.join(root, file)
                im1_path_list.append(path)
    im0_path_list = sorted(im0_path_list)
    im1_path_list = sorted(im1_path_list)
    dist_ = []
    loss_fn = lpips_loss.cuda()
    for i in range(len(im0_path_list)):
        print(im0_path_list[i])
        print()
        print(im1_path_list[i])
        dummy_im0 = trans(PIL.Image.open(im0_path_list[i]))
        dummy_im1 = trans(PIL.Image.open(im1_path_list[i]))
        dummy_im0 = dummy_im0.cuda()
        dummy_im1 = dummy_im1.cuda()
        dist = loss_fn.forward(dummy_im0, dummy_im1)
        dist_.append(dist.mean().item())
    return (sum(dist_)/len(im0_path_list))

    
    

if __name__ == '__main__':
    lpips_loss = lpips.LPIPS(net='vgg').cuda()
    sr_path = '/home/youweiyi/project/ResShift/output/CT_sigma_2_uniform_s0_4_s1_32_rho_1'
    gt_path = '/home/youweiyi/project/ResShift/ImageNet-Test/gt'

    sr_images = read_images(sr_path)
    gt_images = read_images(gt_path)
    sr_images = sr_images.astype(np.float64)
    gt_images = gt_images.astype(np.float64)
    print(sr_images.shape)
    sr_images = torch.tensor(sr_images / 255.0, dtype=torch.float64).permute(0, 3, 1, 2)
    gt_images = torch.tensor(gt_images / 255.0, dtype=torch.float64).permute(0, 3, 1, 2)
    psnr = util_image.batch_PSNR(sr_images, gt_images, ycbcr=True)
    lpips_mean = cal_lpips(sr_path, gt_path)
    psnr /= len(sr_images)
    print(psnr)
    print(lpips_mean)