import os
from typing import Dict
import torch.optim as optim
from tqdm import tqdm
import sys
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

sys.path.append('../')
sys.path.append(os.getcwd())

from diffusion.Diffusion import UNet, GaussianDiffusionSampler,GaussianDiffusionTrainer, GradualWarmupScheduler


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss = trainer(x_0).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        if e == 199:
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))


def get_checkerboard_noise(mask, image_size, batch_size, num_channels=3):
    noise = torch.empty(batch_size, num_channels, image_size, image_size)
    noise.normal_(0, 1)
    noise[mask == 1] = 0
    return noise


def get_checkerboard_mask(image_size, num_blocks, batch_size, num_channels=3, inverted=False):
    assert image_size % num_blocks == 0, 'image_size must be divisible by num_blocks'
    block_len = int(image_size // num_blocks)
    mask = torch.ones((batch_size, num_channels, num_blocks, num_blocks))
    if inverted:
        mask[:, :, ::2, ::2] = 0.
        mask[:, :, 1::2, 1::2] = 0.
    else:
        mask[:, :, 1::2, ::2] = 0.
        mask[:, :, ::2, 1::2] = 0.
    return mask.repeat_interleave(repeats=block_len, dim=2).repeat_interleave(repeats=block_len, dim=3)


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        # gauss_noise = torch.randn(
        #     size=[1, 3, 32, 32], device=device)
        mask = get_checkerboard_mask(32, 8, 1, 3, False).to('cuda:0')
        noise = get_checkerboard_noise(mask, 32, 1, 3)
        noisyImage = get_png_tensor("Noise_pic/9.png", noise)
        # given_noise = get_pkl_data('Noise_pic/car.pkl')

        # saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(noisyImage, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage, mask, 300)
        round1 = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(round1, os.path.join(
            modelConfig["sampled_dir"], 'round1.png'), nrow=modelConfig["nrow"])

        mask = get_checkerboard_mask(32, 8, 1, 3, True).to('cuda:0')
        noise = get_checkerboard_noise(mask, 32, 1, 3)
        sampledImgs[noise != 0] = 0
        noisyImage = sampledImgs.to('cuda:0') + noise.to('cuda:0')

        sampledImgs = sampler(noisyImage, mask, 200)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])


def noise_process(batch_data):
    mask = get_checkerboard_mask(32, 8, 1, 3, False).to('cuda:0')[0]
    noise = get_checkerboard_noise(mask, 32, 1, 3)[0]
    noise_batch_data = torch.zeros_like(batch_data)
    for i in range(batch_data.shape[0]):
        batch_data[i][noise != 0] = 0
        noise_batch_data[i] = batch_data[i] + noise
    return noise_batch_data, mask


def diffusion_test(batch_data, modelConfig: Dict):
    noise_batch_data, mask = noise_process(batch_data)
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)


import torchvision.transforms as transforms
from PIL import Image
import torch


def get_png_tensor(filename, noise):
    image_path = filename
    image = Image.open(image_path)
    transform = transforms.ToTensor()
    tensor_image = transform(image)
    tensor_image = 2.0 * (tensor_image - 0.5)

    # # 4. 将高斯噪声插入到图像中
    # for i in range(0, 31, 2 * block_size):
    #     for j in range(0, 31, 2 * block_size):
    #         noise = torch.empty(3, block_size, block_size).normal_(0, 1)
    #         tensor_image[:, i:i + block_size, j:j + block_size] = noise

    noise1 = noise[0]
    tensor_image[noise1 != 0] = 0
    tensor_image = tensor_image + noise1

    tensor_image = torch.unsqueeze(tensor_image, dim=0)

    # Repeat the tensor and add to the first dimension
    # repeated_tensor = tensor_image.repeat(noise.size()[0], 1, 1, 1)

    # Print tensor shape
    # print('Tensor shape:', repeated_tensor.shape)
    # # Print tensor shape and data type
    # print('Tensor shape:', tensor_image.shape)
    # return repeated_tensor.to("cuda:0")
    return tensor_image.to("cuda:0")


import pickle
import torch


def get_pkl_data(filename):
    # 读取.pkl文件
    with open(filename, 'rb') as file:
        data = torch.load(file)

    # 打印读取的数据
    # print(data.shape)
    print(data)

    new_data = torch.unsqueeze(data, dim=0)
    print(new_data.shape)

    # normalized_tensor = (new_data - new_data.min()) / (new_data.max() - new_data.min())
    #
    # print(normalized_tensor)
    return new_data


if __name__ == '__main__':
    filename = "Noise_pic/airplane.png"
    # get_pkl_data('Noise_pic/car.pkl')
    get_png_tensor(filename)
