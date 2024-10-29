import argparse
import copy
import logging
import os
import sys
import time
from pprint import pformat
import torch
import torch.utils.data
import yaml
from utils.model_trainer_generate import generate_cls_model
from torchvision.utils import save_image

sys.path.append('../')
sys.path.append(os.getcwd())

from diffusion.Diffusion import UNet, GaussianDiffusionSampler
from base import defense
from utils.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.fix_random import fix_random
from utils.save_load_attack import load_attack_result


def get_noise(mask, image_size, batch_size, num_channels=3):
    noise = torch.empty(batch_size, num_channels, image_size, image_size)
    noise.normal_(0, 1)
    # noise[mask == 1] = 0
    return noise


def get_mask(image_size, num_blocks, batch_size, num_channels=3, inverted=False):
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


def diffusion(batch_data, batch_bd_label, batch_clean_label, bd_model, image_path):
    modelConfig = {
        "state": "train",
        "epoch": 200,
        "batch_size": 80,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "training_load_weight": None,
        "save_weight_dir": "./diffusion/Checkpoints/",
        "test_load_weight": "ckpt_999_.pt"
    }
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()

        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

        mask = get_mask(32, 8, 1, 3, True).to('cuda:0')
        noise = get_noise(mask, 32, 1, 3)[0].to("cuda:0")
        number_id = 0
        x0 = batch_data[number_id].unsqueeze(0)
        save_image(x0[0], 'x0.png')
        old = copy.deepcopy(x0)
        T1 = 120
        T2 = 120
        t1 = x0.new_ones([x0.shape[0], ], dtype=torch.long) * T1
        x0 = 2.0 * (x0 - 0.5)
        xt1 = sampler.extract(sampler.sqrt_alpha_t, t1, x0.shape) * x0 + sampler.extract(sampler.sqrt_1_alpha_t, t1,
                                                                                         x0.shape) * noise
        sampledImgs = sampler(xt1, T1, None)
        x0bar = sampledImgs * 0.5 + 0.5
        save_image(x0bar[0], 'x0bar.png')
        new = copy.deepcopy(x0bar)

        from skimage.metrics import structural_similarity
        old = old.cpu().numpy()
        new = new.cpu().numpy()
        a = old[0]
        b = new[0]
        ssim, difference = structural_similarity(a, b, data_range=255, channel_axis=0, gaussian_weights=True,
                                                 sigma=1.5, full=True)
        difference = torch.tensor(difference)
        grey = 0.11 * difference[0] + 0.59 * difference[1] + 0.3 * difference[2]

        # grey = (grey - torch.mean(grey)) / torch.std(grey)
        grey = (grey - torch.min(grey)) / (torch.max(grey) - torch.min(grey))
        save_image(grey, 'trigger.png')
        t2 = ((1 - grey) * T2).to(torch.int64).to(device)
        # print(t2)
        x0bar = 2.0 * (x0bar - 0.5)
        xtbar = x0bar
        # print(xtbar.shape)
        for i in range(32):
            for j in range(32):
                a = sampler.extract(sampler.sqrt_alpha_t, t2[i][j].unsqueeze(0), x0bar.shape).squeeze()
                b = sampler.extract(sampler.sqrt_1_alpha_t,t2[i][j].unsqueeze(0), x0bar.shape).squeeze()
                for k in range(3):
                    xtbar[0][k][i][j] = x0bar[0][k][i][j] * a + noise[k][i][j] * b
        save_image(xtbar*0.5+0.5, 'xtbar.png')
        outx0 = sampler(xtbar, T2, t2)
        save_image(outx0[0] * 0.5 + 0.5, image_path+'Detoxified/{}.png'.format(number_id))
        outx0[0] = outx0[0] * 0.5 + 0.5
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.247, 0.243, 0.261]
        std = torch.as_tensor(std).view(-1, 1, 1).to("cuda:0")
        mean = torch.as_tensor(mean).view(-1, 1, 1).to("cuda:0")
        outx0 = outx0.sub_(mean).div_(std)
        pre = bd_model(outx0)
        print(pre)
        print(batch_bd_label[number_id], batch_clean_label[number_id])


class SampDetox(defense):
    def __init__(self, args):
        self.device = None
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k: v for k, v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

        if 'result_file' in args.__dict__:
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--against', type=str, help='the location of result')
        parser.add_argument('--yaml_path', type=str, default="./config.yaml",
                            help='the path of yaml')

    def set_result(self, result_file):
        attack_file = 'results/' + result_file
        save_path = 'results/' + result_file + '/defense/SampDetox/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        # assert(os.path.exists(save_path))
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)
        self.args.save_path = save_path
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(
            args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

    def defense_with_diffusion(self, data, bd_label, clean_label, bd_model):
        for idx in range(len(data)):
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.247, 0.243, 0.261]
            std = torch.as_tensor(std).view(-1, 1, 1).to(self.device)
            mean = torch.as_tensor(mean).view(-1, 1, 1).to(self.device)
            data[idx] = data[idx].to(self.device)
            data[idx].mul_(std).add_(mean)
            diffusion(data[idx], bd_label[idx], clean_label[idx], bd_model, self.args.save_path)

    def mitigation(self):
        self.device = self.args.device
        fix_random(self.args.random_seed)
        # Prepare model, optimizer, scheduler
        bd_model = generate_cls_model(self.args.model, self.args.num_classes)
        bd_model.load_state_dict(self.result['model'])
        bd_model.to(self.device)
        test_tran = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]),
                                  train=False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size,
                                                     num_workers=self.args.num_workers, drop_last=False,
                                                     shuffle=False, pin_memory=args.pin_memory)
        bd_data = []
        bd_label = []
        clean_label = []
        for batch_idx, (images, labels, *other_info) in enumerate(data_bd_loader):
            bd_data.append(images)
            bd_label.append(labels)
            clean_label.append(other_info[2])
            break
        self.defense_with_diffusion(bd_data, bd_label, clean_label, bd_model)

    def defense(self, against):
        self.set_result(against)
        self.set_logger()
        self.mitigation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    SampDetox.add_arguments(parser)
    args = parser.parse_args()
    SampDetox_method = SampDetox(args)
    SampDetox_method.defense(args.against)
