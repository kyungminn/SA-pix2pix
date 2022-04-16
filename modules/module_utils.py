import os
from os.path import *
from os import *
import random
import cv2
import torch
import csv
import numpy as np
from math import log10
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch
import pytorch_ssim
import FID


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # png 읽을때 경고문 없애줌.


## 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch, dataset_name):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               f"{ckpt_dir}/{dataset_name}/{net}/{net}_{epoch}.pth")

## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load(f'{ckpt_dir}/{ckpt_lst[-1]}')

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('_')[1].split('.pth')[0])

    return net, optim, epoch


def gradient_penalty(netD, input, output, label, device):
    batch_size = input.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    alpha = alpha.expand_as(input)
    interpolated = alpha * input.data + (1 - alpha) * output.data
    interpolated = torch.Tensor(interpolated, requires_grad=True).to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated, label)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return ((gradients_norm - 1) ** 2).mean()



def load_BGR(filepath):
    img_BGR = cv2.imread(filepath, cv2.IMREAD_COLOR)
    return img_BGR


# def load_BGR(filepath):
#     img_BGR = Image.open(filepath)
#     return img_BGR

def order_dir(filepath):
    input_dir = filepath[0]
    label_dir = filepath[1]

    dir_pair = []

    input_img_dirs = [join(input_dir, x) for x in sorted(listdir(input_dir))]
    label_img_dirs = [join(label_dir, x) for x in sorted(listdir(label_dir))]

    for input_img_dir, label_img_dir in zip(input_img_dirs, label_img_dirs):
        # input_dir, target_dir 순으로 데이터셋을 만들어준다.
        pair = [input_img_dir, label_img_dir]
        dir_pair.append(pair)


    # shuffle 한번 먹여주었다.
    random.shuffle(dir_pair)

    return dir_pair



def get_psnr(img1, img2, min_value=0, max_value=255):
    """
    psnr 을 계산해준다.
    이미지가 [0., 255] 이면 min_value=0, max_valu=255 로 해주고 (8bit 영상),
    이미지가 [0, 1023] 이면 min_value=0, max_valu=1023 으로 해주고 (10bit 영상),
    이미지가 [-1,1]의 범위에 있으면 min_value=-1, max_valu=1 로 설정 해준다.
    """
    if type(img1) == torch.Tensor:
        mse = torch.mean((img1 - img2) ** 2)
    else:
        mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = max_value - min_value
    return 10 * log10((PIXEL_MAX ** 2) / mse)



def get_ssim(img1, img2):
    return pytorch_ssim.ssim(img1, img2)



def get_FID(img1, img2, batch_size):
    return FID.calculate_fid(img1, img2, False, batch_size)


def make_dirs(path):
    """
    경로(폴더) 가 있음을 확인하고 없으면 새로 생성한다.
    :param path: 확인할 경로
    :return: path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path




def np_random_rotate(input, target):
    # rot90 설명 : https://stackoverflow.com/questions/63972190/understanding-numpy-rot90-axes
    mode = np.random.randint(4)
    input = np.rot90(input, mode, axes=(1, 2))
    target = np.rot90(target, mode, axes=(1, 2))
    return input, target


def np_random_flip(input, target):
    # Flip array in the left/right direction.
    mode = np.random.randint(2)
    if mode == 1:
        input = np.flip(input, axis=1)
        target = np.flip(target, axis=1)
    return input, target



class LogCSV(object):
    def __init__(self, log_dir):
        """
        :param log_dir: log(csv 파일) 가 저장될 dir
        """
        self.head = False
        self.log_dir = log_dir
        f = open(self.log_dir, 'a')
        f.close()

    def make_head(self, header):
        """
        As of Python 3.6, for the CPython implementation of Python,
        dictionaries maintain insertion order by default.
        dict 에 key 생성한 순서가 그대로 유지됨을 확인.
        """
        self.head = True
        with open(self.log_dir, "a") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(header)

    def __call__(self, log):
        """
        :param log: header 의 각 항목에 해당하는 값들의 list
        """
        with open(self.log_dir, "a") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(log)


class TorchPaddingForOdd(object):
    """
    1/(2^downupcount) 크기로 Down-Sampling 하는 모델 사용시 이미지의 사이즈가 홀수 또는 특정 사이즈일 경우
    일시적으로 padding 을 하여 짝수 등으로 만들어 준 후 모델을 통과시키고,
    마지막으로 unpadding 을 하여 원래 이미지 크기로 만들어준다.
    """
    def __init__(self, downupcount, scale_factor=1):
        self.is_height_even = True
        self.is_width_even = True

        self.scale_factor = scale_factor
        self.downupcount = 2 ** downupcount
        self.pad1 = None
        self.pad2 = None

    def padding(self, img):
        # 홀수면 패딩을 체워주는 것을 해주자
        if img.shape[2] % self.downupcount != 0:
            self.is_height_even = False
            self.pad1 = (img.shape[2]//self.downupcount + 1) * self.downupcount - img.shape[2]
            img_ = torch.zeros(img.shape[0], img.shape[1], img.shape[2] + self.pad1, img.shape[3])
            img_[:img.shape[0], :img.shape[1], :img.shape[2], :img.shape[3]] = img
            for i in range(self.pad1):
                img_[:img.shape[0], :img.shape[1], img.shape[2] + i, :img.shape[3]] = img_[:img.shape[0], :img.shape[1], img.shape[2] - 1, :img.shape[3]]
            img = img_
        if img.shape[3] % self.downupcount != 0:
            self.is_width_even = False
            self.pad2 = (img.shape[3] // self.downupcount + 1) * self.downupcount - img.shape[3]
            img_ = torch.zeros(img.shape[0], img.shape[1], img.shape[2], img.shape[3] + self.pad2)
            img_[:img.shape[0], :img.shape[1], :img.shape[2], :img.shape[3]] = img
            for i in range(self.pad2):
                img_[:img.shape[0], :img.shape[1], :img.shape[2], img.shape[3] + i] = img_[:img.shape[0], :img.shape[1], :img.shape[2], img.shape[3] - 1]
            img = img_
        return img

    def unpadding(self, img):
        # 홀수였으면 패딩을 제거하는 것을 해주자
        if not self.is_height_even:
            img.data = img.data[:img.shape[0], :img.shape[1], :img.shape[2] - self.pad1 * self.scale_factor, :img.shape[3]]
        if not self.is_width_even:
            img.data = img.data[:img.shape[0], :img.shape[1], :img.shape[2], :img.shape[3] - self.pad2 * self.scale_factor]
        return img


def batch2one_img(size, images):
    """
    numpy의 batch 를 타일형태의 한장의 이미지로 만들어준다.
    size: (a, b) 형태의 튜플 a = 세로 타일 개수, b = 가로 타일 개수.
    images: input image 의 shape 은 (batch, h, w, channel) 이다.
    :return: color 일 경우 (h, w, 3), 흑백일 경우 (h, w) 인 한장의 이미지.
    """
    h, w = images.shape[1], images.shape[2]
    # color
    if len(images.shape) == 4:
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]  # 나누기 연산 후 몫이 아닌 나머지를 구함
            j = idx // size[1]  # 나누기 연산 후 소수점 이하의 수를 버리고, 정수 부분의 수만 구함
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    # gray scale
    elif len(images.shape) == 3:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')




def plot_grad_flow(named_parameters, dir, iter):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    os.makedirs(f'{dir}', exist_ok=True)
    ave_grads = []
    # var_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            # var_grads.append(p.grad.var().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.figure()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    # plt.plot(np.arange(len(max_grads)), var_grads, alpha=0.5, lw=1, color="tab:purple")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=1, color="k" )
    # plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    # plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4)], ['max-gradient','mean-gradient'])
    # print(layers)
    plt.savefig(f'{dir}/{iter}.png')
