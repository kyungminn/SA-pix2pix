import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from modules.module_data import ImageDataset
from modules.module_utils import *
from modules.module_eval import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
import models.pix2pix as p2p

import numpy as np
import cv2



## make Parser
parser = argparse.ArgumentParser(description="Train the Task",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr_G", default=1e-4, type=float, dest="lr_G")
parser.add_argument("--lr_D", default=4e-4, type=float, dest="lr_D")
parser.add_argument("--batch_size", default=5, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=200, type=int, dest="num_epoch")
parser.add_argument("--patch_ratio", default=16, type=int, dest="patch_ratio")
parser.add_argument('--lambda_pixel', default=100, type=int, dest='lambda_pixel')
parser.add_argument('--lambda_gp', default=10, type=int, dest='lambda_gp')
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")

parser.add_argument('--direction', default='b2a', type=str, dest='direction')
parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--dataset_name", default="facades", type=str, dest="dataset_name")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")
parser.add_argument("--model", default="pix2pix", type=str, dest="model")
parser.add_argument('--adv_loss', default='wgan-gp', type=str, dest='wgan-gp')
parser.add_argument('--L1loss', default=True, type=bool, dest='L1loss')

opt = parser.parse_args()


## 트레이닝 파라메터 설정하기
lr_G = opt.lr_G
lr_D = opt.lr_D
batch_size = opt.batch_size
num_epoch = opt.num_epoch
patch_ratio = opt.patch_ratio
lambda_pixel = opt.lambda_pixel
lambda_gp = opt.lambda_gp
beta1 = opt.beta1
beta2 = opt.beta2
img_height = opt.img_height
img_width = opt.img_width
channels = opt.channels

direction = opt.direction
data_dir = opt.data_dir
dataset_name = opt.dataset_name
ckpt_dir = opt.ckpt_dir
log_dir = opt.log_dir
result_dir = opt.result_dir
sample_dir = os.path.join(data_dir, 'sample')
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

mode = opt.mode
train_continue = opt.train_continue
model = opt.model
adv_loss = opt.adv_loss
L1loss = opt.L1loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"number of epoch: {num_epoch}")
print(f"learning rate: {lr}")
print(f"batch size: {batch_size}")
print(f'patch ratio: {patch_ratio}')
print(f"data dir: {data_dir}")
print(f"ckpt dir: {ckpt_dir}")
print(f"log dir: {log_dir}")
print(f"result dir: {result_dir}")
print(f"mode: {mode}")
print(f'model: {model}')



## 네트워크 학습하기
if mode == 'train':
    transforms_ = [
        transforms.Resize((img_height, img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataset_train = ImageDataset(f"{data_dir}/{dataset_name}", direction=direction, transforms_=transforms_)
    dataset_val = ImageDataset(f"{data_dir}/{dataset_name}", direction=direction, transforms_=transforms_, mode="val")


    dataloader = DataLoader(
        dataset_train,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=8,
    )

    val_dataloader = DataLoader(
        dataset_val,
        batch_size=10,
        shuffle=True,
        num_workers=1,
    )

    # 그밖에 부수적인 variables 설정하기
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_iter_train = int(np.ceil(num_data_train / batch_size))
    num_iter_val = int(np.ceil(num_data_val / batch_size))

else:
    transforms_ = [
        transforms.Resize((img_height, img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataset_test = ImageDataset(f"./data/{dataset_name}", direction=direction, transforms_=transforms_, mode="test")
    loader_test = DataLoader(dataset_test, batch_size=10, shuffle=False, num_workers=8)

    # 그밖에 부수적인 variables 설정하기
    num_data_test = len(dataset_test)
    num_iter_test = int(np.ceil(num_data_test / batch_size))

##
if model == 'pix2pix':
    netG = p2p.GeneratorUNet().to(device)
    netD = p2p.Discriminator().to(device)

# elif model == 'ResNet_Unet':
#     model = net2.ResUnet101().to(device)
#
# else:
#     model = net3.ResUnet101().to(device)

print(f'\n===> model size')
print(f'Number of params (Generator): {sum([p.data.nelement() for p in netG.parameters()])}')
print(f'Number of params (Discriminator): {sum([p.data.nelement() for p in netD.parameters()])}')
##

# summary(model, (3, 2448, 3264))
optimizer_G = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, beta2))
optimizer_D = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, beta2))
criterion_GAN = nn.BCELoss()
criterion_pixelwise = nn.L1Loss()

# Calculate output of image discriminator (PatchGAN)
patch = (1, img_height // patch_ratio, img_width // patch_ratio)


## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = torch.tensor(imgs["input"])
    real_B = torch.tensor(imgs["label"])
    fake_B = netG(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, f"images/{dataset_name}/{batches_done}.png", nrow=5, normalize=True)


## 네트워크 학습시키기
st_epoch = 0

loss_G_per_epoch = []
loss_D_per_epoch = []
val_loss_G_per_epoch = []
val_loss_D_per_epoch = []
pnsr_per_epoch = []
ssim_per_epoch = []
FID_per_epoch = []

# TRAIN MODE
if mode == 'train':
    if train_continue == "on":
        netG, optimizer_G, st_epoch = load(ckpt_dir=f'{ckpt_dir}/{dataset_name}/netG', net=netG, optim=optimizer_G)
        netD, optimizer_D, st_epoch = load(ckpt_dir=f'{ckpt_dir}/{dataset_name}/netD', net=netD, optim=optimizer_D)

        loss_G_per_epoch = list(np.load(f'{result_dir}/loss_G.npy'))
        loss_D_per_epoch = list(np.load(f'{result_dir}/loss_D.npy'))
        val_loss_G_per_epoch = list(np.load(f'{result_dir}/val_loss_G.npy'))
        val_loss_D_per_epoch = list(np.load(f'{result_dir}/val_loss_D.npy'))
        pnsr_per_epoch = list(np.load(f'{result_dir}/psnr.npy'))
        ssim_per_epoch = list(np.load(f'{result_dir}/ssim.npy'))
        FID_per_epoch = list(np.load(f'{result_dir}/FID.npy'))


    for epoch in range(st_epoch + 1, num_epoch + 1):
        netG.train()
        netD.train()
        loss_G_arr = []
        loss_D_arr = []
        print(f'-' * 100)

        for iter_count, data in enumerate(dataloader, 1):
            loss_G_per_iter = []
            loss_D_per_iter = []

            if iter_count <= 10:
                inp = data['input']
                label = data['label']

                npimg = inp.numpy() * 255
                npimg = np.transpose(npimg, (0, 2, 3, 1))
                npimg = np.squeeze(npimg)
                sample_size = math.ceil(batch_size ** 0.5)
                img = Image.fromarray(np.uint8(batch2one_img((sample_size, sample_size), npimg)))
                img = np.array(img)
                cv2.imwrite(f'{sample_dir}/{iter_count}_input.png', img)

                npimg = label.numpy() * 255
                npimg = np.transpose(npimg, (0, 2, 3, 1))
                npimg = np.squeeze(npimg)
                sample_size = math.ceil(batch_size ** 0.5)
                img = Image.fromarray(np.uint8(batch2one_img((sample_size, sample_size), npimg)))
                img = np.array(img)
                cv2.imwrite(f'{sample_dir}/{iter_count}_label.png', img)

            # Model inputs
            input = data["input"].to(device)
            label = data["label"].to(device)

            output = netG(input)

            # Adversarial ground truths
            real_label = torch.ones((input.size(0), *patch), requires_grad=False).to(device)
            fake_label = torch.zeros((input.size(0), *patch), requires_grad=False).to(device)

            if adv_loss == 'original':

                # --------------------------------------------------------------------
                #  Train Discriminator
                # --------------------------------------------------------------------

                optimizer_D.zero_grad()

                # Real loss
                pred_real = netD(input, label)  # D(x,y)
                loss_D_real = criterion_GAN(pred_real, real_label)  # log(D(x, y))

                # Fake loss
                pred_fake = netD(input, output.detach())  # D(x,G(x))
                loss_D_fake = criterion_GAN(pred_fake, fake_label)  # log(1-D(x,G(x)))

                # Total loss
                loss_D = 0.5 * (loss_D_real + loss_D_fake)  # 0.5{ log(D(x, y)) + log(1-D(x,G(x))) }
                # D가 학습속도가 G에 비해 훨씬 빠르기 때문에 G에 비해 상대적으로 D를 느리게 학습시키게 하기 위해 1/2 곱함

                loss_D.backward()
                optimizer_D.step()


                # --------------------------------------------------------------------
                #  Train Generators
                # --------------------------------------------------------------------

                optimizer_G.zero_grad()

                # GAN loss
                pred_fake = netD(input, label)  # D(x,y)
                loss_GAN = criterion_GAN(pred_fake, real_label)  # log(D(x, y))

                # Pixel-wise loss
                loss_pixel = criterion_pixelwise(output, label)  # |y-G(x)|

                # Total loss
                loss_G = loss_GAN + lambda_pixel * loss_pixel

                loss_G.backward()
                optimizer_G.step()


            elif adv_loss == 'wgan-gp':

                # --------------------------------------------------------------------
                #  Train Discriminator
                # --------------------------------------------------------------------

                optimizer_D.zero_grad()

                # Real loss
                pred_real = netD(input, label)  # D(x,y)
                loss_D_real = torch.mean(pred_real)

                # Fake loss
                pred_fake = netD(input, output.detach())  # D(x,G(x))
                loss_D_fake = torch.mean(pred_fake)

                gp = gradient_penalty(netD, input, output, label, device)

                # Total loss
                loss_D = loss_D_fake - loss_D_real + lambda_gp * gp

                loss_D.backward()
                optimizer_D.step()

                # --------------------------------------------------------------------
                #  Train Generators
                # --------------------------------------------------------------------

                optimizer_G.zero_grad()

                # GAN loss
                pred_fake = netD(input, label)  # D(x,y)
                loss_G = - torch.mean(pred_fake)

                if L1loss:
                    # Pixel-wise loss
                    loss_pixel = criterion_pixelwise(output, label)  # |y-G(x)|

                    # Total loss
                    loss_G = loss_G + lambda_pixel * loss_pixel

                    loss_G.backward()
                    optimizer_G.step()

                else:
                    loss_G.backward()
                    optimizer_G.step()


            # 손실함수 계산
            loss_G_arr += [loss_G.item()]
            loss_G_per_iter += [loss_G.item()]

            loss_D_arr += [loss_D.item()]
            loss_D_per_iter += [loss_D.item()]

            interval = 20
            if iter_count % interval == 0:
                print(f"TRAIN: EPOCH {epoch} / {num_epoch} | iter {iter_count} / {num_iter_train} "
                      f"| G LOSS {np.mean(loss_G_per_iter)} | D LOSS {np.mean(loss_D_per_iter)}")



        loss_D_per_epoch += [np.mean(loss_D_arr)]
        loss_G_per_epoch += [np.mean(loss_G_arr)]
        np.save(f'{result_dir}/loss_D.npy', loss_D_per_epoch)
        np.save(f'{result_dir}/loss_G.npy', loss_G_per_epoch)

        with torch.no_grad():
            netG.eval()
            netD.eval()
            val_loss_G_arr = []
            val_loss_D_arr = []
            psnr_arr = []
            ssim_arr = []
            FID_arr = []

            print(f'-' * 100)
            print(f'\n===> Eval')

            for iter_count, data in enumerate(val_dataloader, 1):

                input = data['input']
                label = data['label']

                output = netG(input)

                # Adversarial ground truths
                real_label = torch.ones((input.size(0), *patch), requires_grad=False).to(device)
                fake_label = torch.zeros((input.size(0), *patch), requires_grad=False).to(device)

                psnr = get_psnr(output, label, min_value=0, max_value=1)
                psnr_arr += [psnr]

                ssim = get_ssim(output, label)
                ssim_arr += [ssim]

                FID = get_FID(output, label, batch_size=label.size(0))
                FID_arr += [FID]

                if adv_loss == 'original':

                    # Real loss
                    pred_real = netD(input, label)  # D(x,y)
                    loss_D_real = criterion_GAN(pred_real, real_label)  # log(D(x, y))

                    # Fake loss
                    pred_fake = netD(input, output.detach())  # D(x,G(x))
                    loss_D_fake = criterion_GAN(pred_fake, fake_label)  # log(1-D(x,G(x)))

                    # Total loss
                    loss_D = 0.5 * (loss_D_real + loss_D_fake)  # 0.5{ log(D(x, y)) + log(1-D(x,G(x))) }
                    # D가 학습속도가 G에 비해 훨씬 빠르기 때문에 G에 비해 상대적으로 D를 느리게 학습시키게 하기 위해 1/2 곱함


                    # --------------------------------------------------------------------
                    #  Train Generators
                    # --------------------------------------------------------------------

                    # GAN loss
                    pred_fake = netD(input, label)  # D(x,y)
                    loss_GAN = criterion_GAN(pred_fake, real_label)  # log(D(x, y))

                    # Pixel-wise loss
                    loss_pixel = criterion_pixelwise(output, label)  # |y-G(x)|

                    # Total loss
                    loss_G = loss_GAN + lambda_pixel * loss_pixel


                elif adv_loss == 'wgan-gp':

                    # --------------------------------------------------------------------
                    #  Train Discriminator
                    # --------------------------------------------------------------------

                    # Real loss
                    pred_real = netD(input, label)  # D(x,y)
                    loss_D_real = torch.mean(pred_real)

                    # Fake loss
                    pred_fake = netD(input, output.detach())  # D(x,G(x))
                    loss_D_fake = torch.mean(pred_fake)

                    gp = gradient_penalty(netD, input, output, label, device)

                    # Total loss
                    loss_D = loss_D_fake - loss_D_real + lambda_gp * gp

                    # --------------------------------------------------------------------
                    #  Train Generators
                    # --------------------------------------------------------------------

                    # GAN loss
                    pred_fake = netD(input, label)  # D(x,y)
                    loss_G = - torch.mean(pred_fake)

                    if L1loss:
                        # Pixel-wise loss
                        loss_pixel = criterion_pixelwise(output, label)  # |y-G(x)|

                        # Total loss
                        loss_G = loss_G + lambda_pixel * loss_pixel



            val_loss_G_per_epoch += [np.mean(val_loss_G_arr)]
            val_loss_D_per_epoch += [np.mean(val_loss_D_arr)]
            pnsr_per_epoch += [np.mean(psnr_arr)]
            ssim_per_epoch += [np.mean(ssim_arr)]
            FID_per_epoch += [np.mean(FID_arr)]


        print(f"VAL: EPOCH {epoch} / {num_epoch} | PSNR {np.mean(psnr_arr)} | SSIM {np.mean(ssim_arr)} | FID {np.mean(FID_arr)}")
        print(f'best psnr: {max(pnsr_per_epoch)} | epoch: {np.argmax(pnsr_per_epoch)+1}\n')
        print(f'best ssim: {max(ssim_per_epoch)} | epoch: {np.argmax(ssim_per_epoch) + 1}\n')
        

        np.save(f'{result_dir}/val_loss_G.npy', val_loss_G_per_epoch)
        np.save(f'{result_dir}/val_loss_D.npy', val_loss_D_per_epoch)
        np.save(f'{result_dir}/psnr.npy', pnsr_per_epoch)
        np.save(f'{result_dir}/ssim.npy', ssim_per_epoch)

        save(ckpt_dir=ckpt_dir, net=netG, optim=optimizer_G, epoch=epoch, dataset_name=dataset_name)
        save(ckpt_dir=ckpt_dir, net=netD, optim=optimizer_D, epoch=epoch, dataset_name=dataset_name)



# TEST MODE
else:
    netG, optimizer_G, st_epoch = load(ckpt_dir=f'{ckpt_dir}/{dataset_name}/netG', net=netG, optim=optimizer_G)

    num = 20000
    with torch.no_grad():
        model.eval()
        loss_arr = []

        for data in loader_test:
            # forward pass
            inp = data['input'].to(device)

            output = model(inp)

            # 현재 output의 값이 0~1의 값을 가지니 255를 곱함.
            output = output.cpu().numpy() * 255

            # output.shape = (batch size, channel num, height, width)
            # shape = (batch size, height, width, channel num)으로 바꾸기 위해 np.transpose 함수 사용
            img = np.transpose(np.around(output), (0, 2, 3, 1))

            # 현재 shape에서 1의 값을 가지는 batch size를 제거하기 위해 np.squeeze 함수를 사용
            img = np.squeeze(img)
            cv2.imwrite(f'{result_dir}/test_{num}.png', img)
            num += 1


