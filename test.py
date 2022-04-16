import os

ckpt_dir = f'./checkpoint/facades/generator_1.pth'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
ckpt_lst = os.listdir(ckpt_dir)
print('a')