import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2

parser = argparse.ArgumentParser(description="Plot result",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--result_dir", default="./result/experiment3", type=str, dest="result_dir")
opt = parser.parse_args()
result_dir = opt.result_dir

loss = np.load(f'{result_dir}/loss.npy')
val_loss = np.load(f'{result_dir}/val_loss.npy')

psnr = np.load(f'{result_dir}/psnr.npy')
all_psnr = np.load(f'{result_dir}/all_psnr.npy')

avg_grad_per_epoch = np.load(f'{result_dir}/avg_grad_per_epoch.npy')
max_grad_per_epoch = np.load(f'{result_dir}/max_grad_per_epoch.npy')
avg_grad_per_iter = np.load(f'{result_dir}/avg_grad_per_iter.npy')
max_grad_per_iter = np.load(f'{result_dir}/max_grad_per_iter.npy')
layers = np.load(f'{result_dir}/layers.npy')

save_dir = f'{result_dir}/plot'
os.makedirs(f'{save_dir}', exist_ok=True)
grad_dir = f'{save_dir}/grad'
os.makedirs(f'{grad_dir}', exist_ok=True)
os.makedirs(f'{grad_dir}/epoch', exist_ok=True)
os.makedirs(f'{grad_dir}/layer', exist_ok=True)


epoch = np.arange(1, psnr.shape[0] + 1)
plt.plot(epoch, loss[:epoch.shape[0]], label='train')
plt.plot(epoch, val_loss, color='orange', label='val')
plt.title('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig(f'{save_dir}/loss.png')

plt.figure()
plt.plot(epoch, psnr, color='salmon')
plt.text(np.argmax(psnr) + 1, max(psnr),
         '(' + str(np.argmax(psnr) + 1) + ', ' + str(np.round(max(psnr), 3)) + ')',
         fontsize=9,
         color='orangered',
         horizontalalignment='center',
         verticalalignment='bottom')
plt.title('psnr')
plt.xlabel('epoch')
plt.savefig(f'{save_dir}/psnr.png')

loss = cv2.imread(f'{save_dir}/loss.png', cv2.IMREAD_COLOR)
psnr = cv2.imread(f'{save_dir}/psnr.png', cv2.IMREAD_COLOR)
total = np.hstack([loss, psnr])
cv2.imwrite(f'{save_dir}/plot.png', total)

color=['firebrick', 'orangered', 'darkorange', 'gold', 'olive', 'yellowgreen', 'limegreen', 'turquoise',\
       'skyblue', 'cornflowerblue', 'mediumslateblue']
# plt.figure()
for i in range(all_psnr.shape[1]):
    # plt.plot(epoch, all_psnr[:, i], label=f'val_{i+1}')
    plt.figure()
    plt.plot(epoch, all_psnr[:, i], color=f'{color[i]}', label=f'val_{i + 1}')
    plt.xlabel('epoch')
    plt.ylabel('psnr')
    plt.legend()
    plt.savefig(f'{save_dir}/val_{i + 1}.png')
    # plt.show()



plt.figure()
for i in range(all_psnr.shape[1]):
    # plt.plot(epoch, all_psnr[:, i], label=f'val_{i+1}')
    # plt.figure()
    plt.plot(epoch, all_psnr[:, i], color=f'{color[i]}', alpha=0.5, label=f'val_{i + 1}')

plt.xlabel('epoch')
plt.ylabel('psnr')
plt.title('all psnr for val')
plt.legend()
plt.savefig(f'{save_dir}/total.png')

v_img = []
h_img = []
for i in range(12):
    if i == 11:
        img = cv2.imread(f'{save_dir}/total.png', cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(f'{save_dir}/val_{i + 1}.png', cv2.IMREAD_COLOR)
    h_img += [img]
    if i % 4 == 3:
        h_stack = np.hstack(h_img)
        v_img += [h_stack]
        h_img = []
img = np.vstack(v_img)
cv2.imwrite(f'{save_dir}/stack.png', img)



# add_epoch = epoch.shape[0]-avg_grad_per_epoch.shape[0]
# new_epoch = np.arange(avg_grad_per_epoch.shape[0]) + add_epoch + 1
num_layer = layers.shape[0]
for i in epoch:
    plt.figure()
    plt.bar(np.arange(1, num_layer + 1), avg_grad_per_epoch[i-1, :], lw=1, alpha=0.5, color='b', label='avg')
    plt.bar(np.arange(1, num_layer + 1), max_grad_per_epoch[i-1, :], lw=1, alpha=0.5, color='c', label='max')
    plt.xlabel('layers')
    plt.title(f'epoch {i}')
    plt.legend()
    plt.savefig(f'{grad_dir}/epoch/epoch_{i}.png')

for i in range(num_layer):
    plt.figure()
    plt.bar(epoch, avg_grad_per_epoch[:, i], lw=1, alpha=0.5, color='b', label='avg')
    plt.bar(epoch, max_grad_per_epoch[:, i], lw=1, alpha=0.5, color='c', label='max')
    plt.xlabel('epoch')
    plt.title(f'{layers[i]}')
    plt.legend()
    plt.savefig(f'{grad_dir}/layer/layer_{i+1}.png')
# iter_count = avg_grad_per_iter.shape[0] // num_layer
# avg_grad = []
# max_grad = []
# for i in range(iter_count):
#     avg = list(avg_grad_per_iter[i * num_layer: (i + 1) * num_layer])
#     max = list(max_grad_per_iter[i * num_layer: (i + 1) * num_layer])
#     avg_grad += [avg]
#     max_grad += [max]
# np.save(f'{result_dir}/avg_grad_iter.npy', avg_grad)
# np.save(f'{result_dir}/max_grad_iter.npy', max_grad)
#
# avg_grad_iter = np.load(f'{result_dir}/avg_grad_iter.npy')
# max_grad_iter = np.load(f'{result_dir}/max_grad_iter.npy')
# epoch_count = avg_grad_per_epoch.shape[0]
# iter_per_epoch = iter_count // epoch_count
# avg_grad_epoch = []
# max_grad_epoch = []
# for i in range(epoch_count):
#     avg = list(avg_grad_iter[i * iter_per_epoch : (i + 1) * iter_per_epoch, :])
#     max = list(max_grad_iter[i * iter_per_epoch : (i + 1) * iter_per_epoch, :])
#     avg_grad_epoch += [np.mean(avg, axis=0)]
#     max_grad_epoch += [np.mean(max, axis=0)]
# np.save(f'{result_dir}/avg_grad_epoch.npy', avg_grad_epoch)
# np.save(f'{result_dir}/max_grad_epoch.npy', max_grad_epoch)
# avg_grad_epoch = np.load(f'{result_dir}/avg_grad_epoch.npy')
# max_grad_epoch = np.load(f'{result_dir}/max_grad_epoch.npy')

# for i in range(avg_grad_per_epoch.shape[1]):
#     plt.plot(new_epoch, avg_grad_per_epoch)


#
# plt.show()


