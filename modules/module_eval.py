
from os import listdir
from os.path import join, basename, splitext
from glob import glob
import warnings

import os
import numpy as np

import modules.module_utils as utils

import cv2

import torch
import torchvision.transforms as transforms


######################################################################################################################
######################################################################################################################
######################################################################################################################


def refine_test_dataset(test_folders_dict):
    """
    임의의 형태의 폴더들을 정제해서 (input.png, target.png)의 형태의 list 를 각 dataset 마다 만들어준다.
    :param test_folders_dict: 초기에 입력으로 받은 무질서한 상태의 데이터 셋 들.
    :return: 정제된 training dataset 의 dict
    dict 의 key : dataset 의 이름.
    dict 의 value (예시) : ([(input0.png, target0.png), (input1.png, target1.png), ...], '학습에 사용될 해당 dataset 의 비율')}
    """
    ##########################################################################################
    refined_datasets = {}

    for key in test_folders_dict.keys():
        input_folder_dir = test_folders_dict[key][0]
        target_folder_dir = test_folders_dict[key][1]
        condition = test_folders_dict[key][2]
        dict_type = test_folders_dict[key][3]
        yuv_dirs_pair = []

        # bvi-dvc DB 10bit
        if dict_type == 0:

            input_yuv_dirs = [join(input_folder_dir, x) for x in sorted(listdir(input_folder_dir))]
            target_yuv_dirs = [join(target_folder_dir, x) for x in sorted(listdir(target_folder_dir))]

            # fixme. 전체 사용중.
            input_yuv_dirs = sorted(glob(os.path.join(input_folder_dir, '*.yuv')))
            target_yuv_dirs = sorted(glob(os.path.join(target_folder_dir, '*.yuv')))

            # 영상들에서 valid set 으로 사용할 비디오만 골라준다. 전체를 사용하면 좋지만 너무 오래걸리니까 ㅠㅠ
            input_yuv_dirs = list(input_yuv_dirs[i] for i in [0, 1, 2, 4, 5, 11])
            target_yuv_dirs = list(target_yuv_dirs[i] for i in [0, 1, 2, 4, 5, 11])


            for input_yuv_dir, target_yuv_dir in zip(input_yuv_dirs, target_yuv_dirs):
                target_basename = os.path.splitext(os.path.basename(target_yuv_dir))[0]

                # w, h 를 읽어주고
                wh = target_basename.split('_')[2].split('x')
                w, h = int(wh[0]), int(wh[1])

                # 비디오에서 내가 테스트 하고자 하는 frame 을 골라준다.

                # for AI in RA dataset
                #frames = [0, 32]
                #frames = [0, 3, 6, 8, 12, 15]
                frames = [0, 3, 10, 12, 16]
                for frame in frames:
                    pair = [input_yuv_dir, target_yuv_dir, w, h, frame]
                    yuv_dirs_pair.append(pair)


        refined_datasets[key] = yuv_dirs_pair

    return refined_datasets


######################################################################################################################
######################################################################################################################
######################################################################################################################


def recon(batch_tensor, net, scale_factor, odd, device):
    # network 에 downscaling 부분이 있으면 영상 사이즈가 downscaling 하는 수 만큼 영상에 padding 을 해줘야 한다.
    # padding 이 된 영상이 network 를 통과한 후 padding 을 지워준다.
    pad = utils.TorchPaddingForOdd(odd, scale_factor=scale_factor)
    batch_tensor = pad.padding(batch_tensor)


    with torch.no_grad():
        batch_tensor_out = net(batch_tensor.to(device))

        batch_tensor_out = pad.unpadding(batch_tensor_out)

        batch_tensor_out = batch_tensor_out
        return batch_tensor_out


def recon_one_frame(img_ori, net, device, scale_factor, downupcount, bit=8):
    # 한 frame 을 복원하는 함수이다.

    # input_ 를 tensor 로 전환해준다.
    if bit == 8:
        totensor = transforms.ToTensor()  # [0,255] -> [0,1] 로 만들어줌.
        input_img = totensor(img_ori)
        dtype = np.uint8
        mybyte_s = 255
    if bit == 10:
        input_img = utils.ToTensor10bit(img_ori)
        if len(input_img.shape) == 2:  # 2 channel 이면 3 channel 로 만들어준다.
            input_img = torch.unsqueeze(input_img, 0)
        dtype = np.uint16
        mybyte_s = 1023

    # conv2d 를 하려면 input 이 4 channel 이어야 한다!
    input_img = input_img.view(1, -1, input_img.shape[1], input_img.shape[2])

    # 복원하기.
    output_img = recon(input_img, net, scale_factor, downupcount, device)

    # 복원하기 직적에 4 channel 로 만들었으니까 다시 3 channel 로 만들어 줘야 한다.
    output_img = output_img.view(-1, output_img.shape[2], output_img.shape[3])

    # 0,1 -> 0,255 or 0,1023
    npimg = output_img.numpy() * mybyte_s

    # 영상 후처리.
    npimg = np.around(npimg)
    npimg = npimg.clip(0, mybyte_s)
    npimg = npimg.astype(dtype=dtype)

    return npimg


def recon_big_one_frame(frame, wh, net, scale_factor, minimum_wh, device, odd=2, bit=8):

    # 딥러닝에 모델을 넣을때 가장 먼저 할 일은 shape 를 3 channel 로 만들어 주는 것!
    if len(frame.shape) == 2:
        frame = np.expand_dims(frame, axis=0)

    ori_w_length, ori_h_length = wh

    # print('결과 영상 껍데기 생성...')
    if bit == 8:
        dtype = np.uint8
    if bit == 10:
        dtype = np.uint16
    img_out_np = np.zeros_like(frame, dtype=dtype)

    # 분할할 사이즈를 계산해준다.
    # print('분할 사이즈 계산하는 중...')
    w_length = ori_w_length
    h_length = ori_h_length
    w_split_count = 0
    h_split_count = 0

    while w_length > minimum_wh and h_length > minimum_wh:
        w_split_count += 1
        h_split_count += 1

        w_length = ori_w_length//w_split_count
        h_length = ori_h_length//h_split_count

    w_pos = 0
    h_pos = 0

    w_count = 0
    h_count = 0
    total_count = 0
    while ori_w_length - w_pos >= w_length:
        w_count += 1
        while ori_h_length - h_pos >= h_length:
            total_count += 1
            # print(f"{total_count}/{w_split_count * h_split_count} Forward Feeding...")
            h_count += 1

            wl = w_length
            hl = h_length

            if w_pos + w_length*2 > ori_w_length:
                wl = ori_w_length - w_pos
            if h_pos + h_length*2 > ori_h_length:
                hl = ori_h_length - h_pos

            j, i, w, h = w_pos, h_pos, wl, hl

            cropped_img = frame[:, i:(i+h), j:(j + w)]
            img_out_np[:, i:(i + h), j:(j + w)] \
                = recon_one_frame(cropped_img, net, device, scale_factor=scale_factor, downupcount=odd, bit=bit)  # 복원 영상이 np 이다.

            h_pos += h_length  # //2

        h_pos = 0
        h_count = 0
        w_pos += w_length  # //2


    return img_out_np


######################################################################################################################
######################################################################################################################
######################################################################################################################



class EvalModule(object):
    """
    영상 전체를 한장씩 eval 해준다. (영상 종류에 따라 다른 폴더에 저장)
    """
    def __init__(self, net_dict, test_datasets, additional_info, cuda_num):
        # gpu 는 뭘 사용할지,
        #self.device = torch.device(f'cuda:{cuda_num}')
        self.device = torch.device(f'cuda')

        # dict 형태의 additional_info 를 통해 그 외 eval 에 필요한 추가 정보를 받는다.
        self.additional_info = additional_info
        self.totensor = transforms.ToTensor()  # [0,255] -> [0,1] 로 만들어줌.

        # 모델 할당하기.
        self.netG = net_dict['G'].to(self.device)

        # 데이터 셋 들 할당하기.
        self.test_datasets = test_datasets


    # RGB 영상으로 저장해서 관찰할 수 있게 해보았다. + psnr 도 측정해보도록 한다.
    def psnr_dict(self):

        psnr_dict = {}

        for input, target in self.test_datasets:
            # input image 를 이름으로 하는 폴더를 만들어준다.
            img_name = splitext(basename(input))[0]

            # 이미지를 cv2에서 BGR형태로 불러오기
            input_img = utils.load_BGR(input)
            target_img = utils.load_BGR(target)

            # psnr 측정
            psnr = utils.get_psnr(input_img, target_img, 0, 255)


            psnr_dict[img_name] = psnr

        return psnr_dict


    def save_output(self, save_dir, iter):
        # dataset 별 psnr 을 저장할 dict, key:데이터 셋 name, value:psnr

        psnr_dict = {}

        for key in self.refined_test_datasets.keys():
            # key 를 이름으로 하는 폴더를 만들어준다.
            utils.make_dirs(f'{save_dir}/{key}')

            psnr_per_frames_dict = {}

            if self.refined_test_datasets[key]:  # 해당 list 가 비어있지 않다면,
                for yuv_pair in self.refined_test_datasets[key]:
                    input_yuv_dir = yuv_pair[0]
                    target_yuv_dir = yuv_pair[1]
                    w = yuv_pair[2]
                    h = yuv_pair[3]
                    start_frame = yuv_pair[4]


                    print(f'{input_yuv_dir} is being evaluated...')


                    # 해당 영상들의 이름 얻기.
                    input_yuv_basename = os.path.splitext(os.path.basename(input_yuv_dir))[0]
                    target_yuv_basename = os.path.splitext(os.path.basename(target_yuv_dir))[0]

                    # 10bit 인지 8bit 인지 판별하는 과정. 이름에 10bit 가 있는지 판별한다.
                    if "10bit" in input_yuv_basename:
                        input_mybit = 10
                    elif "8bit" in input_yuv_basename:
                        input_mybit = 8
                    else:
                        warnings.warn(f"{input_yuv_basename}에 bit 가 명시되어있지 않습니다. 명시가 안되어있을 시 10bit 로 읽습니다.")
                        input_mybit = 10

                    if "10bit" in target_yuv_basename:
                        target_mybit = 10
                    elif "8bit" in target_yuv_basename:
                        target_mybit = 8
                    else:
                        warnings.warn(f"{input_yuv_basename}에 bit 가 명시되어있지 않습니다. 명시가 안되어있을 시 10bit 로 읽습니다.")
                        target_mybit = 10


                    # print(f'reading input frame {input_yuv_dir}...')
                    y_u_v, w_input, h_input = utils.read_one_from_yuvs(
                        input_yuv_dir, w, h, start_frame=start_frame, channel='y_u_v', bit=input_mybit)


                    # print('restoring input frame...')
                    minimum_wh = 1500


                    # input_format 에 맞게 추론 방식을 선택해준다.
                    if self.input_format == 0 or self.input_format == 1:
                        # print('===> Y 복원.')
                        y_recon = recon_big_one_frame(
                            y_u_v['y'], (w, h), scale_factor=1, net=self.netG, minimum_wh=minimum_wh, device=self.device, bit=input_mybit)
                        # print('===> U 복원.')
                        u_recon = recon_big_one_frame(
                            y_u_v['u'], (int(w*0.5), int(h*0.5)), scale_factor=1, net=self.netG, minimum_wh=minimum_wh, device=self.device, bit=input_mybit)
                        # print('===> V 복원.')
                        v_recon = recon_big_one_frame(
                            y_u_v['v'], (int(w*0.5), int(h*0.5)), scale_factor=1, net=self.netG, minimum_wh=minimum_wh, device=self.device, bit=input_mybit)

                    elif self.input_format == 2:
                        y = y_u_v['y']
                        u = cv2.resize(y_u_v['u'], (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                        v = cv2.resize(y_u_v['v'], (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                        input_img = np.stack((y, u, v))

                        # print('===> YUV 복원.')
                        recon = recon_big_one_frame(
                            input_img, (w, h), scale_factor=1, net=self.netG, minimum_wh=minimum_wh,
                            device=self.device, bit=input_mybit)

                        y_recon = recon[0]
                        u_recon = recon[1]
                        v_recon = recon[2]

                        u_recon = cv2.resize(u_recon, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
                        v_recon = cv2.resize(v_recon, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)



                    y_u_v_recon_merged = np.hstack([y_recon.flatten(), u_recon.flatten(), v_recon.flatten()])

                    # print('Saving image... (This may take some time.)')
                    input_yuv_basename = os.path.splitext(os.path.basename(input_yuv_dir))[0]
                    utils.make_dirs(f'{save_dir}/{key}/{input_yuv_basename}')
                    out_dir = f'{save_dir}/{key}/{input_yuv_basename}/{input_yuv_basename}_recon_nof{start_frame}_{str(iter).zfill(10)}'

                    # # save yuv (사용하고 싶을때 주석 풀기)
                    # with open(out_dir + '.yuv', "wb") as f_yuv:
                    #     f_yuv.write(y_u_v_recon_merged.astype('uint8').tobytes())

                    # save png
                    yuv_reshaped = np.reshape(y_u_v_recon_merged, [int(h * 1.5), w])
                    if input_mybit == 8:
                        bgr = cv2.cvtColor(yuv_reshaped, cv2.COLOR_YUV2BGR_I420)
                    else:  # 10 bit 일 경우
                        bgr = cv2.cvtColor(utils.Ten2EightBit(yuv_reshaped), cv2.COLOR_YUV2BGR_I420)
                    cv2.imwrite(out_dir + '.png', bgr)


                    # -------------------------------------------------------------------------
                    # target 과 psnr 을 측정해보자.
                    # print('reading target image...')
                    y_u_v_target, w_input, h_input = utils.read_one_from_yuvs(
                        target_yuv_dir, w, h, start_frame=start_frame, channel='y_u_v', bit=target_mybit)


                    # 연산을 해주기 위해 float32 로 변환해 준다.
                    # 8bit 같은걸로 해놓으면 햇갈린다. 연산할때는 항상 먼저 float32 로 만들어주자!
                    y_recon = y_recon.astype(np.float32)
                    u_recon = u_recon.astype(np.float32)
                    v_recon = v_recon.astype(np.float32)
                    y_u_v_target['y'] = y_u_v_target['y'].astype(np.float32)
                    y_u_v_target['u'] = y_u_v_target['u'].astype(np.float32)
                    y_u_v_target['v'] = y_u_v_target['v'].astype(np.float32)


                    # 8bit 영상이면 10bit 로 만들어주자.
                    # https://jvet-experts.org/doc_end_user/current_document.php?id=10545 참고.
                    if input_mybit == 8:
                        y_recon *= 4
                        u_recon *= 4
                        v_recon *= 4
                    if target_mybit == 8:
                        y_u_v_target['y'] *= 4
                        y_u_v_target['u'] *= 4
                        y_u_v_target['v'] *= 4


                    # print('Calculate PSNR...')
                    # y 에 대한 psnr 을 측정한다.
                    # 10bit 가 본 연구의 기본 값 이기 때문에 10bit 로 측정한다.
                    psnr_y = utils.get_psnr(y_recon.astype(np.float32), y_u_v_target['y'].astype(np.float32), 0, 1023)
                    psnr_u = utils.get_psnr(u_recon.astype(np.float32), y_u_v_target['u'].astype(np.float32), 0, 1023)
                    psnr_v = utils.get_psnr(v_recon.astype(np.float32), y_u_v_target['v'].astype(np.float32), 0, 1023)


                    psnr_per_frames_dict[f'{key}_{input_yuv_basename}-{start_frame}'] = psnr_y
                    # print('\n=============================================')


                    psnr_dict[key] = psnr_per_frames_dict

        return psnr_dict



