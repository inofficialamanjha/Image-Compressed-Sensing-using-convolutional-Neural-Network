import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from torch.autograd import Variable
from network import CSNet_Enhanced
import time, math, glob
import warnings
warnings.filterwarnings("ignore")
import scipy.io as sio
from skimage.measure import compare_ssim
import imutils


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


model = CSNet_Enhanced()
model.load_state_dict(torch.load('CS_net_model_large_final.pth'))

image_list = glob.glob('Test/Set11_mat'+"/*.*")

avg_psnr_predicted = 0.0
avg_ssim_predicted = 0.0
avg_elapsed_time = 0.0
plt.figure()
f, axarr = plt.subplots(11,2 , figsize=(45,45)) 
idx = 0
for image_name in image_list:
    print("Processing ", image_name)
    im_gt_y = sio.loadmat(image_name)['im_gt_y']

    im_gt_y = im_gt_y.astype(float)

    im_input = im_gt_y/255.
    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
    im_input = im_input.cuda()
    model = model.cuda()

    start_time = time.time()
    res = model(im_input)
    elapsed_time = time.time() - start_time
    avg_elapsed_time += elapsed_time
    res = res.cpu()

    im_res_y = res.data[0].numpy().astype(np.float32)

    im_res_y = im_res_y*255.
    im_res_y[im_res_y<0] = 0
    im_res_y[im_res_y>255.] = 255.
    im_res_y = im_res_y[0,:,:]
    axarr[idx][0].imshow(im_gt_y , cmap = 'gray')
    axarr[idx][1].imshow(im_res_y , cmap = 'gray')
    psnr_predicted = PSNR(im_gt_y, im_res_y,shave_border=0)
    (ssim_predicted, diff) = compare_ssim(im_gt_y, im_res_y, full=True)
    print("psnr = " , psnr_predicted)
    print("ssim = " , ssim_predicted)
    avg_psnr_predicted += psnr_predicted
    avg_ssim_predicted += ssim_predicted
    idx = idx+1

print("PSNR_predicted=", avg_psnr_predicted/len(image_list))
print("SSIM_predicted=", avg_ssim_predicted/len(image_list))
print("It takes average {}s for processing".format(avg_elapsed_time/len(image_list)))


