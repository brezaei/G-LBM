import numpy as np
import os
from skimage import io, color
from matplotlib import pyplot as plt
import cv2 as cv



bgBasePth = "/media/behnaz/My Book/BackgroundSubtraction/BackgroundSubtraction_LowRankVAE/result_nbn_nshuffle/results"
orgBasePth = "/media/behnaz/My Book/BackgroundSubtraction/Data/SBMnet2016"
# get the list 
ch_list = os.listdir(bgBasePth)

# read the background and original frame for each video
for entry in ch_list:
    vid_list = os.listdir(os.path.join(bgBasePth, entry))
    for vid_name in vid_list:
        bg_pth = os.path.join(bgBasePth, entry, vid_name, "RESULT_background.jpg")
        org_pth = os.path.join(orgBasePth, entry, vid_name, "input/in000000.jpg")
        im_bg = cv.imread(bg_pth)
        #im_bg = cv.cvtColor(im_bg, cv.COLOR_BGR2GRAY)
        im_org = cv.imread(org_pth)
        #im_org = cv.cvtColor(im_org, cv.COLOR_BGR2GRAY)
        #im_org = cv.GaussianBlur(im_org, (5, 5), 0)
        im_dif = cv.cvtColor(np.abs(im_bg - im_org), cv.COLOR_BGR2GRAY)
        #print(np.min(im_bg), np.max(im_bg))
        #print(im_bg.shape)
        #adaptive Gaussian thresholding
        mask_ag = np.ones(im_dif.shape) * 255 #cv.adaptiveThreshold(im_dif, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, -10)
        mask_ag[0:5, : ] = 0
        mask_ag[-5:, :] = 0
        mask_ag[:, 0:5] = 0
        mask_ag[:, -5:] = 0

        #adaptive mean thresholding
        mask_am = cv.adaptiveThreshold(im_dif, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 6)
        mask_am[0:10, : ] = 0
        mask_am[-10:, :] = 0
        mask_am[:, 0:10] = 0
        mask_am[:, -10:] = 0
        # otsu thresholding
        _, mask_ot = cv.threshold(im_dif, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        mask_ot[0:10, : ] = 0
        mask_ot[-10:, :] = 0
        mask_ot[:, 0:10] = 0
        mask_ot[:, -10:] = 0
        # otsu after gaussian
        im_dif_blur = cv.GaussianBlur(im_dif, (5, 5), 0)
        _, mask_otg = cv.threshold(im_dif_blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        mask_otg[0:10, : ] = 0
        mask_otg[-10:, :] = 0
        mask_otg[:, 0:10] = 0
        mask_otg[:, -10:] = 0
        #img_bg_cor = 
        

        titles=['original image', 'background', 'difrence_image', 'mask_Gaussian thresholding',
                'mask_mean thresholding', 'otsu thresholding', 'otsu after Gaussian']

        images = [im_org, im_bg, im_dif,
                mask_ag, mask_am, mask_ot, mask_otg           
        ]
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
        ax = axes.ravel()
        for i, im in enumerate(images):
            ax[i].set_title(titles[i])
            ax[i].imshow(im, cmap='gray')
            ax[i].set_axis_off()

        ax[7].imshow(im_org)
        ax[7].set_title('diff histogram')
        ax[7].set_axis_off()


        fig.tight_layout()
            
        fig_name = "../postprocess_output/SBMnet2016/{}_{}_masks.png".format(entry, vid_name)
        os.makedirs(os.path.dirname(fig_name),exist_ok=True)
        plt.savefig(fig_name)
        org_name = "../toBeMasked/SBMnet2016/{}_{}_org.png".format(entry, vid_name)
        os.makedirs(os.path.dirname(org_name),exist_ok=True)
        cv.imwrite(org_name, im_org)
        bg_name =  org_name = "../toBeMasked/SBMnet2016/{}_{}_bg.png".format(entry, vid_name)
        os.makedirs(os.path.dirname(bg_name),exist_ok=True)
        cv.imwrite(bg_name, im_bg)
        





# sunstitute the background pixels with original where there is no mask


# save the new background images
