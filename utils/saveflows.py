import os 
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import pyflow


def getListOfFolders(File):
	data = pd.read_csv(File, sep=" ", header=None)[0]
	#data = data.str.split('/',expand=True)[1]
	data = data.str.rstrip(".avi").values.tolist()

	return data


# setting the data addresses
rootDir = "/media/behnaz/My Book/BackgroundSubtraction/Data/SBMnet2016"
saveDir = "/media/behnaz/My Book/BackgroundSubtraction/Data/SBMnet2016_flow"
listOfFolder = '/media/behnaz/My Book/BackgroundSubtraction/Data/SBMnet2016/list.csv'
# setting the pyflow parameters
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0



folderList = getListOfFolders(listOfFolder)[:10]
print("===== start calculating and saving the flow data=================")
print("number of videos to be processes is:{}".format(len(folderList)))


for foldernum,folder in enumerate(folderList):
	frames = [each for each in os.listdir(os.path.join(rootDir,folder)) if each.endswith(('.jpg','.jpeg'))]
	nFrames = len(frames)
	frames.sort()
	if nFrames > 100:
		print(folder, foldernum+1)
		for framenum in range(0,5,nFrames-1):
			imgname = os.path.join(rootDir,folder,frames[framenum])
			img1 = np.array(Image.open(imgname)).astype(float)/255.
			
			imgname = os.path.join(rootDir,folder,frames[framenum+1])
			img2 = np.array(Image.open(imgname)).astype(float)/255.
			
			u, v,_ = pyflow.coarse2fine_flow( img2, img1, alpha, ratio, minWidth, 
								nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
			flow = np.concatenate((u[..., None], v[..., None]), axis=2)
			if not os.path.exists(os.path.join(saveDir, folder)):
				os.makedirs(os.path.join(saveDir, folder))
			#np.save(os.path.join(saveDir, folder,str(framenum)+'.npy'), flow)
			hsv = np.zeros(img1.shape, dtype=np.uint8)
			mo_mask = np.zeros(im1.shape, dtype=np.uint8)
			m_value = np.sqrt(u**2 + v**2)
			theta = np.mean(m_value) * 1.2
			mask = m_value > theta
			mo_mask[mask] = 255
			hsv[:, :, 0] = 255
			hsv[:, :, 1] = 255
			mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
			hsv[..., 0] = ang * 180 / np.pi / 2
			hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
			rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
			cv2.imwrite(os.path.join(saveDir, folder, 'fl_{:6d}.png'.format(framenum)), rgb)
			cv2.imwrite(os.path.join(saveDir, folder, 'fm_{:6d}.png'.format(framenum)), mo_mask)

			