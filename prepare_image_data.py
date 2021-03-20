import pandas as pd 
import glob
import cv2

train_inp_images = glob.glob('/extradata/2021/1/datasets/NTIRE2021_Train_Hazy/*.png')
train_out_images = glob.glob('/extradata/2021/1/datasets/NTIRE2021_Train_HazeFree/*.png')

val_inp_images = glob.glob('/media/newton/newton/nitre/dataset/val/HAZY/*.png')
val_out_images = glob.glob('/media/newton/newton/nitre/dataset/val/GT/*.png')

#test_inp_images = glob.glob('/media/newton/newton/nitre/dataset/train/HAZY/*.png')

xcords = []
ycords = []

for img_path in train_inp_images:
	for i in range(0,1600-512+1,80):
		for j in range(0,1200-512+1,60):
			img_nm = img_path.split('/')[-1]
			print(img_path)
			print(img_nm)
			frame1 = cv2.imread(img_path)
			frame2 = cv2.imread('/extradata/2021/1/datasets/NTIRE2021_Train_HazeFree/' + img_nm)
			cropImg1 = frame1[j:j+512,i:i+512]
			cropImg2 = frame2[j:j+512,i:i+512]
			cv2.imwrite('./dataset-512-overlap/NTIRE2021_Train_Hazy/'+str(i)+'_'+str(j)+'_'+img_nm,cropImg1)
			cv2.imwrite('./dataset-512-overlap/NTIRE2021_Train_HazeFree/'+str(i)+'_'+str(j)+'_'+img_nm,cropImg2)

