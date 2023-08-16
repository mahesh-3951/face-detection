import cv2
import os
import pickle
import numpy as np

data_dir = os.path.join(os.getcwd(),'clean_data')  #cwd-->current working directry
img_dir = os.path.join(os.getcwd(),'images')


image_data = []                                    # for storing the pickel file
labels = []

for i in os.listdir(img_dir):
    image = cv2.imread(os.path.join(img_dir,i))    #read img which referenced by i 
    image = cv2.resize(image,(100,100))            #resize 100x100 -->for converting the images to same dimentio
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #convert to gray
    image_data.append(image)                       #store in image data
    labels.append(str(i).split("_")[0])            #store only names for labeling
    
image_data = np.array(image_data)    
labels = np.array(labels) 


import matplotlib.pyplot as plt
plt.imshow(image_data[395],cmap="gray")           #Print the img which are stored in 395
plt.show()


with open(os.path.join(data_dir,"images.p"),'wb') as f: #converting into pickel file #wb->write back
    pickle.dump(image_data,f)
    
with open(os.path.join(data_dir,"labels.p"),'wb') as f:
    pickle.dump(labels,f)
    

