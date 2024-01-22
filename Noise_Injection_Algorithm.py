import random as rn
import numpy as np
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from numpy import save
from numpy import load

"""
---------------------------------------------------------------
                    Emmanuel Pintelas Code
                ... Simplicity is Elegancy ...
---------------------------------------------------------------
"""

# Add Robustness to Microscope noise
def Craft_Microscope_Effect (root_path = 'Sample_Microscope_Images', dist_path = 'Sample_Regions_M', n_clusters = 5):

    """
    This code can be used for creating a sample dataset with Microscope noise regions.
    Instructions to use:
    1st - manually select some sample images (e.g. 10) containing the Microscope noise
    2nd - Run this script specifying the root and dist dirs, and number of clusters. The dist dirs is the output of this script.
    Running the script, it will segment the Microscope artifact (with also non-Microscope segmented regions based on the number of input clusters)
    for every sample.
    3rd - From the output, select the Microscope regions images, discarding the non-Microscope segmented regions), and save them on a folder
    e.g. Sample_Microscope_Regions
    4th - Now, you can use the Sample_Microscope_Regions via our Injection_Algorithm, which will fuse the segmented Microscope_Regions into
    new case images, in order to perform image augmentation for Skin-Cancer tasks
    """

    size, size = 224, 224

    # load Sample_Microscope_Images
    images = os.listdir(root_path)

    # Extract Sample_Regions
    for im in images:
                im_path = os.path.join(root_path,im)
                img = np.array(Image.open(im_path).convert('RGB'))
                img = cv2.medianBlur(img, 55) # reduce texture and shape variations to assist clustering on capturing the basic shape for segmentation

                # plt.figure()
                # plt.imshow(img)
                # plt.show()

                Values = np.zeros((size*size, 5))
                k=-1
                for i in range(size):
                    for j in range(size):
                        k+=1
                        _xx = img[i, j]
                        _xx = list(_xx)
                        _xx = [i]+[j] +_xx
                        _xx = np.array(_xx)
                        Values[k] = _xx

                #clustering based on blured shape and texture
                X = np.copy(Values)
                for i in range(len(X[0])):#   Standarize
                    X[:, i] = (X[:, i] - np.mean(X[:,i]))/np.std(X[:,i])
                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
                labels = kmeans.labels_
                _centroids = kmeans.cluster_centers_

                Discrete_Colors = []
                for v in range(n_clusters):
                    index = np.argwhere(labels == v)
                    discrete_color = np.zeros(( len(index), 5))
                    g = -1
                    for _ in index:
                        g+=1
                        _ = _[0]
                        discrete_color[g] = Values[_]
                    Discrete_Colors.append(discrete_color)

                segm_n = 0
                for _di_shape in Discrete_Colors:
                    segm_n+=1
                    Segmented_Image = np.copy(img)
                    Segmented_Image.fill(255) 
                    i, j = _di_shape[:, 0].astype(int), _di_shape[:, 1].astype(int)
                    rgb = _di_shape[:, 2:]
                    Segmented_Image[i,j] = rgb.astype(int)
                    # plt.figure()
                    # plt.imshow(Segmented_Image)
                    # plt.show()
                    sample_region = Segmented_Image
                    sample_region = Image.fromarray(sample_region.astype('uint8'), 'RGB')
                    sample_region.save(os.path.join(dist_path, str(segm_n)+ im+'.png'))


# Add Robustness to Hair noise
def Craft_Hair_Effect (root_path = 'Sample_Hair_Images', dist_path = 'Sample_Regions_H', n_clusters = 10):

    """
    This code can be used for creating a sample dataset with Hair noise regions.
    Instructions to use:
    1st - manually select some sample images (e.g. 10) containing the Hair noise
    2nd - Run this script specifying the root and dist dirs, and number of clusters. The dist dirs is the output of this script.
    Running the script, it will segment the Hair artifact (with also non-Hair segmented regions based on the number of input clusters)
    for every sample.
    3rd - From the output, select the Hair regions images, discarding the non-Hair segmented regions), and save them on a folder
    e.g. Sample_Hair_Regions
    4th - Now, you can use the Sample_Hair_Regions via our Injection_Algorithm, which will fuse the segmented Hair_Regions into
    new case images, in order to perform image augmentation for Skin-Cancer tasks
    """

    size, size = 224, 224

    # load Sample_Hair_Images
    images = os.listdir(root_path)

    # Extract Sample_Regions
    for im in images:
                im_path = os.path.join(root_path,im)
                img = np.array(Image.open(im_path).convert('RGB'))

                # plt.figure()
                # plt.imshow(img)
                # plt.show()

                Values = np.zeros((size*size, 5))
                k=-1
                for i in range(size):
                    for j in range(size):
                        k+=1
                        _xx = img[i, j]
                        _xx = list(_xx)
                        _xx = [i]+[j] +_xx
                        _xx = np.array(_xx)
                        Values[k] = _xx

                #clustering based on blured shape and texture
                X = np.copy(Values)
                for i in range(len(X[0])):#   Standarize
                    X[:, i] = (X[:, i] - np.mean(X[:,i]))/np.std(X[:,i])
                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
                labels = kmeans.labels_
                _centroids = kmeans.cluster_centers_

                Discrete_Colors = []
                for v in range(n_clusters):
                    index = np.argwhere(labels == v)
                    discrete_color = np.zeros(( len(index), 5))
                    g = -1
                    for _ in index:
                        g+=1
                        _ = _[0]
                        discrete_color[g] = Values[_]
                    Discrete_Colors.append(discrete_color)

                segm_n = 0
                for _di_shape in Discrete_Colors:
                    segm_n+=1
                    Segmented_Image = np.copy(img)
                    Segmented_Image.fill(255) 
                    i, j = _di_shape[:, 0].astype(int), _di_shape[:, 1].astype(int)
                    rgb = _di_shape[:, 2:]
                    Segmented_Image[i,j] = rgb.astype(int)
                    # plt.figure()
                    # plt.imshow(Segmented_Image)
                    # plt.show()
                    sample_region = Segmented_Image
                    sample_region = Image.fromarray(sample_region.astype('uint8'), 'RGB')
                    sample_region.save(os.path.join(dist_path, str(segm_n)+'_'+im+'.png'))



def Injection_Algorithm (noise_regions_path1 = 'Sample_Microscope_Regions',
                         noise_regions_path2 = 'Sample_Hair_Regions',
                         case_images_path = 'Case_Images',
                         dist_path = 'Augmented_Dataset'):
    
    noise_regions1 = os.listdir(noise_regions_path1)
    noise_regions2 = os.listdir(noise_regions_path2)
    case_images = os.listdir(case_images_path)


    # Microscope-based Augmentation
    # if you want to heavy augment your dataset, then repeat this loop more times e.g. put n_augm = 4
    n_augm = 1
    n_it = 0
    while n_it < n_augm:
        n_it+=1
        for c_im in case_images:
                    # Random Sample from Sample_Microscope_Regions
                    n_r = noise_regions1[rn.randint(0,len(noise_regions1)-1)]
                    n_r_path = os.path.join(noise_regions_path1,n_r)
                    n_r_img = np.array(Image.open(n_r_path).convert('RGB'))

                    # load case image
                    c_im_path = os.path.join(case_images_path,c_im)
                    c_img = np.array(Image.open(c_im_path).convert('RGB'))

                    where = np.argwhere(n_r_img!=255)    
                    for _v in where:
                        c_img[_v[0], _v[1], _v[2]] = n_r_img[_v[0], _v[1], _v[2]]

                    c_img = Image.fromarray(c_img.astype('uint8'), 'RGB')
                    c_img.save(os.path.join(dist_path, str(n_it)+'_me_'+ c_im))

    # Hair-based Augmentation
    # if you want to heavy augment your dataset, then repeat this loop more times e.g. put n_augm = 5
    n_augm = 1
    n_it = 0
    while n_it < n_augm:
        n_it+=1
        for c_im in case_images:
                    # Random Sample from Sample_Hair_Regions
                    n_r = noise_regions2[rn.randint(0,len(noise_regions2)-1)]
                    n_r_path = os.path.join(noise_regions_path2,n_r)
                    n_r_img = np.array(Image.open(n_r_path).convert('RGB'))

                    # load case image
                    c_im_path = os.path.join(case_images_path,c_im)
                    c_img = np.array(Image.open(c_im_path).convert('RGB'))

                    mn_rgb = [np.mean(c_img[:,:,0]), np.mean(c_img[:,:,1]), np.mean(c_img[:,:,2])]

                    c_img_new = np.copy(c_img).astype('float32')
                    c_img = c_img.astype('float32')
                    n_r_img = n_r_img.astype('float32')
                    where = np.argwhere(n_r_img!=255)    
                    for _v in where:
                        c_img_new[_v[0], _v[1], _v[2]] = n_r_img[_v[0], _v[1], _v[2]]*0.75 + mn_rgb[_v[2]]*0.25 

                    c_img_new = Image.fromarray(c_img_new.astype('uint8'), 'RGB')
                    c_img_new.save(os.path.join(dist_path, str(n_it)+'_ne_'+ c_im))



# Craft_Microscope_Effect()
# Craft_Hair_Effect()
Injection_Algorithm() 
