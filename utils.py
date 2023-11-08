# import os, sys, time, math
import math
# from natsort import natsorted, ns

import numpy as np
import cv2
import json 

import matplotlib.pyplot as plt
# import cma
# from PIL import Image
# import glob
# import re
# from tifffile import imwrite

from gvxrPython3 import gvxr
# from gvxrPython3 import json2gvxr

# from gvxrPython3.utils import visualise # Visualise the 3D environment if k3D is supported
# from gvxrPython3.utils import plotScreenshot # Visualise the 3D environment using Matplotlib

# from gvxrPython3.utils import loadSpekpySpectrum # Generate and load an X-ray spectrum using Spekpy
# from gvxrPython3.utils import loadXpecgenSpectrum # Generate and load an X-ray spectrum using xpecgen

use_padding = True
pad_width = 50



def average_images(image_paths):
    
    """Average a list of images."""
  
    # Load the first image to get the shape
    sample_image = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    if sample_image is None:
        raise ValueError(f"Failed to load image: {image_paths[0]}")
    
    avg_image = np.zeros_like(sample_image, dtype=float)
    
    if use_padding:
        avg_image = np.pad(avg_image, (pad_width, pad_width), mode='median')

    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        if use_padding:
            median_value = np.median(image)
            image = np.pad(image, (pad_width, pad_width), 'constant', constant_values=(median_value, median_value))

        avg_image += image.astype(float)
    
    avg_image /= len(image_paths)
    
    return cv2.medianBlur(avg_image.astype(np.single), 3)

def flatField(img, white, dark, epsilon=0.0):

    temp_white = np.copy(white)
    temp_img = np.copy(img)
    
    test = white - dark == 0
    temp_white[test] += 1

    if len(img.shape) == 2:
        temp_img[test] += 1
        return (temp_img - dark + epsilon) / (temp_white - dark + epsilon)
    elif len(img.shape) == 3:
        flat = np.zeros(img.shape, dtype=np.single)
        for i, proj in enumerate(temp_img):
            proj[test] += 1
            flat[i] = (proj - dark + epsilon) / (temp_white - dark + epsilon)
        return flat
    else:
        raise IOError("Bad image dimension: " + str(img.shape))

def getAverageEnergy(k, f):
    
    avg = 0
    for energy, count in zip(k, f):
        avg += energy * count
        
    return avg / np.sum(f)

def getReference(I_flat, angles_in_deg, number_of_angles):

    images = []
    angles = []
    
    for i in range(number_of_angles):

        if number_of_angles == 1:
            index = 0
        else:
            index = round((i + 1) / number_of_angles * (len(angles_in_deg) // 4))
        
        images.append(I_flat[index])
        angles.append(angles_in_deg[index])

    return np.array(images), np.array(angles)

def getXrayImage(x, take_screenshot=False):

    global screenshot
    screenshot = []

    backup = gvxr.getLocalTransformationMatrix("root")

    # Move source, det, object using x
    x_src = x[0]
    y_src = x[1]
    z_src = x[2]
    gvxr.setSourcePosition(x_src, y_src, z_src, "mm")
    
    x_det = x[3]
    y_det = x[4]
    z_det = x[5]
    gvxr.setDetectorPosition(x_det, y_det, z_det, "mm")

    x_obj1 = x[6]
    y_obj1 = x[7]
    z_obj1 = x[8]

    alpha_x = x[9]
    alpha_y = x[10]
    alpha_z = x[11]

    x_obj2 = x[12]
    y_obj2 = x[13]
    z_obj2 = x[14]

    test_image = []

    up_vector = gvxr.getDetectorUpVector();
    
    for rot_angle in selected_angles:
        gvxr.resetSceneTransformation();
    
        
        
        
    #     gvxr.translateNode("root", x_rot_axis_pos, y_rot_axis_pos, z_rot_axis_pos, "mm")
    
        
        gvxr.rotateNode("root", rot_angle, up_vector[0], up_vector[1], up_vector[2])
        
        gvxr.translateNode("root", x_obj1, y_obj1, z_obj1, "mm")
        
        gvxr.rotateNode("root", alpha_x, 1, 0, 0)
        gvxr.rotateNode("root", alpha_y, 0, 1, 0)
        gvxr.rotateNode("root", alpha_z, 0, 0, 1)
        
        gvxr.translateNode("root", -x_obj2, -y_obj2, -z_obj2, "mm")
        
        test_image.append(gvxr.computeXRayImage())
    
        if take_screenshot:

            gvxr.displayScene()        
            screenshot.append(gvxr.takeScreenshot())
        
        gvxr.setLocalTransformationMatrix("root", backup)

        
    
    return np.array(test_image, dtype=np.single) / gvxr.getTotalEnergyWithDetectorResponse()

def compareMAE(ref, test):
    return np.abs(ref - test).mean()

def compareMSE(ref, test):
    return np.square(ref - test).mean()


def fitnessMAE(x):
    global ref_image, best_fitness, fitness_set, counter

    test_image = getXrayImage(x)
    fitness_value = compareMAE(ref_image, test_image)

    if best_fitness > fitness_value:
        fitness_set.append([counter, fitness_value])
        best_fitness = fitness_value

    counter += 1
        
    return fitness_value

def fitnessMSE(x):
    global ref_image, best_fitness, fitness_set, counter

    test_image = getXrayImage(x)
    fitness_value = compareMSE(ref_image, test_image)
    
    if best_fitness > fitness_value:
        fitness_set.append([counter, fitness_value])
        best_fitness = fitness_value

    counter += 1

    return fitness_value

def displayResult(x, figsize=(15, 4)):
    global screenshot
    test_image = getXrayImage(x, True)
    
    ref_tmp = np.copy(ref_image)
    test_tmp = np.copy(test_image)
    
    ref_tmp -= ref_tmp.mean()
    ref_tmp /= ref_tmp.std()

    test_tmp -= test_tmp.mean()
    test_tmp /= test_tmp.std()

    ZNCC = 100 * (ref_tmp * test_tmp).mean()
    
    #fig, axs = plt.subplots(len(screenshot), 4, figsize=figsize)
    fig, axs = plt.subplots(len(screenshot), 4, figsize=figsize, squeeze=False)
    plt.suptitle("Overall ZNCC=" + "{:.4f}".format(ZNCC) + "%")

    for index in range(len(screenshot)):
#         axs[index][0].imshow(screenshot[index])
#         axs[index][1].imshow(ref_image[index], cmap="gray", vmin=0, vmax=1)
#     #   axs[1].imshow(I_flat,cmap="gray", vmin=0, vmax=1)
#         axs[index][2].imshow(test_image[index],cmap="gray", vmin=0, vmax=1)
#         im = axs[index][3].imshow((ref_image[index] - test_image[index]),cmap="gray", vmin=-1, vmax=1)
#         axs[index][0].set_title("Rotation angle: " + str(selected_angles[index]) + "$^\circ$")
        axs[index][0].imshow(screenshot[index])
        axs[index][1].imshow(ref_image[index], cmap="gray", vmin=0, vmax=1)
        axs[index][2].imshow(test_image[index], cmap="gray", vmin=0, vmax=1)
        im = axs[index][3].imshow((ref_image[index] - test_image[index]), cmap="gray", vmin=-1, vmax=1)
        axs[index][0].set_title("Rotation angle: " + str(selected_angles[index]) + "$^\circ$")


#    im = axs[3].imshow((I_flat - test_image),cmap="gray", vmin=-1, vmax=1)
    # cbar = fig.colorbar(im)

#     for ax in axs:
#         ax.set_xlim([100, 600])
#         ax.set_ylim([211, 470])
#    plt.savefig('x_default.jpg', dpi=300, bbox_inches='tight')

    plt.show()    

def displayRef(ref_image):


    for i in range(ref_image.shape[0]): 
        plt.figure(figsize=(5,5))
        # ax = plt.subplot(1, ref_image.shape[i], i+1)
        plt.title("Angle: " + str(selected_angles[i]))
        plt.imshow(ref_image[i], cmap="gray", vmin=0, vmax=1)

    plt.show()