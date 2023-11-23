# import os, sys, time, math
import math
# from natsort import natsorted, ns

import numpy as np
import cv2
import json 

from skimage.metrics import structural_similarity as ssim

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
bbox = None
plot_directory = "."
figsize=(15, 4)

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
    indices = []
    
    for i in range(number_of_angles):

        if number_of_angles == 1:
            index = 0
        else:
            index = round((i + 1) / number_of_angles * (len(angles_in_deg) // 2))
        
        images.append(I_flat[index])
        angles.append(angles_in_deg[index])
        indices.append(index)

    return np.array(images), np.array(angles), np.array(indices)

def rescaleX(x):
    temp = []

    for i, min_bound, max_bound in zip(x, data_range[0], data_range[1]):
        temp.append(min_bound + (max_bound - min_bound) * (i + 1) / 2)

    return temp

def inverseX(x):
    temp = []

    for i, min_bound, max_bound in zip(x, data_range[0], data_range[1]):
        temp.append(-1 + 2 * (i - min_bound) / (max_bound - min_bound))

    return temp



def getXrayImage(x, take_screenshot=False):

    global screenshot, default_up_vector, default_right_vector

    screenshot = []

    identity_matrix = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]
    
#     # Reset transformations
#     gvxr.setLocalTransformationMatrix("root", identity_matrix)

#     for i in range(gvxr.getNumberOfChildren("root")):
#         label = gvxr.getChildLabel("root", i);
#         gvxr.setLocalTransformationMatrix(label, identity_matrix)

    x = rescaleX(x)

    # Move source, det, object using x
    x_src = x[0]
    y_src = x[1]
    z_src = x[2]
    gvxr.setSourcePosition(x_src, y_src, z_src, "mm")
    
    x_det = x[3]
    y_det = x[4]
    z_det = x[5]
    gvxr.setDetectorPosition(x_det, y_det, z_det, "mm")

    x_rot_axis_pos = 0
    y_rot_axis_pos = 0
    z_rot_axis_pos = 0

    x_obj = x[6]
    y_obj = x[7]
    z_obj = 0
    
    alpha_x = x[8]
    alpha_y = x[9]
    # alpha_z = x[14]


    test_image = []

    gvxr.setDetectorUpVector(*default_up_vector);
    gvxr.setDetectorRightVector(*default_right_vector);

    if len(x) == 12:
        gvxr.rotateDetector(x[10], *default_up_vector)
        gvxr.rotateDetector(x[11], *default_right_vector)
        
    up_vector = gvxr.getDetectorUpVector();
        
    # Position the object on the turn table
    for i in range(gvxr.getNumberOfChildren("root")):
        label = gvxr.getChildLabel("root", i);
        gvxr.setLocalTransformationMatrix(label, identity_matrix)
        gvxr.translateNode(label, x_obj, y_obj, z_obj, "mm")  #4
        
        gvxr.rotateNode(label, alpha_x, 1, 0, 0)  #3
        gvxr.rotateNode(label, alpha_y, 0, 1, 0)  #2
        # gvxr.rotateNode(label, alpha_z, 0, 0, 1)  #1
    
    
    label = "root"
    for rot_angle in selected_angles:
    
        # Centre of rotation
        gvxr.setLocalTransformationMatrix("root", identity_matrix)
        
        
        gvxr.translateNode("root", -x_rot_axis_pos, -y_rot_axis_pos, -z_rot_axis_pos, "mm")
        gvxr.rotateNode("root", rot_angle, *up_vector)
        gvxr.translateNode("root", x_rot_axis_pos, y_rot_axis_pos, z_rot_axis_pos, "mm")

      
        bbox = gvxr.getNodeAndChildrenBoundingBox("Rabbit", "mm")
    
    
        
        test_image.append(gvxr.computeXRayImage())
    
        if take_screenshot:

            gvxr.displayScene()        
            screenshot.append(gvxr.takeScreenshot())
    

    return np.array(test_image, dtype=np.single) / gvxr.getTotalEnergyWithDetectorResponse(), bbox

def applyTransformation(x):

    global default_up_vector, default_right_vector
    
    identity_matrix = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]
    
    x = rescaleX(x)

    # Move source, det, object using x
    x_src = x[0]
    y_src = x[1]
    z_src = x[2]
    gvxr.setSourcePosition(x_src, y_src, z_src, "mm")

    x_det = x[3]
    y_det = x[4]
    z_det = x[5]
    gvxr.setDetectorPosition(x_det, y_det, z_det, "mm")

    x_rot_axis_pos = 0
    y_rot_axis_pos = 0
    z_rot_axis_pos = 0

    x_obj = x[6]
    y_obj = x[7]
    z_obj = 0

    alpha_x = x[8]
    alpha_y = x[9]
    # alpha_z = x[14]

    gvxr.setDetectorUpVector(*default_up_vector);
    gvxr.setDetectorRightVector(*default_right_vector);

    if len(x) == 12:
        gvxr.rotateDetector(x[10], *default_up_vector)
        gvxr.rotateDetector(x[11], *default_right_vector)
        
    up_vector = gvxr.getDetectorUpVector();

    # Centre of rotation
    gvxr.setLocalTransformationMatrix("root", identity_matrix)
    gvxr.translateNode("root", x_rot_axis_pos, y_rot_axis_pos, z_rot_axis_pos, "mm") #6

    bbox = gvxr.getNodeAndChildrenBoundingBox("Rabbit", "mm")
       
    # Position the object on the turn table
    for i in range(gvxr.getNumberOfChildren("root")):
        label = gvxr.getChildLabel("root", i);
        gvxr.setLocalTransformationMatrix(label, identity_matrix)
        gvxr.translateNode(label, x_obj, y_obj, z_obj, "mm")  #4
        
        gvxr.rotateNode(label, alpha_x, 1, 0, 0)  #3
        gvxr.rotateNode(label, alpha_y, 0, 1, 0)  #2
        # gvxr.rotateNode(label, alpha_z, 0, 0, 1)  #1
        
    return bbox

def compareMAE(ref, test):
    return np.abs(ref - test).mean()

def compareMSE(ref, test):
    return np.square(ref - test).mean()


def fitnessMAE(x):
    global ref_image, best_fitness, fitness_set, counter, bbox

    test_image, bbox = getXrayImage(x)
    fitness_value = compareMAE(ref_image, test_image)

    if best_fitness > fitness_value:
        fitness_set.append([counter, fitness_value])
        best_fitness = fitness_value
        displayResult(x, figsize)
        plt.savefig(plot_directory + "/plot_" + str(counter) + ".png")
        plt.close()

    counter += 1
        
    return fitness_value

def fitnessMSE(x):
    global ref_image, best_fitness, fitness_set, counter, bbox

    test_image, bbox = getXrayImage(x)
    fitness_value = compareMSE(ref_image, test_image)
    
    if best_fitness > fitness_value:
        fitness_set.append([counter, fitness_value])
        best_fitness = fitness_value
        displayResult(x, figsize)
        plt.savefig(plot_directory + "/plot_" + str(counter) + ".png")
        plt.close()

    counter += 1

    return fitness_value

def fitnessSSIM(x):
    global ref_image, best_fitness, fitness_set, counter, bbox

    test_image, bbox = getXrayImage(x)
    metrics = ssim(ref_image, test_image, data_range=1)
    fitness_value = 1.0 / metrics
    
    if best_fitness > fitness_value:
        fitness_set.append([counter, metrics])
        best_fitness = fitness_value
        displayResult(x, figsize)
        plt.savefig(plot_directory + "/plot_" + str(counter) + ".png")
        plt.close()

    counter += 1

    return fitness_value

def ZNCC(img1, img2):
    return np.mean(((img1 - img1.mean()) / img1.std()) * ((img2 - img2.mean()) / img2.std()))
    
def fitnessZNCC(x):
    global ref_image, best_fitness, fitness_set, counter, bbox

    test_image, bbox = getXrayImage(x)
    metrics = ZNCC(ref_image, test_image)
    fitness_value = 1.0 / metrics
    
    if best_fitness > fitness_value:
        fitness_set.append([counter, metrics])
        best_fitness = fitness_value
        displayResult(x, figsize)
        plt.savefig(plot_directory + "/plot_" + str(counter) + ".png")
        plt.close()

    counter += 1

    return fitness_value

def displayResult(x, figsize=(15, 4)):
    global screenshot, bbox
    test_image, bbox = getXrayImage(x, True)
    
    ref_tmp = np.copy(ref_image)
    test_tmp = np.copy(test_image)

    MAE = compareMAE(ref_image, test_image);
    RMSE = math.sqrt(compareMSE(ref_image, test_image));
    SSIM = 0.0
    
    for img1, img2 in zip(ref_image, test_image):
        SSIM += ssim(img1, img2, data_range=1);
    SSIM /= ref_image.shape[0]
    
    ref_tmp -= ref_tmp.mean()
    ref_tmp /= ref_tmp.std()

    test_tmp -= test_tmp.mean()
    test_tmp /= test_tmp.std()

    ZNCC = 100 * (ref_tmp * test_tmp).mean()
    
    #fig, axs = plt.subplots(len(screenshot), 4, figsize=figsize)
    fig, axs = plt.subplots(len(screenshot), 4, figsize=figsize, squeeze=False)
    plt.suptitle("Overall ZNCC=" + "{:.4f}".format(ZNCC) + "%\n" +
                "Overall MAE=" + "{:.4f}".format(MAE) + "\n" +
                "Overall RMSE=" + "{:.4f}".format(RMSE) + "\n" +
                "Overall SSIM=" + "{:.4f}".format(SSIM))

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
        axs[index][0].set_title("Image: " + str(indices[index]) + " Rotation angle: "  + str(selected_angles[index]) + "$^\circ$")

#    im = axs[3].imshow((I_flat - test_image),cmap="gray", vmin=-1, vmax=1)
    # cbar = fig.colorbar(im)

#     for ax in axs:
#         ax.set_xlim([100, 600])
#         ax.set_ylim([211, 470])
#    plt.savefig('x_default.jpg', dpi=300, bbox_inches='tight')


def displayRef(ref_image):


    for i in range(ref_image.shape[0]): 
        plt.figure(figsize=(5,5))
        # ax = plt.subplot(1, ref_image.shape[i], i+1)
        plt.title("Angle: " + str(selected_angles[i]))
        plt.imshow(ref_image[i], cmap="gray", vmin=0, vmax=1)

    plt.show()
    
