import os
import glob
import math
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Change base path below to the folder that contains images to split
base_path = '../dataset/train'
img_folders = glob.glob(base_path + '/*')
num_images = len(img_folders)
images = []
image_names = []
for folder in img_folders:
    images_path = glob.glob(folder + '/*')
    img_plot_path = images_path[random.randint(0, 50)]
    img = Image.open(img_plot_path)
    img = img.resize((320, 320), Image.BILINEAR)
    if img.mode != "RGB":
        img = img.convert('RGB')
    images.append(np.asarray(img))
    image_names.append(os.path.split(folder)[-1])


# Prepare figure for the plot
fig = plt.figure(figsize=(30, 5))
# Display one by one the requested number of images
# Put maximum 3 images per row
n_rows = int(math.ceil(num_images / 3))
for idx in np.arange(num_images):
    # Add sub-plot and display the image
    ax = fig.add_subplot(n_rows, 3, idx + 1, xticks=[], yticks=[])
    img = images[idx]
    # plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.imshow(img)
    ax.set_title(image_names[idx])
plt.show()

