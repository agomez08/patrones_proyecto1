import random
import glob
import os

# Change base path below to the folder that contains images to split
base_path = '../dataset/train'
extensions = ['jpg', 'jpeg', 'png']
images_path = []
for ext in extensions:
    images_path += glob.glob(base_path + '/*.' + ext) + glob.glob(base_path + '/*.' + ext.upper())

# Determine 20% random indices for the images to transfer to test folder
imgs_count = len(images_path)
test_imgs_count = int(0.2 * imgs_count)
print("imgs_count={}, test_imgs_count={}".format(imgs_count, test_imgs_count))
test_indices = random.sample(range(imgs_count), test_imgs_count)

# Use these indices to extract their full path
test_images_path = [images_path[test_idx] for test_idx in test_indices]
print("test_images_path for test folder: {}".format(test_images_path))

# Create folder for test files
test_dir_path = os.path.join(base_path, 'test')
os.mkdir(test_dir_path)

# Now transfer the files
for test_image in test_images_path:
    file_name = os.path.split(test_image)[-1]
    os.rename(test_image, os.path.join(test_dir_path, file_name))
