import os
import glob
import math
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader, sampler, Dataset
import numpy as np
import h5py
from PIL import Image
# To attempt to get reproducible results
np.random.seed(42)


class DatasetInterface:
    """Define custom class to interface with images from dataset."""

    def __init__(self, directory: str, classes_names, batch_size, validation_size=0.2,
                 train_transform=None, test_transform=None):
        """Initialize instance of DatasetInterface."""
        self._directory = directory
        self._batch_size = batch_size
        self._classes_names = classes_names

        # Setup database elements
        self._setup_dataset(train_transform, test_transform)
        # Setup samplers for training and validation
        self._setup_samplers(validation_size)
        # Setup dataloaders
        self._setup_dataloaders(batch_size)

    def get_training_loader(self) -> DataLoader:
        """Return reference to training loader for this interface."""
        return self._train_loader

    def get_validation_loader(self) -> DataLoader:
        """Return reference to validation loader for this interface."""
        return self._valid_loader

    def get_testing_loader(self) -> DataLoader:
        """Return reference to testing loader for this interface."""
        return self._test_loader

    def plot_training_images(self, num_images=20):
        """Create plot with some of the training images to confirm they were loaded correctly."""
        # Extract one batch of training images
        images, labels = iter(self.get_training_loader()).next()
        # Convert to numpy to simplify plotting
        images = images.numpy()

        # Prepare figure for the plot
        fig = plt.figure(figsize=(30, 5))
        # Display one by one the requested number of images
        # Put maximum 10 images per row
        n_rows = int(math.ceil(num_images / 10))
        for idx in np.arange(num_images):
            # Add sub-plot and display the image
            ax = fig.add_subplot(n_rows, 10, idx + 1, xticks=[], yticks=[])
            img = images[idx] / 2 + 0.5
            plt.imshow(np.transpose(img, (1, 2, 0)))
            ax.set_title(self._classes_names[labels[idx]])

    def _setup_dataset(self, train_transform=None, test_transform=None):
        """Setup dataset for later loading images."""
        self._train_data = None
        self._test_data = None

    def _setup_samplers(self, validation_size):
        """Setup samplers for loading validation and training images."""
        if 0 < validation_size < 1:
            # When validation size is valid, a portion of the indices is used for validation
            # Extract total number of instances of training, and create list with all indices
            num_train = len(self._train_data)
            indices = list(range(num_train))
            # Shuffle list with indices
            np.random.shuffle(indices)
            # Determine number of indices to use for validation
            split = int(np.floor(validation_size * num_train))
            # Separate indices, some for training, some for validation
            train_idx, valid_idx = indices[split:], indices[:split]

            # Now define Subset Random Samplers to extract indices from training portion of the dataset
            self._train_sampler = sampler.SubsetRandomSampler(train_idx)
            self._valid_sampler = sampler.SubsetRandomSampler(valid_idx)
        else:
            # No samplers if validation size is not valid
            self._train_sampler = None
            self._valid_sampler = None

    def _setup_dataloaders(self, batch_size):
        """Setup data-loaders for training, validation and testing."""
        self._train_loader = DataLoader(self._train_data, batch_size=batch_size, sampler=self._train_sampler)
        if self._valid_sampler:
            self._valid_loader = DataLoader(self._train_data, batch_size=batch_size, sampler=self._valid_sampler)
        else:
            self._valid_loader = None
        self._test_loader = DataLoader(self._test_data, batch_size=batch_size)


class ImgFolderDatasetInterface(DatasetInterface):
    """Define custom class to interface with images from dataset in disk folders."""

    def _setup_dataset(self, train_transform=None, test_transform=None):
        """Setup dataset for later loading images."""
        # Prepare ImageFolder dataset for train and test data
        self._train_data = datasets.ImageFolder(self._directory + '/train', transform=train_transform)
        self._test_data = datasets.ImageFolder(self._directory + '/test', transform=test_transform)


class DatasetH5(Dataset):
    """Define custom Dataset that loads images from H5 file."""

    def __init__(self, in_file, transform=None):
        """Initialize instance of DatasetH5."""
        super(DatasetH5, self).__init__()

        # Load file to read
        self.file = h5py.File(in_file, 'r')
        # Save transform to apply
        self.transform = transform

    def __getitem__(self, index):
        """Get next item for iterator."""
        # Obtain next input and target data based on provided index
        x = Image.fromarray(self.file['X'][index])
        y = int(self.file['Y'][index][0, 0])

        # Apply transform to x if needed
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        """Return number of elements in dataset."""
        return self.file['X'].shape[0]


class H5FileDatasetInterface(DatasetInterface):
    """Define custom class to interface with memory-mapped H5 file for dataset."""

    def __init__(self, directory: str, classes_names, batch_size, validation_size=0.2,
                 train_transform=None, test_transform=None, images_size=256):
        """Initialize instance of H5FileDatasetInterface."""
        # Save desired images size and call parent constructor
        self._images_size = images_size
        super(H5FileDatasetInterface, self).__init__(directory, classes_names, batch_size, validation_size,
                                                     train_transform, test_transform)

    def _setup_dataset(self, train_transform=None, test_transform=None):
        """Setup dataset for later loading images."""
        # Before setting up the dataset, we need to create the memory mapped H5 files
        tr_file_name = 'data_training.h5'
        self._create_h5_file(tr_file_name, os.path.join(self._directory, 'train'))
        test_file_name = 'data_test.h5'
        self._create_h5_file(test_file_name, os.path.join(self._directory, 'test'))

        # With the files created, prepare DatasetH5 dataset for train and test data
        self._train_data = DatasetH5(tr_file_name, transform=train_transform)
        self._test_data = DatasetH5(test_file_name, transform=test_transform)

    def _create_h5_file(self, f_name, directory_path):
        """Create new h5 file with images from directory."""
        # Determine path to images to add to this file
        images_path = self._get_dir_images_path(directory_path)
        n_samples = len(images_path)
        print("Creating h5 file '{}' with {} samples from path '{}'"
              .format(f_name, n_samples, directory_path), flush=True)
        # Create file
        with h5py.File(f_name, "w") as out:
            out.create_dataset("X", (n_samples, self._images_size, self._images_size, 3), dtype='u1')
            out.create_dataset("Y", (n_samples, 1, 1), dtype='i4')
        # Add content of images to h5 file
        self._add_h5_file_content(f_name, images_path)

    def _add_h5_file_content(self, f_name, images_path):
        """Add contents from disk into memory mapped h5 file."""
        with h5py.File(f_name, "a") as out:
            class_idx = -1
            last_class_name = ""
            # Go through each of the images
            for idx, img_path in enumerate(images_path):
                # Open and resize to expected size
                img = Image.open(img_path)
                img = img.resize((self._images_size, self._images_size), Image.BILINEAR)
                if img.mode != "RGB":
                    img = img.convert('RGB')
                # Save to h5 file
                out['X'][idx] = np.asarray(img)
                # Determine name of this class
                class_name = os.path.split(os.path.split(img_path)[-2])[-1]
                if class_name != last_class_name:
                    # Increase index if the class name changed
                    class_idx += 1
                    last_class_name = class_name
                    print("New class idx {} for class_name {}".format(class_idx, last_class_name))
                # Save class index in h5 file
                out['Y'][idx] = class_idx

    @staticmethod
    def _get_dir_images_path(dir_path):
        """Return full path to all JPG images in dir_path."""
        dir_path += "/*"
        classes_dir = sorted(glob.glob(dir_path))
        extensions = ['jpg', 'jpeg', 'png']
        images_path = []
        for class_dir in classes_dir:
            for ext in extensions:
                images_path += glob.glob(class_dir + '/*.' + ext) +\
                               glob.glob(class_dir + '/*.' + ext.upper())
        return images_path
