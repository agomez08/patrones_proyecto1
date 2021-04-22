# Tutorial - Kitchen utensils pattern recognition system

Guide for setting, configuring, and using a MIPI-CSI Raspberry Pi v2 camera with the NVIDIA Jetson Nano Developer Kit for the recognition of 10 different types of kitchen utensils.

<h2>Hardware - Brief description</h2>

<h3>Raspberry Pi Camera Module v2</h3> 

The v2 Camera Module has a Sony IMX219 8-megapixel sensor (compared to the 5-megapixel OmniVision OV5647 sensor of the original camera). It can be used to take high-definition video, as well as stills photographs. It supports 1080p30, 720p60 and VGA90 video modes, as well as still capture. It attaches via a 15cm ribbon cable to the CSI port on the Raspberry Pi.

<h3>NVIDIA Jetson Nano Developer Kit</h3>

This developer kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing.

Key features:
- GPU: 128-core Maxwell™ GPU.
- CPU: quad-core ARM® Cortex®-A57 CPU.
- Memory: 4GB 64-bit LPDDR4.

<h2>Hardware - Setup</h2>

The Jetson Nano Developer Kit uses a microSD card as a boot device and for main storage. It’s important to have a card that’s fast and large enough; the minimum recommended is a 32 GB UHS-1 card.

To prepare your microSD card, you’ll need a computer with Internet connection and the ability to read and write SD cards, either via a built-in SD card slot or adapter.

Download the Jetson Nano Developer Kit SD Card Image from https://developer.nvidia.com/jetson-nano-sd-card-image, and note where it was saved on the computer. When developing this project, the latest available image version was "jetson-nano-jp451-sd-card-image".

Write the image to your microSD card by using Etcher software.

The camera should be installed in the MIPI-CSI Camera Connector on the carrier board. The pins on the camera ribbon should face the Jetson Nano module, the stripe faces outward.

<h2>Software - Brief description</h2>

<h3>Linux4Tegra OS</h3>

The official operating system for the Jetson Nano and other Jetson boards is called Linux4Tegra, which is actually a version of Ubuntu 18.04 that’s designed to run on Nvidia’s hardware.

<h3>PyTorch</h3>

Is an optimized tensor library for deep learning using GPUs and CPUs. It is an open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing, primarily developed by Facebook's AI Research lab. It is free and open-source software released under the Modified BSD license.

<h3>OpenCV</h3>

Is an open source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in the commercial products. Being a BSD-licensed product, OpenCV makes it easy for businesses to utilize and modify the code.

The library has more than 2500 optimized algorithms, which includes a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms. 

<h2>Software - Setup</h2>

The NVIDIA Jetson Nano has two power modes pre-defined:

- “Mode 0” is 10W (must use at least a 5V @ 4A power source)
- “Mode 1” is 5W

Important note: “Mode 0” is the default mode with 10W power mode and maximum performance (using all 4 cores and maximun GPU frequency). This is the chosen mode for this project.

When using JetPack 4.5.1, there are a couple of steps in order to setup the OS's lite mode (which helps save much needed RAM). First you will need to select LXDE as your desktop environment. This selection is available when you login (this screen is known as a greeter) in the settings menu (the gear icon).

The pattern recognition system runs best if there is a "swap" file of size 4GB, so that if the Jetson Nano is a little short of RAM it can extend a bit by swapping with some of the (slower) disk space. After setting up the microSD card and booting the system, run the following commands:

```
# Disable ZRAM:
sudo systemctl disable nvzramconfig

# Create 4GB swap file
sudo fallocate -l 4G /mnt/4GB.swap
sudo chmod 600 /mnt/4GB.swap
sudo mkswap /mnt/4GB.swap

# Append the following line to /etc/fstab
sudo echo "/mnt/4GB.swap swap swap defaults 0 0" >> /etc/fstab

# REBOOT!
```

PyTorch was installed by following these steps: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048

OpenCV was installed using the following guide: https://github.com/mdegans/nano_build_opencv

<h3>Dataset</h3>

We decided to use the Edinburgh Kitchen Utensils dataset. This dataset has a total of 897 image with 20 classes. Also, in order to prepare the dataset, the following actions were taken:
- condense categories into a single category
- removed low quality images after a visual inspection
- images were added using standard google searches of the different utensils. This with the help of the "Download All Images" Google Chrome plugging.

These are the 10 chosen categories
- Bottle Opener
- Dinner Fork
- Dinner Knife
- Fish Slice
- Kitchen Knife
- Ladle
- Potato Peeler
- Spatula
- Spoon
- Whisk

<h3>Training</h3>

We relied on an Inception-v3 pre-trained model for features extraction and perform transfer learning. The full process is explained via this Jupyter notebook: 

:page_facing_up: [tl_inceptionv3.ipynb](https://github.com/agomez08/patrones_proyecto1/blob/main/training/tl_inceptionv3.ipynb)

<h3>Live camera prediction</h3>

This is a description of the [kitchen_utensils_predictor.py](https://github.com/agomez08/patrones_proyecto1/blob/main/trained_classifier_tests/kitchen_utensils_predictor.py)
Python script that enables the live streaming, applies the previously generated model, and processes the captured image to display an on-screen prediction of the object in front of the camera.

<h4>Perform necessary imports</h4>

```python
import time
import torch
from collections import OrderedDict
from torchvision import transforms, models
from PIL import Image
import cv2
```

<h4>General constants for the code</h4>
<h5>Define project task and what categories of data will be used</h5>

```python
classes_names = ['BOTTLE_OPENER', 'DINNER_FORK', 'DINNER_KNIFE', 'FISH_SLICE', 'KITCHEN_KNIFE', 'LADLE',
                 'POTATO_PEELER', 'SPATULA', 'SPOON', 'WHISK']
NUM_CLASSES = len(classes_names)
MODEL_FILE_PATH = '../training/models/tl_inceptionv3_model_v2.pt'
SCORE_THRESHOLD = 60
```
<h4>Define functions</h4>
<h5>Return torch device to use. It will return cuda if a supported GPU is available</h5>

```python
def get_runtime_device():
    # Determine device to use (to use GPU if available). 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
```

<h5>Return trained model for evaluation</h5>

```python
def get_trained_model(pt_model_file_path, device):
    # Define already trained Inception-v3 model for learning transfer. 
    model = models.inception_v3(pretrained=True, aux_logits=False)
    # Define custom classifier as FCL
    classifier = torch.nn.Sequential(OrderedDict([('fc1', torch.nn.Linear(2048, 1024)),
                                                  ('relu1', torch.nn.ReLU()),
                                                  ('drop1', torch.nn.Dropout(0.2)),
                                                  ('fc2', torch.nn.Linear(1024, 512)),
                                                  ('relu2', torch.nn.ReLU()),
                                                  ('drop2', torch.nn.Dropout(0.2)),
                                                  ('fc3', torch.nn.Linear(512, NUM_CLASSES)),
                                                  ('output', torch.nn.LogSoftmax(dim=1))
                                                  ]))
    # Override classifier in model with our custom FCL
    model.fc = classifier
    # Disable gradient in the parameters of the model, since we don't want to train it again
    for param in model.parameters():
        param.requires_grad = False
    # Load weights from file
    if device == torch.device('cpu'):
        model.load_state_dict(torch.load(pt_model_file_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(pt_model_file_path))
    # Transfer model to device to use
    model.to(device)
    return model
```

<h5>Pre-process an image before it can be passed through the model</h5>

```python
def pre_process_image(image):
    # Convert from opencv image array to pillow image
    image = Image.fromarray(image)
    # Perform resizing with pillow since this is how it was performed during training
    image = image.resize((310, 310), Image.BILINEAR)
    # Define necessary transforms to align with what the model expects at the input
    test_transforms = transforms.Compose([transforms.CenterCrop(299),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    return test_transforms(image).unsqueeze(0)
```

<h5>Return prediction from provided image</h5>

```python
def get_prediction(model, device, image):
    # Apply pre-processing to image
    tensor_img = pre_process_image(image)
    # Pass through model and obtain log soft max predictions
    logps = model.forward(tensor_img.to(device))
    # Obtain max to determine predicted category
    logsoft_prop, predictions = torch.max(logps, 1)
    # Obtain name of predicted class
    predicted_class = classes_names[predictions.item()]
    # And its probability
    prob = torch.exp(logsoft_prop).item()
    return 100 * prob, predicted_class
```

<h5>Returns a GStreamer pipeline for capturing from the CSI camera. Defaults to 1280x720 @ 60fps, flips the image by setting the flip_method (most common values: 0 and 2), display_width and display_height determine the size of the window on the screen</h5>

```python
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
 ````

<h5>Core function that performs predictions of kitchen utensils based on trained model</h5>

```python
def utensils_predictor():
	device = get_runtime_device()
	print("Using device {}".format(device))
	
	# Load model
	model = get_trained_model(MODEL_FILE_PATH, device)
	# Since we are using our model only for inference, switch to `eval` mode:
	model.eval()

	# Setup camera acquisition
	print("Setting up video capture...")
	cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)

	print("Starting Loop...")

	# Now run live acquisition and prediction
	start_time = time.time()
	
	if cap.isOpened():
		while True:
			# Time for FPS calculation
			loop_time = time.time() - start_time
			start_time = time.time()
			fps = 1 / loop_time

			# Capture new frame
			ret, frame = cap.read()

			# Perform prediction
			score_class, pred_class = get_prediction(model, device, frame)

			# When not very certain, better say we don't know
			if score_class < SCORE_THRESHOLD:
			    pred_class = 'Nothing'

			# Overlay on the image the prediction, the score and the FPS
			cv2.putText(frame, pred_class, (X_CENTER, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
				    (255, 255, 255), 3)
			cv2.putText(frame, '(SCORE = %.2f)' % score_class, (X_CENTER, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
				    (255, 255, 255), 2)
			cv2.putText(frame, '(FPS = %.2f)' % fps, (X_CENTER, 140), cv2.FONT_HERSHEY_SIMPLEX, 1,
				    (255, 255, 255), 2)
			cv2.imshow('Kitchen Utensils Detector', frame)

			# Quit when ESC or q are pressed
			if cv2.waitKey(20) & 0xFF == ord('q'):
				break
		cap.release()
		cv2.destroyAllWindows()
	else:
        	print("Unable to open camera")

if __name__ == "__main__":
    utensils_predictor()
```

After running the Python script, the camera will begin to capture and analyse the objects placed in front of it. If the object is not part of the 10 categories, "Nothing" should be displayed on screen. On the other hand, if the object is part of the 10 categories, the corresponding name should appear on-screen, along with a prediction percentage.

<h3>Relevant links</h3>

:page_facing_up: [Raspberry Pi Camera Module v2](https://www.raspberrypi.org/products/camera-module-v2/)

:page_facing_up: [NVIDIA Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit/)

:page_facing_up: [Save 1GB of Memory! Use LXDE on your Jetson](https://www.jetsonhacks.com/2020/11/07/save-1gb-of-memory-use-lxde-on-your-jetson/)

:page_facing_up: [PyTorch](https://pytorch.org/)

:page_facing_up: [OpenCV](https://opencv.org/)

:page_facing_up: [Edinburgh Kitchen Utensils dataset](https://homepages.inf.ed.ac.uk/rbf/UTENSILS/)

:page_facing_up: [Download All Images](https://chrome.google.com/webstore/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm?hl=en)
