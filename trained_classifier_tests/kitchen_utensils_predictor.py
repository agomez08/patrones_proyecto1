# Perform necessary imports
import time
import torch
from collections import OrderedDict
from torchvision import transforms, models
from PIL import Image
import cv2


# General constants for the code
classes_names = ['BOTTLE_OPENER', 'DINNER_FORK', 'DINNER_KNIFE', 'FISH_SLICE', 'KITCHEN_KNIFE', 'LADLE',
                 'POTATO_PEELER', 'SPATULA', 'SPOON', 'WHISK']
NUM_CLASSES = len(classes_names)
MODEL_FILE_PATH = 'tl_inceptionv3_model.pt'
SCORE_THRESHOLD = 60
# Camera resolution parameters
X_RES = 1280
Y_RES = 720
X_CENTER = X_RES // 2


# Define functions
def get_runtime_device():
    """Return torch device to use. It will return cuda if a supported GPU is available"""
    # Determine device to use (to use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_trained_model(pt_model_file_path, device):
    """Return trained model for evaluation."""
    # Define already trained Inception-v3 model for learning transfer
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


def pre_process_image(image):
    """Pre-process an image before it can be passed through the model."""
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


def get_prediction(model, device, image):
    """Return prediction from provided image."""
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


def utensils_predictor():
    """Core function that performs predictions of kitchen utensils based on trained model."""
    device = get_runtime_device()
    print("Using device {}".format(device))
    # Load model
    model = get_trained_model(MODEL_FILE_PATH, device)
    # Since we are using our model only for inference, switch to `eval` mode:
    model.eval()

    # Setup camera acquisition
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, X_RES)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Y_RES)

    # Now run live acquisition and prediction
    start_time = time.time()
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


if __name__ == "__main__":
    utensils_predictor()
