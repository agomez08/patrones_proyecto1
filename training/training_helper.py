"""This module implements helper functions to assist with the training process."""
import numpy as np
import torch


def train_batch(optimizer, model, criterion, inputs, target):
    """Perform one iteration of training over a batch of the dataset."""
    # Start batch with zero gradients
    optimizer.zero_grad()

    # Forward pass inputs through model
    logps = model.forward(inputs)
    # Calculate loss for predictions
    loss = criterion(logps, target)
    # Do back propagation to update gradient
    loss.backward()
    # Let the optimizer update coefficients
    optimizer.step()

    # Calculate and return batch loss
    batch_loss = loss.item() * inputs.size(0)

    return batch_loss


def train_epoch(optimizer, model, criterion, train_loader, device):
    """Perform one epoch iteration of training."""
    # Switch model to training mode
    epoch_loss = 0.0
    model.train()
    # Run through each of the batches
    for batch_idx, (inputs, target) in enumerate(train_loader):
        # print("DEBUG: Starting TRAINING batch {}".format(batch_idx))
        # Mode tensors to GPU if available
        inputs, target = inputs.to(device), target.to(device)

        # Perform training over batch and update training loss
        batch_loss = train_batch(optimizer, model, criterion, inputs, target)
        epoch_loss += batch_loss

    # Return average epoch loss
    return epoch_loss / len(train_loader.sampler)


def validate_batch(model, criterion, inputs, target):
    """Perform one iteration of training over a batch of the dataset."""
    # Forward pass inputs through model
    logps = model.forward(inputs)
    # Calculate loss for predictions
    loss = criterion(logps, target)

    # Calculate and return batch loss
    validation_loss = loss.item() * inputs.size(0)

    return validation_loss


def validate_epoch(model, criterion, valid_loader, device):
    """Perform one epoch iteration of validation."""
    # Switch model to validation mode
    epoch_loss = 0.0
    model.eval()
    for batch_idx, (inputs, target) in enumerate(valid_loader):
        # print("DEBUG: Starting VALIDATION batch {}".format(batch_idx))
        # Mode tensors to GPU if available
        inputs, target = inputs.to(device), target.to(device)

        # Perform validation over batch and update validation loss
        batch_loss = validate_batch(model, criterion, inputs, target)
        epoch_loss += batch_loss

    # Return average epoch loss
    return epoch_loss / len(valid_loader.sampler)


def test_eval_batch(model, criterion, inputs, target):
    """Perform one iteration of testing over a batch of the dataset and report loss and predictions."""
    # Forward pass inputs through model
    logps = model.forward(inputs)
    # Calculate loss for predictions
    loss = criterion(logps, target)

    # Use logps probabilities to determine the prediction (prediction is class with maximum logps)
    _, predictions = torch.max(logps, 1)

    # Calculate batch loss
    validation_loss = loss.item() * inputs.size(0)

    return validation_loss, predictions


def test_eval(num_classes, model, criterion, test_loader, device):
    """Perform evaluation of results for trained model over the testing portion of the dataset."""
    # track test loss
    test_loss = 0.0
    class_correct_list = list(0. for _ in range(num_classes))
    class_total_list = list(0. for _ in range(num_classes))

    model.eval()
    # iterate over test data
    for batch_idx, (inputs, target) in enumerate(test_loader):
        # print("DEBUG: Starting TESTING batch {}".format(batch_idx))

        # Mode tensors to GPU if available
        inputs, target = inputs.to(device), target.to(device)

        # Perform evaluation over batch and update test loss
        batch_loss, predictions = test_eval_batch(model, criterion, inputs, target)
        test_loss += batch_loss

        # Compare predictions against target
        correct_tensor = predictions.eq(target.data.view_as(predictions))
        if torch.cuda.is_available():
            # Bring back to CPU if GPU was being used
            correct = np.squeeze(correct_tensor.cpu().numpy())
        else:
            correct = np.squeeze(correct_tensor.numpy())
        # For each of the possible targets
        for i in range(target.size(0)):
            # Save the number of elements classified correctly and the total number of elements
            label = target.data[i]
            class_correct_list[label] += correct[i].item()
            class_total_list[label] += 1

    # Determine average testing loss
    test_loss_avg = test_loss / len(test_loader.dataset)

    # Determine results for each of the classes
    classes_results = []
    for i in range(num_classes):
        class_correct = np.sum(class_correct_list[i])
        class_total = np.sum(class_total_list[i])
        class_accuracy = 100 * class_correct / class_total
        class_results = {'class_accuracy': class_accuracy, 'class_correct': class_correct, 'class_total': class_total}
        classes_results.append(class_results)

    # Calculate global accuracy
    global_accuracy = 100. * np.sum(class_correct_list) / np.sum(class_total_list)

    # Put all results in dictionary and return it
    results = {'test_loss_avg': test_loss_avg, 'classes_results': classes_results, 'global_accuracy': global_accuracy}

    return results
