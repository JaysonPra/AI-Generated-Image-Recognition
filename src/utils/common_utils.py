import torch

def run_epoch(model, loader, criterion, device, optimizer=None):
    """Runs the Training / Testing logic

    Args:
        model (torch.nn.Module): The Neural Network model
        loader (torch.utils.data.DataLoader): The DataLoader for the Images 
        criterion (torch.nn.Module): The loss function used to evaluate the model's performance
        device (torch.device): The device used to train/test the model
        optimizer (torch.optim.Optimizer, optional): The optimizer used for weight updates. Defaults to None.

    Returns:
        tuple: The average loss and accuracy of the run
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    
    return running_loss / len(loader), 100. * correct / total