import torch

def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)

            outputs = model(image)
            loss = criterion(outputs, targets)

            if is_train:
                optimzer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    
    return running_loss / len(loader), 100. * correct / total