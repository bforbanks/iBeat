import torch as t


def test_net(loaded_net, loader, device, criterion):
    loaded_net.eval()
    test_loss = 0
    predictions = []
    actuals = []
    with t.no_grad():
        for data, target in loader:
            # Send data to GPU
            data, target = data.to(device), target.to(device)

            # Forward pass, get predictions
            output = loaded_net(data)

            # Calculate batch loss
            test_loss += criterion(output.view(-1, 430), target).item()

            # Collect predictions and actuals
            predictions.extend(output.squeeze().tolist())
            actuals.extend(target.tolist())

        return test_loss / len(loader), predictions, actuals
