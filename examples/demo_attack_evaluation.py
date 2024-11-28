import torch
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the backdoor model
state_dict = torch.load("backdoor_model.pth")
backdoor_model = models.resnet18(weights=None)
backdoor_model.fc = torch.nn.Linear(backdoor_model.fc.in_features, 10)
backdoor_model.load_state_dict(state_dict)
backdoor_model.to(device)

# load poison test set
poison_testset = torch.load("image_badnet_poison_test_set.pt")
testloader = torch.utils.data.DataLoader(poison_testset, batch_size=32, shuffle=False)

backdoor_model.eval()
robustness = 0
success = 0
total = 0
with torch.no_grad():
    for inputs, labels, is_poison, pre_labels in testloader:
        inputs, labels, pre_labels = (
            inputs.to(device),
            labels.to(device),
            pre_labels.to(device),
        )
        outputs = backdoor_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        robustness += (predicted == pre_labels).sum().item()
        success += (predicted == labels).sum().item()

print(
    f"Robust Accuracy of the model on the test images: {100 * robustness / total:.2f}%"
)
print(
    f"Attack Success Rate of the model on the test images: {100 * success / total:.2f}%"
)
