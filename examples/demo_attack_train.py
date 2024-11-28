import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
poisonset = torch.load("image_badnet_poison_train_set.pt")
backdoor_trainloader = torch.utils.data.DataLoader(
    poisonset, batch_size=32, shuffle=True
)

# define your model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model.to(device)

# define your criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels, is_poison, pre_labels in tqdm(
        backdoor_trainloader, desc="training"
    ):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(backdoor_trainloader):.4f}"
    )

torch.save(model.state_dict(), "backdoor_model.pth")
