import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models.video import R3D_18_Weights, r3d_18

# Hyperparameters
batch_size = 4
num_epochs = 20
learning_rate = 0.001
num_classes = 51  # HMDB51 has 51 classes

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model with pre-trained weights
weights = R3D_18_Weights.KINETICS400_V1
model = r3d_18(num_classes=num_classes).to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define transformations for the dataset
transform = transforms.Compose(
    [
        transforms.Resize((128, 171)),
        transforms.RandomCrop((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
        ),
    ]
)
from utils.data import load_dataset
from torchvision.datasets import HMDB51
from configs.settings import BASE_DIR

ds_dir = BASE_DIR / "data" / "hmdb51"
# Load HMDB51 dataset
train_dataset = HMDB51(
    root=ds_dir / "data",
    annotation_path=ds_dir / "test_train_splits",
    frames_per_clip=16,
    train=True,
    transform=transform,
    # num_workers=4,
    # output_format="TCHW",
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for videos, audios, labels in train_loader:
        print(videos.shape)
        # videos = videos.permute(0, 2, 1, 3, 4)
        videos, labels = videos.to(device), labels.to(device)

        # Forward pass
        outputs = model(transform(videos))
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}"
    )

# Save the model
torch.save(model.state_dict(), "r3d_18_hmdb51.pth")

print("Training complete!")
