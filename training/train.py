import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.convnext_model import ConvNeXtClassifier
from torch.optim import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"

# transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ใช้ dataset ตัวอย่างก่อน
train_dataset = datasets.FakeData(
    size=500,
    image_size=(3,224,224),
    num_classes=2,
    transform=transform
)

test_dataset = datasets.FakeData(
    size=100,
    image_size=(3,224,224),
    num_classes=2,
    transform=transform
)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32)

model = ConvNeXtClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(),lr=1e-4)

# training
for epoch in range(3):

    model.train()
    total_loss = 0

    for images,labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print("Epoch:",epoch,"Loss:",total_loss)

# evaluation
correct = 0
total = 0

model.eval()

with torch.no_grad():

    for images,labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _,predicted = torch.max(outputs,1)

        total += labels.size(0)
        correct += (predicted==labels).sum().item()

accuracy = 100*correct/total

print("Accuracy:",accuracy)
