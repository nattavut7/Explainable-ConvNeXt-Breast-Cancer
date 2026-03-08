
import torch
from torch.utils.data import DataLoader
from models.convnext_model import ConvNeXtClassifier
from utils.dataset_loader import BreakHisDataset
from torch.optim import Adam
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ConvNeXtClassifier().to(device)

optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

criterion = torch.nn.CrossEntropyLoss()

# Placeholder dataset
train_dataset = []

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)

for epoch in range(100):

    model.train()

    total_loss = 0

    for images,labels in tqdm(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print("Epoch:",epoch,"Loss:",total_loss)
