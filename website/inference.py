import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class TSignDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(94208, 128)
        self.fc2 = nn.Linear(128, 4)

        self.loss_fn = nn.NLLLoss()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = x.flatten(start_dim=0)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        output = F.log_softmax(x, dim=0)
        return output

prediction_dict = {
    0: "Crosswalk",
    1: "Traffic Light",
    2: "Stop",
    3: "Speed Limit"
}

def get_prediction(model, frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert to PIL Image first
        transforms.Resize((300, 400)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
    ])

    frame = transform(frame)
    outputs = model(frame)
    print(torch.exp(outputs))
    _, predicted = torch.max(outputs, 0)
    return prediction_dict[predicted.item()]

