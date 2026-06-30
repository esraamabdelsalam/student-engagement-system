import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import gdown
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/engagement/engagement.pth"
FILE_ID = "1vyaQTQfc12nSjlVwiakfDcmBE-AW79Og"

SEQUENCE_LENGTH = 16

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

if not os.path.exists(MODEL_PATH):
    gdown.download(
        f"https://drive.google.com/uc?id={FILE_ID}",
        MODEL_PATH,
        quiet=False
    )


class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.cnn.fc = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = x.view(B, T, 2048)

        _, (h_n, _) = self.lstm(x)

        x = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(x)


model = CNN_BiLSTM()
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) else state)

model.to(device)
model.eval()

idx_to_class = {
    0: "Disengaged",
    1: "Engaged"
}


def predict_engagement_sequence(frames):
    if len(frames) != SEQUENCE_LENGTH:
        return None

    x = torch.stack(frames).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return {
        "engagement": idx_to_class[int(pred.item())],
        "confidence": float(conf.item())
    }