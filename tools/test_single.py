import torch, timm
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path

IMG_SIZE = 224
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Load class map
idx_to_species = json.load(open("models/class_maps/idx_to_species.json"))

# Load model
ckpt = "models/checkpoints/plant_id_resnet18_best.pth"
num_classes = len(idx_to_species)
model = timm.create_model("resnet18", pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

# Test image
img = Image.open("test_leaf.jpg").convert("RGB")
x = TRANSFORM(img).unsqueeze(0)
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    top5 = torch.topk(probs, 5)

for score, idx in zip(top5.values[0], top5.indices[0]):
    print(f"{idx_to_species[str(int(idx.item()))]}: {score.item()*100:.2f}%")
