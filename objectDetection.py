import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.optim as optim

# Custom Dataset for PPE Detection
class PPEKitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(os.path.join(root_dir, 'images'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'images', self.images[idx])
        image = Image.open(img_name).convert("RGB")  # Ensure image is in RGB format
        label_name = os.path.join(self.root_dir, 'labels', self.images[idx].replace('.jpg', '.txt'))
        
        # Load your labels here
        labels = self.load_labels(label_name)

        if self.transform:
            image = self.transform(image)

        return image, labels

    def load_labels(self, label_file):
        boxes = []
        labels = []
        if not os.path.exists(label_file):
            return {'boxes': torch.tensor(boxes, dtype=torch.float32), 'labels': torch.tensor(labels)}

        with open(label_file, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                x_center, y_center, width, height = x_center * 256, y_center * 256, width * 256, height * 256  # Adjust this if image size is different
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                boxes.append([x1, y1, x2, y2])
                labels.append(int(class_id))

        return {'boxes': torch.tensor(boxes, dtype=torch.float32), 'labels': torch.tensor(labels, dtype=torch.int64)}

# Custom collate function to handle variable-sized inputs
def collate_fn(batch):
    return tuple(zip(*batch))

# Define your transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create datasets
train_dataset = PPEKitDataset('/home/tst/Desktop/Object Detection/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

val_dataset = PPEKitDataset('/home/tst/Desktop/Object Detection/valid', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Load a pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
num_classes = 2  # 1 class (PPE) + background

# Replace the pre-trained head with a new one
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Set device and optimizer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()

        print(f'Epoch: {epoch + 1}')
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        print(f'Epoch: {epoch + 1}, Loss: {losses.item()}')
    

    # Save the model after each epoch
    model_save_path = f'ppe_detection_model_epoch_{epoch + 1}_loss_{losses.item():.4f}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

# Example Inference on New Image
def inference(image_path):
    model.eval()
    model.load_state_dict(torch.load('ppe_detection_model_epoch_10_loss_0.1234.pth'))  # Adjust this path
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image)

    # Process and visualize the prediction
    return prediction

# Example usage
pred = inference('/path/to/new/image.jpg')
print(pred)
