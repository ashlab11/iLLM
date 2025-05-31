import time
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData

batch_size = 64
# 1. Setup c
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# 2. Initialize ResNet-50
model = models.resnet50().to(device)
model.train()

# 3. Create a synthetic dataset for throughput testing
#    (ImageNet-sized images, random labels)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
dataset = FakeData(
    size=10000,
    image_size=(3, 224, 224),
    num_classes=1000,
    transform=transform
)
loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

# 4. Warm-up (one batch)
for imgs, _ in loader:
    imgs = imgs.to(device)
    _ = model(imgs)
    break

# 5. Measure throughput (images/sec)
num_batches = 50
start = time.time()
for i, (imgs, _) in enumerate(loader):
    imgs = imgs.to(device)
    _ = model(imgs)
    if i + 1 == num_batches:
        break
end = time.time()

images_per_sec_m1 = (num_batches * batch_size) / (end - start)

# 6. Reference V100 performance (FP32 ResNet-50): ~1400 images/sec; DLPerf=21
ref_images_per_sec_v100 = 1400.0  # 
dlperf_v100 = 21.0

# 7. Compute DLPerf for M1
dlperf_m1 = (images_per_sec_m1 / ref_images_per_sec_v100) * dlperf_v100

print(f"macOS M1 ResNet-50 throughput: {images_per_sec_m1:.1f} images/sec")
print(f"Approximate DLPerf for M1: {dlperf_m1:.2f}")
