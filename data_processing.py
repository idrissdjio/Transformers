import pathlib
from glob import glob
import imageio.v3 as iio
from PIL import Image
import torch
from torchvision import transforms

folder_dir = pathlib.Path("./vid_000")

images = [img for img in glob(str(folder_dir / "*.png"))]

def get_image_dimensions(image_path):
    image = iio.imread(image_path)
    return image.shape

for image in images:
    dimensions = get_image_dimensions(image)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

image_tensors = []

try:
    for image in images:
        image = Image.open(image)
        tensor = transform(image)
        image_tensors.append(tensor)
except Exception as e:
    print(f"An error occurred: {e}")

image_tensors = torch.stack(image_tensors)
print(image_tensors.size())

# Using sliding window approach to split images sequentially
def sliding_window(images, window_size):
    num_images = images.size(0)
    num_channels = images.size(1)
    image_height = images.size(2)
    image_width = images.size(3)
    result = []
    for i in range(num_images - window_size + 1):
        window = images[i:i+window_size].permute(1, 0, 2, 3)
        result.append(window.unsqueeze(0))
    return torch.cat(result, dim=0)

window_size = 6
data = sliding_window(image_tensors, window_size)
print(data.size())  # torch.Size([162, 3, 6, 224, 224])


# save torch value
torch.save(data, 'data_tensor.pt')


