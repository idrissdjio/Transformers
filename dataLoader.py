import h5py
import torch

# Function to recursively print the size of datasets in an HDF5 file
def print_dataset_sizes(obj, prefix=''):
    if isinstance(obj, h5py.Dataset):
        print(f'{prefix}{obj.name}: {obj.shape}')
    elif isinstance(obj, h5py.Group):
        for key in obj.keys():
            print_dataset_sizes(obj[key], prefix + '  ')

# Open the HDF5 file
file_path = 'RW8_put_the_bread_on_the_white_plate_demo.hdf5'
with h5py.File(file_path, 'r') as hdf5_file:
    print_dataset_sizes(hdf5_file)


# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import h5py
# import numpy as np

# class CustomImageDataset(Dataset):
#     def __init__(self, hdf5_path, transform=None):
#         self.hdf5_path = hdf5_path
#         self.file = h5py.File(self.hdf5_path, 'r')
#         self.dataset = self.file['RW8_put_the_bread_on_the_white_plate_demo.hdf5']  # Replace 'dataset_name' with the actual name of your dataset
#         self.transform = transform

#     def __len__(self):
#         # Assuming each sample consists of 6 images, adjust accordingly
#         return len(self.dataset) // 6

#     def __getitem__(self, idx):
#         # Load 6 images for the given index
#         images = []
#         for i in range(6):
#             image = self.dataset[idx * 6 + i]
#             # Convert image data to PIL Image for transformations
#             image = Image.fromarray(image.astype('uint8'), 'RGB')
#             if self.transform:
#                 image = self.transform(image)
#             images.append(image)
        
#         # Stack images to create a single tensor
#         images = torch.stack(images)
#         return images

# # Define transformations, including resizing to 300x300
# transform = transforms.Compose([
#     transforms.Resize((300, 300)),
#     transforms.ToTensor(),
# ])

# # Initialize the dataset
# dataset = CustomImageDataset('RW8_put_the_bread_on_the_white_plate_demo.hdf5', transform=transform)

# # Example: Load the dataset using DataLoader
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# # Iterate through the DataLoader (example)
# for images in dataloader:
    # print("Batch shape:", images.shape)
    # break