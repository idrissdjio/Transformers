import torch
from robotic_transformer_pytorch import MaxViT, RT1
from tqdm import tqdm

vit = MaxViT(
    num_classes = 1000,
    dim_conv_stem = 64,
    dim = 96,
    dim_head = 32,
    depth = (2, 2, 5, 2),
    window_size = 6,
    mbconv_expansion_rate = 4,
    mbconv_shrinkage_rate = 0.25,
    dropout = 0.1
)

model = RT1(
    vit = vit,
    num_actions = 11,
    depth = 6,
    heads = 8,
    dim_head = 64,
    cond_drop_prob = 0.2
)

video = torch.randn(2, 3, 6, 128, 128)
data = torch.load('data_tensor.pt')
print(data.size())  #torch.Size([162, 3, 6, 224, 224])

# add 162 same instructions

instructions = []
string = 'pick up the book in the middle and place it on the cabinet shelf'
for i in range(data.size(0)):
    instructions.append(string)

print("Model is being trained")    

train_logits = model(data, instructions) # (2, 6, 11, 256) # (batch, frames, actions, bins)

print("About to end training")

# after much training
model.eval()
eval_logits = model(video, instructions, cond_scale = 3.) # classifier free guidance with conditional scale of 3
print("ended training")
