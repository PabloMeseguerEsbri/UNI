import argparse
import os
import random
import sys

import h5py
import numpy as np
import timm
import torch
from PIL import Image
from huggingface_hub import login
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm

login(token='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')  # login with your User Access Token, found at https://huggingface.co/settings/tokens

parser = argparse.ArgumentParser(description="Feature extraction w/ UNI")
parser.add_argument('--folder_path', type=str)
parser.add_argument('--folder_save', type=str)
parser.add_argument('--data_loading', type=str, choices=["PATH", "CLAM", "AI4SKIN"])
parser.add_argument('--reverse', action='store_true')
parser.add_argument('--random', action='store_true')
parser.add_argument('--slicing', action='store_true')
parser.add_argument('--slicing_shape', type=int, choices=[512,256])
args = parser.parse_args()

folder_path = args.folder_path
folder_save = args.folder_save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model creation
model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True).to(device)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()

# Data loading
if args.data_loading == "CLAM":
    list_wsi = os.listdir(folder_path)
elif args.data_loading == "PATH":
    list_wsi = [item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))]
elif args.data_loading == "AI4SKIN":
    patches = pd.read_csv("./assets/data/csv/AI4SKIN_patches.csv")['images'].values
    list_wsi = pd.read_csv("./assets/data/csv/AI4SKIN_WSI.csv", delimiter=",")['WSI'].values
    patches_ids = np.array([patch[:9] for patch in patches])

list_wsi.reverse() if args.reverse else None
random.shuffle(list_wsi) if args.random else None

for name_wsi in list_wsi:
    if args.data_loading == "CLAM":
        name_wsi = name_wsi[:-3]
    file_npy = os.path.join(folder_save, name_wsi + ".npy")
    if os.path.isfile(file_npy):
        continue
    print(name_wsi)
    sys.stdout.flush()

    if args.data_loading == "CLAM":
        with h5py.File(os.path.join(folder_path, name_wsi + ".h5"), 'r') as file:
            images = file['imgs'][:]
            coords = file['coords'][:]
        if args.slicing:
            images, coords = slicing(size_out=args.slicing_shape, images=images, coords=coords)  # Slicing
        images = [Image.fromarray(patch) for patch in images]

    if args.data_loading == "PATH":
        folder_wsi = os.path.join(folder_path, name_wsi)
        images_list = os.listdir(folder_wsi)
        images_list = [file for file in images_list if file.lower().endswith('.png')]
        images = [Image.open(os.path.join(folder_wsi, patch)) for patch in tqdm(images_list) if os.path.isfile(os.path.join(folder_wsi, patch))]

    if args.data_loading == "AI4SKIN":
        img_files = patches[np.array(patches_ids) == name_wsi]  # All patches of one WSI
        if img_files.size == 0:
            print("WSI does not have any patch")
            continue
        if name_wsi.startswith("HCUV"):
            folder_wsi = args.folder_png + "Images/"
        elif name_wsi.startswith("HUSC"):
            folder_wsi = args.folder_png + "Images_Jose/"
        if img_files.size > 7500:
            img_files = img_files[:7500]
        images = [Image.open(folder_wsi + patch) for patch in tqdm(img_files) if os.path.isfile(folder_wsi + patch)]  # Data loading

    patch_embeddings = []
    for img in tqdm(images):
        img = transform(img).unsqueeze(dim=0).to(device)
        with torch.inference_mode():
            x = model(img).cpu().numpy().squeeze()
        patch_embeddings.append(x)
    patch_embeddings = np.stack(patch_embeddings)
    np.save(file_npy, patch_embeddings)