import numpy as np
from tqdm import tqdm
def isWhitePatch_S(patch, rgbThresh=220, percentage=0.2):
    num_pixels = patch.shape[0] * patch.shape[1]
    return True if np.all(patch > rgbThresh, axis=2).sum() > num_pixels * percentage else False

def isBlackPatch_S(patch, rgbThresh=20, percentage=0.05):
    num_pixels = patch.shape[0] * patch.shape[1]
    return True if np.all(np.array(patch) < rgbThresh, axis=2).sum() > num_pixels * percentage else False

def slicing(images, coords, size_out):
    size_in = images.shape[1]
    resize_factor = int(size_in / size_out)
    steps = [i * (size_in // resize_factor) for i in range(resize_factor)]
    new_patches, new_coords = [], []
    for it, img_patch in enumerate(tqdm(images)):
        coords_patch = coords[it]
        for x_step in steps:
            for y_step in steps:
                resized_patch = img_patch[x_step:x_step + size_out, y_step:y_step + size_out,:]
                if isWhitePatch_S(resized_patch, rgbThresh=220, percentage=0.5):
                    continue
                if isBlackPatch_S(resized_patch, rgbThresh=20, percentage=0.05):
                    continue
                resized_coords = [coords_patch[0]+x_step, coords_patch[1]+y_step]
                new_patches.append(resized_patch)
                new_coords.append(resized_coords)

    new_patches = np.stack(new_patches)
    new_coords = np.stack(new_coords)
    return new_patches, new_coords