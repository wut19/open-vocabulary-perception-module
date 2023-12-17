from vild.vild import detect
from meshrecon.rgbd2pos import estimate_pos
from segment.segment import segmentation
import os
import numpy as np


def get_position(category_name, obs_path='/home/wutong/ComputerVision/observations', save_path='/home/wutong/ComputerVision/results'):
    
    image_path = os.path.join(obs_path,'images')
    rgbd_path = os.path.join(obs_path, 'rgbds')
    num_camera = 2
    os.makedirs(save_path, exist_ok=True)
    masks = []
    rgbs = []
    depths = []
    for i in range(num_camera):
        bbox = detect(os.path.join(image_path,f'{i+1}.png'),category_name=category_name,save_path=os.path.join(save_path,f'vild{i+1}.png')).tolist()
        mask = segmentation(os.path.join(image_path,f'{i+1}.png'),bbox=bbox,save_path=os.path.join(save_path,f'sam{i+1}.png'))
        masks.append(mask)
        rgb = np.load(os.path.join(rgbd_path,f'rgb{i+1}.npy'))
        depth = np.load(os.path.join(rgbd_path,f'depth{i+1}.npy'))
        rgbs.append(rgb)
        depths.append(depth)
    rgbs = np.stack(rgbs,axis=0)
    depths = np.stack(depths,axis=0)
    masks = np.stack(masks, axis=0)
    cam_params = np.load(os.path.join(obs_path,'cam_params.npy'),allow_pickle=True).item()
    obs = {
        'rgb': rgbs,
        'depth': depths,
        'mask': masks,
    }
    pos = estimate_pos(obs, cam_params)
    return pos

if __name__ == "__main__":
    print(get_position('tomato'))