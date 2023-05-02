from torchvision import transforms
import numpy as np 
from skimage.color import rgb2lab, rgb2gray
from dataloader.process import *
import cv2
class Color():
    def __init__(self, size):
        self.size = size
        # self.encode_layer = NNEncLayer()
        # self.boost_layer = PriorBoostLayer()
        # self.mask_layer = NonGrayMaskLayer()
        
    def __call__(self, img) :
        ## whether need to normalize ?
        img = cv2.resize(img, (self.size, self.size))
        img_lab = rgb2lab(img)
        img_lab = img_lab.astype('float32')
        img_gray = img_lab[:,:,0:1] ## notes, 0:1 rather than 0 to keep the third dim 
        img_ab = img_lab[:,:,1:3]
        return img_gray, img_ab
        

class SR():
    def __init__(self, size) -> None:
        self.resize = transforms.Resize(size)
        
    def __call__(self, img):
        img = self.resize(img)
        return img
    
    
class JigSaw():
    def __init__(self, size, permu_path) -> None:
        self.resize = transforms.Resize(size)
        self.permus = np.load(permu_path)
        self.permus_len = self.permus.shape[0]
        
    def __call__(self, img):
        img = self.resize(img)
        s = float(img.size[0]) / 3
        a = int(s / 2)
        tiles = [None] * 9
        for n in range(9):
            i = n / 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())
            # Normalize the patches indipendently to avoid low level features shortcut
            # m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
            # s[s == 0] = 1
            # norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
            # tile = norm(tile)
            tiles[n] = tile
        order = np.random.randint(self.permus_len)
        data = [tiles[self.permus[order][t]] for t in range(9)]

        return data, int(order)
        