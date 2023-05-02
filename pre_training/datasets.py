from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import torchvision.io as tio
from yaml import parse
from PIL import Image
import torch


def read_txt(path):
    f = open(path)
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    f.close()
    return lines

class clc_dataset_folder(Dataset):
    def __init__(self, data_root, folder, transform):
        self.data_root = data_root
        self.folder = folder
        self.transform = transform
        self.img_paths = []
        self.labels = []
        self.parse()
    
    def parse(self):
        folder_path = os.path.join(self.data_root, self.folder)
        sub_folders = next(os.walk(folder_path))[1] ### subdirectory in folder_path
        for i, sub_f in enumerate(sub_folders):
            cate = i
            for img_name in os.listdir(os.path.join(folder_path,sub_f)):
                img_path = os.path.join(folder_path, sub_f, img_name)
                self.img_paths.append(img_path)
                self.labels.append(cate)
                
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        label = torch.tensor(label).long()
        return {'img':img,'label':label, 'path': img_path}
    
    def __len__(self):
        return len(self.img_paths)
    

class ldl_dataset(Dataset):
    def __init__(self,data_root,split_txt, transform):
        self.data_root = data_root
        self.split_txt = split_txt
        self.transform = transform
        self.img_paths = []
        self.labels = []
        self.parse_ldl()
        
    def parse_ldl(self):
        lines = read_txt(os.path.join(self.data_root, self.split_txt))
        lines = lines[1:] ## get rid of head
        for line in lines:
            elems = line.split(' ')
            img_name = elems[0]
            label = elems[1:]
            label = [float(l) for l in label]
            vote_num = sum(label)
            label = [l / vote_num for l in label]
            img_path = os.path.join(self.data_root, img_name)
            self.img_paths.append(img_path)
            self.labels.append(label)
            
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        label = torch.tensor(label)
        return {'img':img,'label':label, 'path': img_path}
    
    def __len__(self):
        return len(self.img_paths)


class mll_dataset(Dataset):
    def __init__(self,data_root,split_txt, transform):
        self.data_root = data_root
        self.split_txt = split_txt
        self.transform = transform
        self.img_paths = []
        self.labels = []
        self.parse_ldl()
        
    def parse_ldl(self):
        lines = read_txt(os.path.join(self.data_root, self.split_txt))
        lines = lines[1:] ## get rid of head
        for line in lines:
            elems = line.split(' ')
            img_name = elems[0]
            label = elems[1:]
            label = [float(l) for l in label]
            img_path = os.path.join(self.data_root, img_name)
            self.img_paths.append(img_path)
            self.labels.append(label)
            
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        label = torch.tensor(label)
        return {'img':img,'label':label, 'path': img_path}
    
    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    test_transforms =  transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
])

    ### test clc folder dateaset
    data = clc_dataset_folder('./datasets/WEBEmo/WEBEmo','WEBEmo',test_transforms)
    data[99]
    loader = DataLoader(data,64)
    for data in loader:
        print()
    print()

        
            