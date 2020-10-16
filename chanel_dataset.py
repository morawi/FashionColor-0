# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:30:27 2020

@author: malrawi
"""



from torch.utils.data import Dataset # Dataset class from PyTorch
from PIL import Image# PIL is a nice Python Image Library that we can use to handle images
import torchvision.transforms as transforms # torch transform used for computer vision applications

import torch
import glob


''' Class names
0   background
1	bag	
2	belt	
3	boots	
4	footwear	
5	outer	coat/jacket/suit/blazers/cardigan/sweater/Jumpsuits/Rompers/vest
6	dress	dress/t-shirt dress
7	sunglasses	
8	pants	pants/jeans/leggings
9	top	top/blouse/t-shirt/shirt
10	shorts	
11	skirt	
12	headwear	
13	scarf & tie	

'''

def get_class_names_md(): 
    ModaNet_class_names = ['background', 'bag', 'belt', 'boots', 'footwear',
                           'outer', 'dress', 'sunglasses', 'pants', 'top',
                           'shorts', 'skirt', 'headwear', 'scrtf-tie' ]
    return ModaNet_class_names


def get_class_names(): 
    # names ordered according to label id, 0 for background and 59 for wedges    
    class_names = ['background',  'accessories',  'bag',  'belt',  'blazer',
 'blouse',  'bodysuit',  'boots',  'bra',  'bracelet',  'cape',  'cardigan',
 'clogs', 'coat',  'dress', 'earrings', 'flats', 'glasses', 'gloves', 'hair',
 'hat', 'heels', 'hoodie', 'intimate', 'jacket', 'jeans', 'jumper', 'leggings',
 'loafers', 'necklace', 'panties', 'pants', 'pumps', 'purse', 'ring', 'romper',
 'sandals', 'scarf', 'shirt', 'shoes', 'shorts', 'skin', 'skirt', 'sneakers',
 'socks', 'stockings', 'suit', 'sunglasses', 'sweater', 'sweatshirt', 'swimwear',
 't-shirt', 'tie', 'tights', 'top', 'vest', 'wallet', 'watch', 'wedges']
    
    return class_names

    


class ChenelDataset(Dataset):
    def __init__(self, root='C:/MyPrograms/Data/ChanelFashion/', 
                 transforms_=None, HPC_run=False, model_nm='ModaNet' ):
        
        if HPC_run:
            root = '/home/malrawi/MyPrograms/Data/Modanet'        
                        
        if transforms_ != None:
            self.transforms = transforms.Compose(transforms_) # image transform
        else: self.transforms=None               
              
        
        self.path2images = root
        self.files = sorted(glob.glob(root + "/*.jpg") )  
                          
        if model_nm=='ModaNet':
            self.class_names = get_class_names_md()  
        else:  
            self.class_names = get_class_names()
              
    
    def __getitem__(self, index):    
        img_name= self.files[index]                              
        image = Image.open(img_name)
        
        if self.transforms != None:
            image = self.transforms(image)                
        
        return image, img_name
     

    def __len__(self): # this function returns the length of the dataset, the source might not equal the target if the data is unaligned
        return len(self.files)
    
    def number_of_classes(self):
        return(len(self.class_names)) # this should do

# x_data = ChenelDataset(transforms_=None)
# # # im, tg = x_data[0] # [12839]
# # for idx in range(len(x_data)):
# #     # print(idx,',', end='')
# aa, name = x_data[10]

    
