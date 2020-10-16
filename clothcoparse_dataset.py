import glob
import os
import scipy.io as sio
from torch.utils.data import Dataset # Dataset class from PyTorch
from PIL import Image, ImageChops # PIL is a nice Python Image Library that we can use to handle images
import numpy as np

# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html


def get_clothCoParse_class_names(): 
    # Item names ordered according to label id, 0 for background and 59 for wedges
    # A dictionary that contains each label name, and the possible (max) number of colors
    # The number of colors can be changed, these are used as upper bounds when estimating 
    # the number of colors in the item of the label
       
    min_ = 3  # min_num_colors
    low_ = 5  # low_num_colors
    mid_ = 8 # mid_num_colors
    max_ = 17  # max_num_colors
    
    
    class_names = {'background': 0,  # this will be ignored
                   'accessories': low_,  
                   'bag': mid_,  
                   'belt': min_,  
                   'blazer': max_,
                   'blouse': max_,  
                   'bodysuit': max_,  
                   'boots': mid_,  
                   'bra': mid_,  
                   'bracelet': low_,  
                   'cape': max_,  
                   'cardigan': max_,
                   'clogs': mid_, 
                   'coat': max_,  
                   'dress': max_, 
                   'earrings': min_, 
                   'flats': low_, 
                   'glasses': min_, 
                   'gloves': low_, 
                   'hair': mid_, # I had to add this up, because sometimes women's hair has a lot of colors 
                   'hat': mid_, 
                   'heels': min_, 
                   'hoodie': max_, 
                   'intimate': max_, 
                   'jacket': max_, 
                   'jeans': max_, 
                   'jumper': max_, 
                   'leggings': max_,
                   'loafers': low_, 
                   'necklace': min_, 
                   'panties':min_, 
                   'pants': max_, 
                   'pumps': low_, # there are multicolor pumps, but since this item will appear small in the image, it will be hard to get the colors
                   'purse': mid_, 
                   'ring': min_, 
                   'romper': max_,
                   'sandals': low_, 
                   'scarf': max_, 
                   'shirt': max_, 
                   'shoes': low_, 
                   'shorts': max_, 
                   'skin': min_,   # skin has one color
                   'skirt': max_, 
                   'sneakers': mid_,
                   'socks': low_, 
                   'stockings': mid_, 
                   'suit': max_, 
                   'sunglasses': min_, 
                   'sweater': max_,
                   'sweatshirt':max_, 
                   'swimwear': max_,
                   't-shirt': max_,
                   'tie': max_,
                   'tights': max_, 
                   'top': max_,
                   'vest': max_,
                   'wallet': min_, 
                   'watch': min_,
                   'wedges': mid_}
    
    return class_names
        

class ImageDataset(Dataset):
    def __init__(self, root, mode="train",  HPC_run=False):
        
        self.class_names = list(get_clothCoParse_class_names().keys())
        
        if HPC_run:
            root = '/home/malrawi/MyPrograms/Data/ClothCoParse'
        
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*")) # get the source image file-names
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*")) # get the target image file-names
        
    def number_of_classes(self, opt):
        return(len(self.class_names)) # this should do
  

    def __getitem__(self, index):   
                
        annot = sio.loadmat(self.files_B[index % len(self.files_B)])
        mask = annot["groundtruth"]
        image_A = Image.open(self.files_A[index % len(self.files_A)]) # read the image, according to the file name, index select which image to read; index=1 means get the first image in the list self.files_A
       
        # instances are encoded as different colors
        obj_ids = np.unique(mask)[1:] # first id is the background, so remove it     
        masks = mask == obj_ids[:, None, None] # split the color-encoded mask into a set of binary masks

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)                       
        masked_img = []; labels =[]
        for i in range(num_objs):
            img = ImageChops.multiply(image_A, Image.fromarray(255*masks[i]).convert('RGB') )
            masked_img.append(np.array(img, dtype='uint8'))                               
            labels.append(self.class_names[obj_ids[i]])
                              
        image_id = index
        fname = os.path.basename(self.files_A[index % len(self.files_A)])          
        
        return image_A, masked_img, labels, image_id, masks, fname
     

    def __len__(self): # this function returns the length of the dataset, the source might not equal the target if the data is unaligned
        return len(self.files_B)

