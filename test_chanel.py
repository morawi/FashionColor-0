# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:50:45 2020

@author: malrawi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:19:54 2020

@author: malrawi

"""

from models import get_model_instance_segmentation
import torch
from draw_image import instance_segmentation_api
import argparse
from chanel_dataset import ChenelDataset
from PIL import Image, ImageChops
import numpy as np
from color_extractor import ColorExtractor
from color_table import ColorTable

parser = argparse.ArgumentParser()
parser.add_argument("--path2model", type=str, default=None, help="path to the model")
parser.add_argument("--model_name", type=str, default=None, help="name of the model")
parser.add_argument("--dataset_name", type=str, default="ClothCoParse", help="name of the dataset")
cnf = parser.parse_args()

cnf.HPC_run = 0

cnf.n_cpu=0
cnf.cuda = True # this will definetly work on the cpu if it is false
# cnf.method = '3D_1D' # methods are: {'3D_1D', '3D'}
cnf.num_colors = 13 # 16 # we perhapse need to use different set of colors depending on the item, for skin should be 4
cnf.max_num_colors = cnf.num_colors
cnf.use_quantize = False
cnf.clustering_method = 'kmeans' # {'kmeans', 'fcmeans}
cnf.clsuter_1D_method='Diff'# 'MeanSift'  # {'MeanSift', 'Diff', '2nd_fcm'}

dataset_name = '' # 'ClothCoParse'
cnf.model = 'ModaNet' # {'ModaNet', 'ClothCoParse'}

chanel_data = ChenelDataset(transforms_=None, model_nm = cnf.model)

if cnf.model == 'ModaNet':
    cnf.model_name='maskrcnn_50.pth'
    cnf.path2model = 'C:/MyPrograms/saved_models/Modanet/mask_rcnn-Apr-26-at-15-58/'
else:    
    cnf.model_name='maskrcnn_700.pth'
    cnf.path2model = 'C:/MyPrograms/saved_models/ClothCoParse/mask_rcnn-Apr-8-at-2-8/' # keep bkgrnd

device = torch.device('cuda' if cnf.cuda else 'cpu')
model = get_model_instance_segmentation(chanel_data.number_of_classes()) 
print("loading model", cnf.model_name )        
model.load_state_dict(torch.load(cnf.path2model+cnf.model_name,  map_location=device ))  
model.to(device)
model.eval()


cnf.num_tsts= 1000 # inf


for i, item in enumerate(chanel_data):
    if i >= cnf.num_tsts: break
    image = item[0]    
    masks, labels = instance_segmentation_api(model, image, device, chanel_data.class_names, 
                                              threshold=0.7, rect_th=1, text_size=1, text_th=3)    
    masked_img = []
    for jj in range(len(labels)):
        masks[jj] = masks[jj].astype(dtype='uint')
        # zz[masks[jj]] = jj+1; # Image.fromarray(zz*255/(jj+1)).show()        
        img = ImageChops.multiply(image, Image.fromarray(255*masks[jj]).convert('RGB') ); # img.show()
        masked_img.append(np.array(img, dtype='uint8'))
        
    one_person_clothing_colors = ColorExtractor(masks, labels, masked_img, cnf,                                                                                                             
                                                         image_name=item[1][-10:-1]) 
    one_person_clothing_colors.pie_chart(image, fname=item[1][-10:-1], figure_size=(4, 4))

            
 
    # color_table_obj = ColorTable(dataset.class_names, cnf)
    
