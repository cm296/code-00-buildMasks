import os
# from PIL import Image
# import math
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
# from random import randrange
import ProduceNewMasks as msk
import ProcessSegMask as atr
import csv
import time

#Code to produce masks of PaddedBeaconImages
local = 1
SaveFolder = 'masks-ade-4'

# def makeDict_mod_pd(p):
#     path = p['filepath']+'/'+p['filename']
#     s={}
#     s['filepath'] = path[0]
#     s['category'] = p.object[0]
#     return s

def makeDict_mod(p):
    path = p.filepath+'/'+p.filename
    s={}
    s['filepath'] = path
    s['category'] = p.object
    return s

#masktype are 'object', 'scene' or 'PaddedBeacon'
def ProduceImageMasks(masktype = [],cat = []):
    # print('Running Original Script')
    #takes model and loads the features for that image, needs path to of files in directory
    if local:
        Ade_subset = pd.read_csv("../Ade20K_labels/Ade20K_labels_local.txt")
    else:
        Ade_subset = pd.read_csv("../Ade20K_labels/Ade20K_labels_marcc.txt")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not cat:
        conditions = np.unique(Ade_subset.object)
        for c in tqdm(conditions):
            Ade_cat = Ade_subset.loc[Ade_subset.object == c,:].reset_index()
            # stimuli = Ade_subset.loc[Ade_subset.object ==c,'filepath']+'/'+Ade_subset.loc[Ade_subset.object ==c,'filename']
            [create_mask(makeDict_mod(i[1]),masktype,timestr) for i in Ade_cat.iterrows()]
    else: 
        Ade_cat = Ade_subset.loc[Ade_subset.object == cat,:].reset_index()
        [create_mask(makeDict_mod(i[1]),masktype,timestr) for i in Ade_cat.iterrows()]




def create_mask(s,masktype=[],timestr = '0000'):
    # print('Building mask for', s['filepath'])
    objMasks = atr.wrapSegProcessing(s,timestr,SaveFolder)
    if len(objMasks)==0:
        new_mask = []
        print('not mask possible')
        return new_mask

    new_mask = msk.wrapProduceNewMask(s, objMasks,masktype,timestr,SaveFolder)
    
    if new_mask.size == 0:
        print('empty')
        return new_mask
    new_mask = SaveMask(s,new_mask,SaveFolder,masktype)
    return new_mask



def SaveMask(s,image,SaveFolder,masktype):
    filepath = s['filepath'].split('/')
    if local:
        limit = 4
    else:
        limit = 8
    filepath[0:limit] = []
    # print('filepath', filepath)
    SaveMaskPath = ['..',SaveFolder, masktype ]
    SaveMaskPath = '/'.join( SaveMaskPath + filepath[:-1])
    # print('SaveMaskPath',SaveMaskPath)
    if not os.path.exists(SaveMaskPath):
        os.makedirs(SaveMaskPath)
    image.save(SaveMaskPath+'/'+filepath[-1]+'_'+s['category']+'_'+ masktype + '.jpg')
    return image



