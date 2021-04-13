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
mirror = 1
convert = 0
masktype='mirror'


def makeDict_mod_pd(p):
    path = p['filepath']+'/'+p['filename']
    s={}
    s['filepath'] = path[0]
    s['category'] = p.object[0]
    return s

def makeDict_mod(p):
    path = p.filepath+'/'+p.filename
    s={}
    s['filepath'] = path
    s['category'] = p.object
    return s

def saveImageMasks():
    # print('Running Original Script')
    #takes model and loads the features for that image, needs path to of files in directory
    if local:
        Ade_subset = pd.read_csv("../Ade20K_labels/Ade20K_labels_local.txt")
    else:
        Ade_subset = pd.read_csv("../Ade20K_labels/Ade20K_labels_marcc.txt")
    # Ade_subset_obj = Ade_subset[Ade_subset['object'].str.contains("car")]

    conditions = np.unique(Ade_subset.object)
    timestr = time.strftime("%Y%m%d-%H%M%S")

    for c in tqdm(conditions):
        
        stimuli = Ade_subset.loc[Ade_subset.object ==c,'filepath']+'/'+Ade_subset.loc[Ade_subset.object ==c,'filename']

        [create_mask_horizontalband(makeDict(pathi,c),timestr) for pathi in stimuli]


def saveImageMasks_cat(c):
    # print('Running Original Script')
    #takes model and loads the features for that image, needs path to of files in directory
    if local:
        Ade_subset = pd.read_csv("../Ade20K_labels/Ade20K_labels_local.txt")
    else:
        Ade_subset = pd.read_csv("../Ade20K_labels/Ade20K_labels_marcc.txt")

    Ade_cat = Ade_subset.loc[Ade_subset.object == c,:].reset_index()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    [create_mask_horizontalband(makeDict_mod(i[1]),timestr) for i in Ade_cat.iterrows()]

def makeDict_mod(p):
    # print('p', p)
    print('p[filepath]',p['filepath'])
    print('p[filename]',p['filename'])
    path = p['filepath']+'/'+p['filename']
    # print('path',path)
    s={}
    s['filepath'] = path
    s['category'] = p.object
    # print('s',s)
    return s

def makeDict(path, category):
    s={}
    s['filepath'] = path
    s['category'] = category
    return s

def create_mask_horizontalband(s,timestr = '0000'):
    print('Building mask for', s['filepath'])
    

    objMasks = atr.wrapSegProcessing(s,timestr)
    if len(objMasks)==0:
        new_mask = []
        print('not mask possible')
        return new_mask

    new_mask = msk.wrapProduceNewMask(s, objMasks,timestr)
    
    if new_mask.size == 0:
        print('empty')
        return new_mask
    new_mask = SaveMask(s,new_mask)
    return new_mask



def SaveMask(s,image):
    filepath = s['filepath'].split('/')
    if local:
        limit = 4
    else:
        limit = 8
    filepath[0:limit] = []
    # print('filepath', filepath)
    SaveMaskPath = ['..','masks-ade-4', 'PaddedBeacon']
    SaveMaskPath = '/'.join( SaveMaskPath + filepath[:-1])
    # print('SaveMaskPath',SaveMaskPath)
    if not os.path.exists(SaveMaskPath):
        os.makedirs(SaveMaskPath)
    image.save(SaveMaskPath+'/'+filepath[-1]+'_'+s['category']+'_'+'PaddedBeacon.jpg')
    return image



