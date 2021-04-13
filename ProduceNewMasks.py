import os
from PIL import Image
import math
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
import cv2
from gluoncv.data.transforms import mask as tmask
import csv


def wrapProduceNewMask(s, objMasks,Errortimestr,cond = 'PadBeacon'):
    image = BuildImgMask_PadBeacon(s, objMasks,Errortimestr,cond)
    return image


def BuildImgMask_PadBeacon(s, objMasks,Errortimestr,cond = 'PadBeacon'):
    strike = 0
    image_mask = Image.open(s['filepath']+'.jpg').convert('RGB')
    image_mask = np.array(image_mask)
    if cond == 'PadBeacon':
        new_mask = Execute_PadBeacon(s, image_mask,objMasks,Errortimestr)

    image = Image.fromarray(new_mask , 'RGB')
    return image


def Execute_PadBeacon(s, image_mask,objMasks,Errortimestr):
    height,width,idx,obj,whole_len = getparam_PadBeacon(objMasks)
    new_mask, strike, secondhalf = runLeftBottom(height,width,image_mask,whole_len,idx)
    new_mask,strike = runRightTop(height,width,obj,image_mask,new_mask, secondhalf,whole_len,idx,strike)
    if strike: #check if size is too big and wirte in csv file
        writeWarning_border(s,strike,Errortimestr)
    return new_mask

def writeWarning_border(s,strike,Errortimestr):
    if strike ==1:
        print('segmentation mask is touches left or bottom border for: ', s['filepath'])
    elif strike == 2:
        print('segmentation mask touches right and left, top and bottom border for: ', s['filepath'])
    with open('../masks-ade-4/PaddedBeacon/SegMaskTooBig'+ Errortimestr +'.csv', 'a') as f:
        f.write("%s,%s\n"%(s['category'],s['filepath']))


def getparam_PadBeacon(objMasks):
    obj_h = []
    #get heigh values for seg mask
    for r in objMasks:
        #iterate over rows (height)
        #if there's any true value, then assign it to be true
        obj_h.append(any(r))
    obj_w = []
    #get width values for seg mask
    for c in np.transpose(objMasks):
        #iterate over columns (width)
        #if there's any true value, then assign it to be true
        obj_w.append(any(c))
    #get indices of boolean values
    # print('obj_h,obj_w',obj_h,obj_w)
    idx_h = [i for i, x in enumerate(obj_h) if x]
    # print('idx_h',idx_h)
    idx_w = [i for i, x in enumerate(obj_w) if x]
    # print('idx_w',idx_w)
    height = np.max(idx_h).astype(int)-np.min(idx_h) +1
    width = np.max(idx_w).astype(int)-np.min(idx_w) +1
    if height<=width:
        idx = idx_h
        obj = obj_h
        whole_len = np.ceil(height/2).astype(int)
    else:    
        idx = idx_w
        obj = obj_w
        whole_len = np.ceil(width/2).astype(int)
    return height,width,idx,obj,whole_len

def runLeftBottom(height,width,image_mask,whole_len,idx):
    strike = 0
    idx_Min = np.min(idx)
    new_mask =  image_mask
    if height<=width:
        dist = height
    else:
        dist = width

    secondhalf_pt = idx_Min - np.ceil(dist/2)
    secondhalf = np.arange(secondhalf_pt, idx_Min) 
    if secondhalf_pt<0:
        beyond_len = len(secondhalf)
        within_len = len(np.arange(0,idx_Min))
        if within_len != 0:
            repeats = whole_len // within_len
            add = np.mod(whole_len,within_len).astype(int)
            start =0
            end = within_len 
            end = end
            for i in  np.arange(0,repeats):
                if height<=width:
                    new_mask[idx[start:end],:,:] = image_mask[secondhalf[-within_len:].astype(int),:,:] 
                else:
                    new_mask[:,idx[start:end],:] = image_mask[:,secondhalf[-within_len:].astype(int),:] 
                start = end
                end = start + within_len
            if height<=width: 
                new_mask[idx[start:start+add],:,:] = image_mask[secondhalf[-add:].astype(int),:,:] 
            else:
                new_mask[:,idx[start:start+add],:] = image_mask[:,secondhalf[-add:].astype(int),:] 
        else:
            strike = 1;
            
    else:            
        indices = secondhalf.astype(int)
        if height<=width: 
            new_mask[idx[0:whole_len],:,:] = image_mask[indices,:,:] 
        else:
            new_mask[:,idx[0:whole_len],:] = image_mask[:,indices,:] 
    return new_mask,strike, secondhalf 

def runRightTop(height,width,obj,image_mask,new_mask, secondhalf,whole_len,idx,strike):
    idx_Max = np.max(idx)
    if height<=width:
        dist = height
        checksize = new_mask.shape[0]
    else:
        dist = width
        checksize = new_mask.shape[1]
    firsthalf_pt = idx_Max + np.ceil(dist/2)
    firsthalf = np.arange(idx_Max, firsthalf_pt)
    if sum(obj) < len(firsthalf) + len(secondhalf):
        firsthalf= firsthalf[:-1]
    #then do the right or top side
    if firsthalf_pt>checksize:
        within_len = len(np.arange(idx_Max,checksize))
        if strike:
            whole_len = dist.astype(int)
            start = 0
            end = start + within_len
        else:
            start = whole_len.astype(int) -1
            end = whole_len + within_len - 1
            end = end.astype(int)
        first_repeats = whole_len // within_len
        if within_len != 0:
            add = np.mod(whole_len,within_len).astype(int) 
            for i in  np.arange(0,first_repeats):
                if height<=width: 
                    new_mask[idx[start:end],:,:] = image_mask[firsthalf[0:within_len].astype(int),:,:] 
                else:
                    new_mask[:,idx[start:end],:] = image_mask[:,firsthalf[0:within_len].astype(int),:] 
                start = end
                end = start + within_len
            if height<=width:
                new_mask[idx[start:start+add],:,:] = image_mask[firsthalf[0:add].astype(int),:,:]  
            else:
                new_mask[:,idx[start:start+add],:] = image_mask[:,firsthalf[0:add].astype(int),:] 

            image = Image.fromarray(new_mask , 'RGB')
        else:
            if strike:
                strike = 2
                image = []
                return image, strike
            else: 
                print('WHAT HAPPENS HERE?')
    else:
        indices = firsthalf.ravel().astype(int)
        if height<=width: 
            if len(indices)>=whole_len:
                new_mask[idx[-whole_len:],:,:] = image_mask[indices[0:whole_len],:,:] 
                if np.mod(len(indices),whole_len):
                    add = np.mod(len(indices),whole_len).astype(int)
                    new_mask[idx[len(indices)-add:len(indices)],:,:] = image_mask[indices[0:add],:,:] 

            elif len(indices)<whole_len:
                whole_len = len(indices)
                new_mask[idx[-whole_len:],:,:] = image_mask[indices,:,:] 
                
        else:
            if len(indices)>=whole_len:
                new_mask[:,idx[-whole_len:],:] = image_mask[:,indices[0:whole_len],:] 
                if np.mod(len(indices),whole_len):
                    add = np.mod(len(indices),whole_len).astype(int)
                    new_mask[:,idx[len(indices)-add:len(indices)],:] = image_mask[:,indices[0:add],:] 
            elif len(indices)<whole_len:
                whole_len = len(indices)
                new_mask[:,idx[-whole_len:],:] = image_mask[:,indices,:] 
    return new_mask, strike
