import os
from PIL import Image
import math
import pandas as pd
import numpy as np
from random import randrange
import cv2
import csv


def wrapSegProcessing(s, Errortimestr,SaveFolder):
    ObjectInstanceMasks,atr = processAtr(s, Errortimestr,SaveFolder)
    objMasks = buildObjMask(s,ObjectInstanceMasks,atr,Errortimestr,SaveFolder)
    return objMasks



def processAtr(s, Errortimestr,SaveFolder):
    atr = pd.read_csv(s['filepath']+'_atr.txt', sep=' # ', header=None,index_col=4,engine = 'python')
    #,usecols = [0,4],engine = 'python')
    #rename indices of atr when not matching out labels
    if 'book' in s['category']:
        if any('books' in s for s in np.unique(atr.index)):
            atr = atr.rename( index={'books': 'book'}) #If it's found, then rename the atr file
    if s['category'] not in atr.index:
        # print('not founds the category')
        # print('atr file:', s['filepath']+'_atr.txt')
        atr = renameAtr_speed(atr,s['category'])
    #if it can't be found in atr.txt then report it and interrupt the function
    if s['category'] not in atr.index:
        print('can''t find label')
        
        with open('../'+SaveFolder+'/SegMasksNotFound_'+ Errortimestr +'.csv', 'a') as f:
            for key in s.keys():
                f.write("%s,%s\n"%(key,s[key]))
        new_mask = []
        return new_mask
    ObjectInstanceMasks = load_ob_mask(atr,s,SaveFolder)
    return ObjectInstanceMasks,atr


def buildObjMask(s,ObjectInstanceMasks,atr,Errortimestr,SaveFolder):
	# get the index from the _atr.txt file that identifies object in image
    if not atr.loc[s['category'],0].astype('int8').shape:                
        mask_values = atr.loc[s['category'],0].astype('int8')
    else:
        mask_values = atr.loc[s['category'],0].to_numpy().tolist()

	#build am objmask with only values for object of interest
    objMasks = [] #start building a mask with our values!

    if (atr.loc[s['category'],1] != 0).any(): #If it's not in the column of object instances, then maybe it's in the part images?
        objMasks,mask_values = run_parts_imgs(s,atr,mask_values)

    else: #If it is in the column of the object instances, check that there are no mismatched between segmentatio and object
        if np.max(ObjectInstanceMasks) != np.max(atr[atr.iloc[:,1]==0].iloc[:,0]):
            print('_atr file mislabeled for this object for cat and file: ', s['category'],s['filepath'])
            with open('../'+SaveFolder+'/AtrFileIndexError_'+ Errortimestr +'.csv', 'a') as f:
                f.write("%s,%s\n"%(s['category'],s['filepath']))
            new_mask = []
            return new_mask
    if mask_values: #if there are still values in main seg mask after removing the ones present in parts 1 and 2
        if isinstance(mask_values, np.int8):
            objMasks.append(ObjectInstanceMasks == mask_values)
        else:
            for i in mask_values:
                if len(objMasks) is 0:
                    objMasks.append(ObjectInstanceMasks == i)
                else:
                    objMasks = np.logical_or(objMasks,ObjectInstanceMasks == i)
    objMasks = np.squeeze(objMasks)
    return objMasks



def load_ob_mask(atr,s,SaveFolder):
    col_parts = 1
    if (atr.loc[s['category'],col_parts] == 1).all(): #if it's a "part" object (such as mouse) then column 1 has nonzero, then load the parts image
        image_mask = Image.open(s['filepath']+'_parts_'+str(1)+'.png').convert('RGB')            
    #if it's all in part 2 .png
    elif (atr.loc[s['category'],col_parts] == 2).all(): #if it's a "part" object (such as mouse) then column 1 has nonzero, then load the parts image
        #if the file doesn't exist, report it and interrupt the function
        if not os.path.isfile(s['filepath']+'_parts_'+str(2)+'.png'):
            # print('parts_2 does not exist')
            with open('../'+SaveFolder+'/SegMasksNotFound.csv', 'a') as f:
                for key in s.keys():
                    f.write("%s,%s\n"%(key,s[key]))
            return new_mask
        else:
            image_mask = Image.open(s['filepath']+'_parts_'+str(2)+'.png').convert('RGB')            
    else:
        #if it's all in the main segmentation mask
        image_mask = Image.open(s['filepath']+'_seg.png').convert('RGB')
    image_mask = np.array(image_mask)
    u,Ind,Inv = np.unique(image_mask[:, :, 2],return_index=True, return_inverse=True)
    ObjectInstanceMasks = np.reshape(Inv,(image_mask.shape[0], image_mask.shape[1]));
    return ObjectInstanceMasks



def renameAtr_speed(atr,cat):
    # print('cat not found, looking at atr file for: ', cat)
    group = pd.read_csv("../Ade20K_labels/saved_groups.csv",index_col='Object')
    group = group[["MoreNames1","MoreNames2","MoreNames3","MoreNames4"]].astype(int,errors='ignore')
    group = group.loc[cat]
    group = group.to_numpy(dtype = str).tolist()
    # group = group.split(",")
    newgroup = []
    for word in group:
        if word == 'nan':
            continue
        # print('word', word)
        newgroup = [w.strip() for w in word.split(",")]
    newgroup.insert(0,cat)
    
    # print('np.unique(atr.index):', np.unique(atr.index))
    keeplooking = 1 #if this stays zero then we check inside atr file because cat doesn't match any of our atr indices
    for thisCat in np.unique(atr.index): #Go through all the indices of the atr file
        # print('ThisCat: ', thisCat)
        for s in newgroup:
            # print('s1: ', s)
            # print('compare s and this cat:', thisCat, s)
            if thisCat == s or thisCat == s+'s':
                keeplooking = 0
                # print('hereee1')
                # print('thisCat printing: ', thisCat)
                # print('found index to substitute: ', thisCat)
                atr = atr.rename( index={thisCat: cat}) #If it's found, then rename the atr file
        if keeplooking == 0:
            # print('correct index was', thisCat)
            return atr
    for thisCat in np.unique(atr.index):
        #otherwise look for all the atr labels to what Group they correspond to
        atr_v = atr.loc[thisCat].to_numpy().flatten().astype('str') 
        # print('atr_v: ', atr_v)
        for atrWord in atr_v:
            # print('atrWord: ', atrWord)                 
            if isinstance(atrWord, str):
                if atrWord.isalpha():
                    # print('thisCat:', thisCat)
                    # print('compare cat and this atrWord:', cat,atrWord)
                    if cat == atrWord or cat+'s' == atrWord or cat == atrWord+'s':
                        keeplooking = 0
                        # print('found index to substitute at step 2: ', thisCat)
                        # print('string matched was: ', atrWord)
                        atr = atr.rename( index={thisCat: cat})
                if ',' in atrWord:
                    # print('here3')
                    newatrWord = [w.strip() for w in atrWord.split(",")]
                    for w in newatrWord:
                        if cat == w or cat+'s' == w or cat == w+'s':
                            keeplooking = 0
                            # print('found index to substitute at step 2b: ', thisCat)
                            # print('string matched was: ', thisCat)
                            atr = atr.rename( index={thisCat: cat})
        if keeplooking == 0:
            # print('correct index was', thisCat)
            return atr
    if keeplooking == 1:
        print('no match found for ', cat)
        print('newgroup', newgroup)  
        print('np.unique(atr.index):', np.unique(atr.index))
    return atr


def run_parts_imgs(s,atr,mask_values):
    objMasks = []
    col_parts = 1
    if (atr.loc[s['category'],col_parts] == 1).any(): 
        if not (atr.loc[s['category'],col_parts] == 1).all(): #if not all categories are parts 1
            
            index_parts_1, Obj_Mask1, mask_values = ComputePartsMask(1,s,atr,mask_values)
    if (atr.loc[s['category'],col_parts] == 2).any(): 
        if not (atr.loc[s['category'],col_parts] == 2).all(): #if not all categories are parts 2
            index_parts_2, Obj_Mask2, mask_values = ComputePartsMask(2,s,atr,mask_values)
    
    if 'index_parts_1' in locals():
        objMasks = Obj_Mask1
    if 'index_parts_2' in locals():
        objMasks = appendObjMask(objMasks,Obj_Mask2)
    return objMasks,mask_values

def ComputePartsMask(part_n,s,atr,mask_values):
    col_parts = 1
    catInfo = atr.loc[s['category'],:]
    atr_parts_1 = catInfo.loc[atr.loc[s['category'],col_parts]==part_n]
    image_mask_1 = Image.open(s['filepath']+'_parts_'+str(part_n)+'.png').convert('RGB')
    image_mask_1 = np.array(image_mask_1)
    u_1,Ind_1,Inv_1 = np.unique(image_mask_1[:, :, 2],return_index=True, return_inverse=True)
    ObjectInstanceMasks_1 = np.reshape(Inv_1,(image_mask_1.shape[0], image_mask_1.shape[1])); 
    Obj_Mask1 = []
    for i in atr_parts_1[0]:
        if len(Obj_Mask1) is 0:
            Obj_Mask1.append(ObjectInstanceMasks_1 == i)
        else:
            Obj_Mask1 = np.logical_or(Obj_Mask1,ObjectInstanceMasks_1 == i)
    Obj_Mask1 = np.squeeze(Obj_Mask1)
    to_remove = atr_parts_1[0].to_numpy().tolist()
    mask_values = [x for x in mask_values if x not in to_remove]
    return atr_parts_1, Obj_Mask1, mask_values



def appendObjMask(objMasks,thisMap):
    if len(objMasks) is 0:
        objMasks = thisMap
    else:
        objMasks = np.logical_or(objMasks,thisMap)
    return objMasks



