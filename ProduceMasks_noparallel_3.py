import os
from PIL import Image
# import Image
import math
import pandas as pd
# import os.path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
import cv2
from gluoncv.data.transforms import mask as tmask
import csv
#Masktypes = 'object', 'sceneConxexHull','mask'
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

    for c in tqdm(conditions):
        
        stimuli = Ade_subset.loc[Ade_subset.object ==c,'filepath']+'/'+Ade_subset.loc[Ade_subset.object ==c,'filename']

        [create_mask_horizontalband(makeDict(pathi,c)) for pathi in stimuli]


def saveImageMasks_cat(c):
    # print('Running Original Script')
    #takes model and loads the features for that image, needs path to of files in directory
    if local:
        Ade_subset = pd.read_csv("../Ade20K_labels/Ade20K_labels_local.txt")
    else:
        Ade_subset = pd.read_csv("../Ade20K_labels/Ade20K_labels_marcc.txt")

    Ade_cat = Ade_subset.loc[Ade_subset.object == c,:].reset_index()
    [create_mask_horizontalband(makeDict_mod(i[1])) for i in Ade_cat.iterrows()]

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

def create_mask_horizontalband(s):
    print('Building mask for', s['filepath'])
    atr = pd.read_csv(s['filepath']+'_atr.txt', sep=' # ', header=None,index_col=4,engine = 'python')
    #,usecols = [0,4],engine = 'python')
    #rename indices of atr when not matching out labels
    if 'book' in s['category']:
        if any('books' in s for s in np.unique(atr.index)):
            atr = atr.rename( index={'books': 'book'}) #If it's found, then rename the atr file
    if s['category'] not in atr.index:
        # print('not founds the category')
        # print('atr file:', s['filepath']+'_atr.txt')
        atr = renameCat_speed(atr,s['category'])
    #if it can't be found in atr.txt then report it and interrupt the function
    if s['category'] not in atr.index:
        print('can''t find label')
        with open('SegMasksNotFound.csv', 'a') as f:
            for key in s.keys():
                f.write("%s,%s\n"%(key,s[key]))
        new_mask = []
        return new_mask
    # get the index from the _atr.txt file that identifies object in image
    if not atr.loc[s['category'],0].astype('int8').shape:                
        mask_values = atr.loc[s['category'],0].astype('int8')
    else:
        mask_values = atr.loc[s['category'],0].to_numpy().tolist()
        # print('mask values: ', mask_values)
    ObjectInstanceMasks = load_ob_mask(atr,s)
    #build am objmask with only values for object of interest
    objMasks = [] #start building a mask with our values!

    if (atr.loc[s['category'],1] != 0).any(): #If it's not in the column of object instances, then maybe it's in the part images?
        objMasks,mask_values = run_parts_imgs(s,atr,mask_values)

    else: #If it is in the column of the object instances, check that there are no mismatched between segmentatio and object
        if np.max(ObjectInstanceMasks) != np.max(atr[atr.iloc[:,1]==0].iloc[:,0]):
            print('_atr file mislabeled for this object for cat and file: ', s['category'],s['filepath'])
            with open('AtrFileMismatch.csv', 'a') as f:
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
    new_mask = BuildImgMask_mirror(s, objMasks)
    if new_mask.size == 0:
        print('empty')
        return new_mask
    new_mask = SaveMask(s,new_mask)
    return new_mask

def BuildImgMask_mirror(s, objMasks):
    strike = 0
    image_mask = Image.open(s['filepath']+'.jpg').convert('RGB')
    image_mask = np.array(image_mask)
    new_mask = np.zeros((image_mask.shape[0], image_mask.shape[1]))
    new_mask = np.repeat(new_mask[:, :, np.newaxis], 3, axis=2)
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
    if mirror:    
        new_mask = image_mask #make object white
        #get indices of boolean values
        idx_h = [i for i, x in enumerate(obj_h) if x]
        idx_w = [i for i, x in enumerate(obj_w) if x]
        idx_h_Max = np.max(idx_h)
        idx_h_Min = np.min(idx_h)
        idx_w_Max = np.max(idx_w)
        idx_w_Min = np.min(idx_w)
        height = idx_h_Max-idx_h_Min +1
        width = idx_w_Max-idx_w_Min +1
        #check ehther is taller or wider, and decide based on that whether to make horizontal or vertical bars
        # print('height:', height, '; width: ', width)
        if height<=width:
            idx = idx_h
            obj = obj_h
            idx_Max = idx_h_Max
            idx_Min = idx_h_Min
            dist = height
            checksize = new_mask.shape[0]
        else:
            # print('do width')
            idx = idx_w
            obj = obj_w
            idx_Max = idx_w_Max
            idx_Min = idx_w_Min
            dist = width
            checksize = new_mask.shape[1]

        # print('obj',obj)
        #Max and min indices from which to flip image
        firsthalf_pt = idx_Max + np.ceil(dist/2)
        secondhalf_pt = idx_Min - np.ceil(dist/2)

        firsthalf = np.arange(idx_Max, firsthalf_pt)
        secondhalf = np.arange(secondhalf_pt, idx_Min) 

        # print('idx', idx)

        if sum(obj) < len(firsthalf) + len(secondhalf):
            firsthalf= firsthalf[:-1]
            # print('element removed')

        

        whole_len = np.ceil(dist/2).astype(int)
        #First do the left or bottom side
        #Check that are within the bounds. if not, then copy from the other side
        if secondhalf_pt<0:
            # print('here1 a')
            # within_len = len(np.arange(checksize,idx_Min))
            # firsthalf_pt = firsthalf_pt + np.abs(secondhalf_pt)
            beyond_len = len(secondhalf)
            within_len = len(np.arange(0,idx_Min))
            # print('within len', within_len)

            if within_len != 0:
                repeats = whole_len // within_len
                # print('whole len', whole_len)
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
                new_mask[:,idx[start:start+add],:] = image_mask[:,secondhalf[-add:].astype(int),:] 
            else:
                strike = 1;
                print('segmentation mas is too big for: ', s['filepath'])
                with open('SegMaskTooBig.csv', 'a') as f:
                    f.write("%s,%s\n"%(s['category'],s['filepath']))
        else:
            # print('here1 b')
            if convert:
                indices = np.flip(secondhalf)
                indices = indices.astype(int)
            else:
                indices = secondhalf.astype(int)
            if height<=width: 
                new_mask[idx[0:whole_len],:,:] = image_mask[indices,:,:] 
            else:
                new_mask[:,idx[0:whole_len],:] = image_mask[:,indices,:] 
        #then do the right or top side
        if firsthalf_pt>checksize:
            # print('here2 a')
            
            within_len = len(np.arange(idx_Max,checksize))
            first_repeats = whole_len // within_len

            if within_len != 0:
                add = np.mod(whole_len,within_len).astype(int)
                start = whole_len.astype(int) -1
                end = whole_len +within_len -1
                end = end.astype(int)
                for i in  np.arange(0,first_repeats):
                    if height<=width: 
                        new_mask[idx[start:end],:,:] = image_mask[firsthalf[0:within_len].astype(int),:,:] 
                    else:
                        new_mask[:,idx[start:end],:] = image_mask[:,firsthalf[0:within_len].astype(int),:] 
                    start = end
                    end = start + within_len
                new_mask[:,idx[start:start+add],:] = image_mask[:,firsthalf[0:add].astype(int),:] 
                image = Image.fromarray(new_mask , 'RGB')
            else:
                if strike:
                    print('part 2 segmentation mas is too big for: ', s['filepath'])
                    with open('SegMaskTooBig.csv', 'a') as f:
                        f.write("%s,%s\n"%(s['category'],s['filepath']))
                    image = []
                    return image
                else: 
                    print('WHAT HAPPENS HERE?')
        else:
            # print('here2 b')
            if convert:
                indices = np.flip(firsthalf)
                indices = indices.ravel().astype(int)
            else:
                indices = firsthalf.ravel().astype(int)
            # print('idx[whole_len-1:]',image_mask[indices,:,:] )
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
                # print('whole_len',whole_len)
                # print('idx length', len(idx))
                # print('indices length', len(indices))
                if len(indices)>=whole_len:
                    new_mask[:,idx[-whole_len:],:] = image_mask[:,indices[0:whole_len],:] 
                    if np.mod(len(indices),whole_len):
                        add = np.mod(len(indices),whole_len).astype(int)
                        new_mask[:,idx[len(indices)-add:len(indices)],:] = image_mask[:,indices[0:add],:] 
                elif len(indices)<whole_len:
                    whole_len = len(indices)
                    new_mask[:,idx[-whole_len:],:] = image_mask[:,indices,:] 
        image = Image.fromarray(new_mask , 'RGB')
        return image
    else:
        new_mask[obj,:,:] = 255. #make object white
        makeNoisyMask(new_mask,image_mask)
    return image


def SaveMask(s,image):
    filepath = s['filepath'].split('/')
    if local:
        limit = 4
    else:
        limit = 8
    filepath[0:limit] = []
    print('filepath', filepath)
    SaveMaskPath = ['..','masks-ade-3', 'sceneConvexHull' + 'Only']
    SaveMaskPath = '/'.join( SaveMaskPath + filepath[:-1])
    print('SaveMaskPath',SaveMaskPath)
    if not os.path.exists(SaveMaskPath):
        os.makedirs(SaveMaskPath)
    image.save(SaveMaskPath+'/'+filepath[-1]+'_'+s['category']+'_'+'sceneConvexHull'+'Only.jpg')
    return image



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

def makeNoisyMask(new_mask,image_mask):
    blur = cv2.blur(new_mask, (3, 3)) # blur the image
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    thresh = (thresh).astype(np.uint8)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY);
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create hull array for convex hull points
    hull = []
    # calculate points for each contour
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))# creating convex hull object for each contour
    # create an empty black image
    new_mask = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
    #fill contours with white
    cv2.fillPoly(new_mask, pts =hull, color=(255,255,255)); 
    mask_e = dilate_mask(new_mask)
    mask_smoothed = smooth_mask(mask_e) #smooth mask
    #Create random pixels and mask them
    image = RndPixlMask(image_mask,mask_smoothed,masktype='sceneConvexHull')
    return image

def load_ob_mask(atr,s):
    col_parts = 1
    if (atr.loc[s['category'],col_parts] == 1).all(): #if it's a "part" object (such as mouse) then column 1 has nonzero, then load the parts image
        image_mask = Image.open(s['filepath']+'_parts_'+str(1)+'.png').convert('RGB')            
    #if it's all in part 2 .png
    elif (atr.loc[s['category'],col_parts] == 2).all(): #if it's a "part" object (such as mouse) then column 1 has nonzero, then load the parts image
        #if the file doesn't exist, report it and interrupt the function
        if not os.path.isfile(s['filepath']+'_parts_'+str(2)+'.png'):
            # print('parts_2 does not exist')
            with open('SegMasksNotFound.csv', 'a') as f:
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

def overlay_image(foreground_image,background_image, foreground_mask):
    background_mask = cv2.cvtColor(255 - cv2.cvtColor(foreground_mask, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    masked_fg = (foreground_image * (1 / 255.0)) * (foreground_mask * (1 / 255.0))
    masked_bg = (background_image * (1 / 255.0)) * (background_mask * (1 / 255.0))
    return np.uint8(cv2.addWeighted(masked_fg, 255.0, masked_bg, 255.0, 0.0))

def dilate_mask(new_mask, dilatation_size = 60):
    # Options: cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE
    dilatation_type = cv2.MORPH_CROSS
    element = cv2.getStructuringElement(dilatation_type,(2*dilatation_size + 1, 2*dilatation_size+1),(dilatation_size, dilatation_size))
    mask_e = cv2.dilate(new_mask, element)
    return mask_e

def smooth_mask(mask_e):
    #Invert te colors and Smooth the Mask
    mask_smoothed = np.zeros(mask_e.shape)
    mask_smoothed[mask_e == 0.] = 255.
    mask_smoothed = cv2.blur(mask_smoothed.astype(np.uint8),(40, 40), 0)
    return mask_smoothed

def RndPixlMask(image_mask,mask_smoothed,masktype):
    rndPixImg = np.random.randint(low = 0, high =200,size=mask_smoothed.shape[0:2],dtype=np.uint8) #noise
    rndPixImg[mask_smoothed[:,:,1]==255.] = 255. #what is object is random gray and black pixels
    rndPixImg= np.repeat(rndPixImg[:, :, np.newaxis], 3, axis=2)
    #Overlay image
    if masktype == 'sceneConvexHull':
    	rndPixImg = overlay_image(image_mask, rndPixImg,mask_smoothed)
    elif masktype == 'maskWhite':
    	rndPixImg = overlay_image(mask_smoothed, rndPixImg,mask_smoothed)
    elif masktype == 'maskGray':
    	mask_gray = np.ones(mask_smoothed.shape)*170.
    	rndPixImg = overlay_image(mask_gray, rndPixImg,mask_smoothed)
    image = Image.fromarray(rndPixImg , 'RGB')
    return image


    
def renameCat_speed(atr,cat):
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

def find_contours(img):
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    morphed = cv2.cvtColor(morphed, cv2.COLOR_BGR2GRAY);
    contours = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[-2]

def mask_from_contours(ref_img, contours):
    mask = np.zeros(ref_img.shape, np.uint8)
    mask = cv2.drawContours(mask, contours, -1, (255,255,255), -1)
    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)