import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import ProduceNewMasks as msk
import ProcessSegMask as atr
import csv
import time
import pylab as P

local = 1
SaveFolder = 'masks-ade-4'


def main():
	#Load the csv file contaiing the scene subset
	if local:
		Ade_subset = pd.read_csv("../Ade20K_labels/Ade20K_labels_local_scene_subset.txt")
	else:
		Ade_subset = pd.read_csv("../Ade20K_labels/Ade20K_labels_marcc_scene_subset.txt")
	## Set up dataset to work from
	#count the number of cocurrences of object in the dataset by scene category
	obj_count_byscene = pd.DataFrame({'countOcc': Ade_subset.groupby(['object','scene']).object.count()})
	#Reorganize indexing, since output is a multiindex dataframe
	#Set "scene" index under scene column
	obj_count_byscene['scene']=obj_count_byscene.index.get_level_values(1)
	#Set "object" as only index
	obj_count_byscene.index = obj_count_byscene.index.get_level_values(0)
	#set the scene label as index, and the object label as a column
	scene_count = obj_count_byscene.set_index('scene')
	scene_count['object'] = obj_count_byscene.index

	# sc_by_cat = scene_count[scene_count.index == 'air_base']
	# nImages = int(sc_by_cat[sc_by_cat['object'] == 'airplane'].countOcc)

	newdf = pd.DataFrame()
	missing = pd.DataFrame()
	missobj = [] 
	misssc = []
	misscount = []
	#Now looping through the index/object labels
	for obj in tqdm(np.unique(obj_count_byscene.index)):
		#Pick up all the counts containing this object
		thisobj = obj_count_byscene.loc[obj]
		#eliminate all of this object occcurrences
		Ade_NoObj = Ade_subset[Ade_subset['object'] != obj]
		#eliminate all images that contain this object, by picking up the filename
		ListFiles = np.unique(Ade_subset[Ade_subset['object'] == obj].filename)
		Ade_NoObj = Ade_NoObj[~Ade_NoObj.filename.isin( ListFiles)]
		#group and count scene by keeping filename and filepath
		Ade_NoObj = pd.DataFrame({'countob' : Ade_NoObj.groupby(['filename','scene','filepath']).scene.count()})
		Ade_NoObj['filepath'] = Ade_NoObj.index.get_level_values(2)
		Ade_NoObj['scene'] = Ade_NoObj.index.get_level_values(1)
		#set filename as the index 
		Ade_NoObj.index = Ade_NoObj.index.get_level_values(0)
		#Now loop over all objects that appear in a given scene label
		for sc in thisobj['scene']:
			#Count the times this object appears in this scene
			sc_by_cat = scene_count[scene_count.index==sc]
			nImages = int(sc_by_cat[sc_by_cat['object'] == obj].countOcc)
			#if there are no scenes WITHOUT this object save it in the log
			if len(Ade_NoObj[Ade_NoObj['scene']==sc]) == 0:
				missobj.append(obj)
				misssc.append(sc)
				misscount.append(nImages)
				continue
			#if there are more non-object-containing scenes than object-containing scenes, then pick without replacement
			elif nImages < len(Ade_NoObj[Ade_NoObj['scene']==sc]):
				newsample = Ade_NoObj[Ade_NoObj['scene']==sc].sample(nImages)
			else:
				#otherwise pick with replacement
				newsample = Ade_NoObj[Ade_NoObj['scene']==sc].sample(nImages,replace=True)
			#Seva information about object category and scene
			newsample = newsample.assign(objectCat = obj)
			newsample = newsample.assign(scene = sc)
			newdf = newdf.append(newsample)
	#save information about hte missing items
	missing_items = pd.DataFrame({'object': missobj,'scene': misssc,'countObj': misscount})
	newdf = newdf.reset_index()
	newdf = newdf[["objectCat", "filepath", "filename",'scene']]
	newdf = newdf.rename(columns={"objectCat": "object"})
	if local:
		newdf.to_csv('../Ade20K_labels/Ade20K_labels_local_scene_subset_resampled.txt')    
	else:
		newdf.to_csv('../Ade20K_labels/Ade20K_labels_marcc_scene_subset_resampled.txt')   
	missing_items.to_csv('../Ade20K_labels/missingItems_resampled.txt')


def countMissing():
	if local:
		Ade_subset = pd.read_csv("../Ade20K_labels/Ade20K_labels_local_scene_subset.txt")
	else:
		Ade_subset = pd.read_csv("../Ade20K_labels/Ade20K_labels_marcc_scene_subset.txt")
	missing_items = pd.read_csv('../Ade20K_labels/missingItems_resampled.txt')
	countMissing = pd.DataFrame({'countObj': missing_items.groupby('object').object.count()})
	countCats = pd.DataFrame({'countObj': Ade_subset.groupby('object').object.count()})
	countcats_all = countMissing.merge(countCats, left_index = True, right_index=True, how='inner')
	countcats_all['percent'] = countcats_all['countObj_x']/countcats_all['countObj_y']
	P.figure(figsize=(8, 6), dpi=80)
	bp = P.boxplot(countcats_all['percent'])
	y = countcats_all['percent']
	x = np.random.normal(1, 0.04, size=len(y))
	P.plot(x, y, 'r.', alpha=0.2)
	P.show()



if __name__ == '__main__':
	main()
else:
	print('loaded as module')







