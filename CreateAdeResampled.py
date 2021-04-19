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
	if local:
		Ade_subset = pd.read_csv("../Ade20K_labels/Ade20K_labels_local_scene_subset.txt")
	else:
		Ade_subset = pd.read_csv("../Ade20K_labels/Ade20K_labels_marcc_scene_subset.txt")

	obj_count_byscene = pd.DataFrame({'countOcc': Ade_subset.groupby(['object','scene']).object.count()})
	obj_count_byscene['scene']=obj_count_byscene.index.get_level_values(1)
	obj_count_byscene.index = obj_count_byscene.index.get_level_values(0)

	scene_count = obj_count_byscene.set_index('scene')
	scene_count['object'] = obj_count_byscene.index

	sc_by_cat = scene_count[scene_count.index == 'air_base']
	nImages = int(sc_by_cat[sc_by_cat['object'] == 'airplane'].countOcc)

	newdf = pd.DataFrame()
	missing = pd.DataFrame()
	missobj = [] 
	misssc = []
	misscount = []
	for obj in tqdm(np.unique(obj_count_byscene.index)):
		thisobj = obj_count_byscene.loc[obj]
		#eliminate all of this object occcurrences
		Ade_NoObj = Ade_subset[Ade_subset['object'] != obj]
		#eliminate all images that contain this object
		ListFiles = np.unique(Ade_subset[Ade_subset['object'] == obj].filename)
		Ade_NoObj = Ade_NoObj[~Ade_NoObj.filename.isin( ListFiles)]
		#group by the remaining things
		Ade_NoObj = pd.DataFrame({'countob' : Ade_NoObj.groupby(['filename','scene','filepath']).scene.count()})
		Ade_NoObj['filepath'] = Ade_NoObj.index.get_level_values(2)
		Ade_NoObj['scene'] = Ade_NoObj.index.get_level_values(1)
		Ade_NoObj.index = Ade_NoObj.index.get_level_values(0)
		for sc in thisobj['scene']:
			sc_by_cat = scene_count[scene_count.index==sc]
			nImages = int(sc_by_cat[sc_by_cat['object'] == obj].countOcc)
			if len(Ade_NoObj[Ade_NoObj['scene']==sc]) == 0:
				missobj.append(obj)
				misssc.append(sc)
				misscount.append(nImages)
				continue
			elif nImages < len(Ade_NoObj[Ade_NoObj['scene']==sc]):
				newsample = Ade_NoObj[Ade_NoObj['scene']==sc].sample(nImages)
			else:
				newsample = Ade_NoObj[Ade_NoObj['scene']==sc].sample(nImages,replace=True)
			newsample = newsample.assign(objectCat = obj)
			newsample = newsample.assign(scene = sc)
			newdf = newdf.append(newsample)
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







