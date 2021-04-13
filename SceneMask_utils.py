
from PIL import Image
import numpy as np
import cv2


# def find_contours(img):
#     kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
#     morphed  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#     morphed = cv2.cvtColor(morphed, cv2.COLOR_BGR2GRAY);
#     contours = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours[-2]

def blur_and_tresh(new_mask):
    blur = cv2.blur(new_mask, (3, 3)) # blur the image
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    thresh = (thresh).astype(np.uint8)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY);
    return thresh

def compute_convex_hull(thresh):
	# create hull array for convex hull points
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # calculate points for each contour
	hull = []
	for i in range(len(contours)):
		# creating convex hull object for each contour
		hull.append(cv2.convexHull(contours[i], False))
		# create an empty black image
		new_mask_black = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8) 
	#fill contours with white
	cv2.fillPoly(new_mask_black, pts =hull, color=(255,255,255)); 
	return new_mask_black



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


def RndPixlMask(image_mask,mask_smoothed,background = 'scene'):
    rndPixImg = np.random.randint(low = 0, high =200,size=mask_smoothed.shape[0:2],dtype=np.uint8) #noise
    rndPixImg[mask_smoothed[:,:,1]==255.] = 255. #what is object is random gray and black pixels
    rndPixImg= np.repeat(rndPixImg[:, :, np.newaxis], 3, axis=2)
    #Overlay image
    if background == 'scene':
    	rndPixImg = overlay_image(image_mask, rndPixImg,mask_smoothed)
    elif background == 'maskGray':
    	mask_gray = np.ones(mask_smoothed.shape)*170.
    	rndPixImg = overlay_image(mask_gray, rndPixImg,mask_smoothed)
    return rndPixImg

def overlay_image(foreground_image,background_image, foreground_mask):
    background_mask = cv2.cvtColor(255 - cv2.cvtColor(foreground_mask, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    masked_fg = (foreground_image * (1 / 255.0)) * (foreground_mask * (1 / 255.0))
    masked_bg = (background_image * (1 / 255.0)) * (background_mask * (1 / 255.0))
    return np.uint8(cv2.addWeighted(masked_fg, 255.0, masked_bg, 255.0, 0.0))

