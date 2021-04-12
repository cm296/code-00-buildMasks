import ProduceMasks_noparallel as mask
import sys
masktype = sys.argv[1]
mask.saveImageMasks(masktype = masktype)