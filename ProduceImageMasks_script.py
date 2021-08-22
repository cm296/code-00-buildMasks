import ProduceMasks_wrapper as mask
import sys
#Possible options: 'PadBeacon', 'object', 'scene'
masktype = sys.argv[1]
mask.ProduceImageMasks(masktype = masktype)