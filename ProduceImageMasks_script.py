import ProduceMasks_wrapper as mask
import sys
masktype = sys.argv[1]
mask.ProduceImageMasks(masktype = masktype)