# code-00-buildMasks

## Part of ExploringContextProject, builds masks from ADE20k Dataset based on a set of 81 object categories. Options are: object-only mask (scene covered), scene-only mask (object covered).

## Execute ProduceImageMasks_script.py to run analysis and create object mask and pad beacon mask

## In scene-only option, it creates a convex hull around the object to eliminate shape information.

##Execute CreateAdeResampled.py to run the scene-only version from manuscript, which for each row of csv file saves path for an alternative image with same scene label