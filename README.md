# code-00-buildMasks

## Part of ExploringContextProject, builds masks from ADE20k Dataset based on a set of 81 object categories. Options are: object-only mask (scene covered), scene-only mask (object covered).

## In scene-only option, it creates a convex hull around the object to eliminate shape information.

## Also creates mask with gray background and object covered by noise.

## It also runs a depth estimate model for images in ADE20k and creates depth images for the images of interest
