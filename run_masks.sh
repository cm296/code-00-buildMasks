#!/bin/bash -l

#SBATCH
#SBATCH --job-name=GrayMask_Cat
#SBATCH --time=72:00:0
##SBATCH --mem=8G
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=18
#SBATCH --mail-type=end
#SBATCH --mail-user=cmagri1@jhu.edu

# This is a script to render Blender models on MARCC.
# Change mail-user flag above to your MARCC usename. Change job-name flag above to whatever you like.
# Make sure to alot enough time for your job with the time flag above (but not too much more, otherwise MARCC can't allocate resources efficiently)
# Go to MARCC website to change the other flags according to your needs.
# Change the location and file name of your .blend file on line 27. Also change the location and file name of your desired rendered image.
# Put this script in your /scratch or /work folder.
# Run this script by cd-ing into the directory it is located in, and typing in the Terminal: "sbatch slurm_blender_example.sh" or whatever name you decide to give it.
# Check your job finding the output file in the same directory where your script is. E.g. slurm-29927031.out. Open this output file by typing: "nano slurm-29927031.out"
# Quit nano by pressing: CTRL + x
# Check your job queue typing "sqme" in the Terminal
# Cancel a job by typing "scancel JOB_NUMBER" (e.g. scancel 29927031) in the Terminal
# Kudos to Li Guo who passed along her knowledge!

#module load blender
ml python/3.7-anaconda

baseDir=/home-1/cmagri1@jhu.edu/
# curDir= /home-1/cmagri1@jhu.edu/work/cmagri/Project-ExploringContext/adeContext-datasets/code-00-buildMasks/

# cd baseDir
conda activate ${baseDir}/my_plot_env
# cd curDir
# blendFileDir=${baseDir}/final_blends
# blendFilename=fall_shatter
# outDir=${blendFileDir}/${blendFilename}

# mkdir -p ${outDir}

# blender -b ${blendFileDir}/${blendFilename}.blend -x 1 -o ${outDir}/${blendFilename} -a
python ProduceMasks_script_noparallel.py object

echo "Finished with job $SLURM_JOBID"
