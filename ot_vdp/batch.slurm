#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE        #Do not propagate environment
#SBATCH --get-user-env=L     #Replicate login environment
  
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=ot_vdp_ShStab
#SBATCH --time=48:00:00            #Set the wall clock limit to 24 hrs
##SBATCH --partition=gpu
#SBATCH --ntasks=16                 #Request 1 task
#SBATCH --ntasks-per-node=16      #Request 1 task/core per node
##SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --mem=32G                #Request 16GB per node
#SBATCH --output=out/ot_vdp_ShStab.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vaishnavtv@tamu.edu
#SBATCH --account=122756175418


##First Executable Line
ml purge
echo -n 'Started job at: ';date
echo ""
cd /scratch/user/vish0908/Documents/neuripsCode/ot_vdp
julia ot_ll_vdp.jl

echo -n 'Ended job at:';date
echo ""
exit
