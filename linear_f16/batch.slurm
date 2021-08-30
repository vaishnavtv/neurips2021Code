#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE        #Do not propagate environment
#SBATCH --get-user-env=L     #Replicate login environment
  
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=linear_f16_gpu_ADAM
#SBATCH --time=48:00:00            #Set the wall clock limit to 24 hrs
#SBATCH --partition=gpu
#SBATCH --ntasks=16                 #Request 1 task
#SBATCH --ntasks-per-node=16      #Request 1 task/core per node
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --mem=32G                #Request 16GB per node
#SBATCH --output=out/linear_f16_ADAM.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vaishnavtv@tamu.edu
##SBATCH --account=122756175418


##First Executable Line
ml purge
echo -n 'Started job at: ';date
echo ""
cd /scratch/user/vish0908/Documents/neuripsCode/linear_f16
julia linear_f16.jl

echo -n 'Ended job at:';date
echo ""
exit
