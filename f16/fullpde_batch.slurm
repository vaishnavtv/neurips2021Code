#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE        #Do not propagate environment
#SBATCH --get-user-env=L     #Replicate login environment
  
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=fullpde_f16_1
#SBATCH --time=48:00:00            
#SBATCH --ntasks=24                
#SBATCH --ntasks-per-node=24      
#SBATCH --mem=100G                
#SBATCH --output=outQuad/fullpde_f16_1.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vaishnavtv@tamu.edu
##SBATCH --account=122756175418


##First Executable Line
ml purge
echo -n 'Started job at: ';date
echo ""
cd /scratch/user/vish0908/Documents/neuripsCode/f16
export JULIA_NUM_THREADS=24
julia fullpde_baseline_f16.jl

echo -n 'Ended job at:';date
echo ""
exit
