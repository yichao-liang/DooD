#!/bin/bash

#SBATCH --job-name=glot
#SBATCH --partition=tenenbaum
#SBATCH --qos=tenenbaum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=0-1:00
# SBATCH --mail-type=begin
# SBATCH --mail-type=end
# SBATCH --mail-type=fail
# SBATCH --mail-user=ycliang6@gmail.edu
#SBATCH --output=output/S-%x.%j.out
#SBATCH --error=output/S-%x.%j.err

source activate glot
cd /om2/user/ycliang/hierarchical/handwritten_characters

$@