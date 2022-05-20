#!/bin/bash

#SBATCH --job-name=
#SBATCH --partition=tenenbaum
#SBATCH --qos=tenenbaum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH --constraint=5GB
##SBATCH --constraint=3GB
#SBATCH --time=1-00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=ycliang6@gmail.edu
#SBATCH --output=output/%x.%j.out
#SBATCH --error=output/%x.%j.err

source activate glot
cd /om2/user/ycliang/hierarchical/handwritten_characters

$@