#!/bin/bash

<<<<<<< HEAD
#SBATCH --job-name=
=======
#SBATCH --job-name=full-seq_beta
>>>>>>> 12f3d0d3476f5242d9364abfdd958a334db6871e
#SBATCH --partition=tenenbaum
#SBATCH --qos=tenenbaum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=ycliang6@gmail.edu
<<<<<<< HEAD
#SBATCH --output=output/%x.%j.out
#SBATCH --error=output/%x.%j.err
=======
#SBATCH --output=output/S-%x.%j.out
#SBATCH --error=output/S-%x.%j.err
>>>>>>> 12f3d0d3476f5242d9364abfdd958a334db6871e

source activate glot
cd /om2/user/ycliang/hierarchical/handwritten_characters

$@