#!/bin/bash  

# e.g.:
# sbatch execute.sh python sweep.py -ds qd -m 20MT --seed 0 --beta 2.5
# Train Full model
# sbatch -J om execute.sh python sweep.py -m M --beta 4 -ds om --seed 0 -trn -ct
# sbatch -J om execute.sh python sweep.py -m M --beta 4 -ds om --seed 1 -trn
# sbatch -J om execute.sh python sweep.py -m M --beta 4 -ds om --seed 2 -trn
# sbatch -J om execute.sh python sweep.py -m M --beta 4 -ds om --seed 3 -trn
# sbatch -J om execute.sh python sweep.py -m M --beta 4 -ds om --seed 4 -trn

# sbatch -J qd execute.sh python sweep.py -m M --beta 4 -ds qd --seed 0 -trn
# sbatch -J qd execute.sh python sweep.py -m M --beta 4 -ds qd --seed 1 -trn
# sbatch -J qd execute.sh python sweep.py -m M --beta 4 -ds qd --seed 2 -trn
# sbatch -J qd execute.sh python sweep.py -m M --beta 4 -ds qd --seed 3 -trn
# sbatch -J qd execute.sh python sweep.py -m M --beta 4 -ds qd --seed 4 -trn

# sbatch -J km execute.sh python sweep.py -m M --beta 4 -ds km --seed 0 -trn
# sbatch -J km execute.sh python sweep.py -m M --beta 4 -ds km --seed 1 -trn
# sbatch -J km execute.sh python sweep.py -m M --beta 4 -ds km --seed 2 -trn
# sbatch -J km execute.sh python sweep.py -m M --beta 4 -ds km --seed 3 -trn
# sbatch -J km execute.sh python sweep.py -m M --beta 4 -ds km --seed 4 -trn

# sbatch -J em execute.sh python sweep.py -m M --beta 4 -ds em --seed 0 -trn
# sbatch -J em execute.sh python sweep.py -m M --beta 4 -ds em --seed 1 -trn
# sbatch -J em execute.sh python sweep.py -m M --beta 4 -ds em --seed 2 -trn
# sbatch -J em execute.sh python sweep.py -m M --beta 4 -ds em --seed 3 -trn
# sbatch -J em execute.sh python sweep.py -m M --beta 4 -ds em --seed 4 -trn

# sbatch -J mn execute.sh python sweep.py -m M --beta 4 -ds mn --seed 0 -trn
# sbatch -J mn execute.sh python sweep.py -m M --beta 4 -ds mn --seed 1 -trn
# sbatch -J mn execute.sh python sweep.py -m M --beta 4 -ds mn --seed 2 -trn
# sbatch -J mn execute.sh python sweep.py -m M --beta 4 -ds mn --seed 3 -trn
# sbatch -J mn execute.sh python sweep.py -m M --beta 4 -ds mn --seed 4 -trn

# sbatch -J omT execute.sh python sweep.py -m MT --beta 4 -ds om --seed 0 -trn -ct
# sbatch -J omT execute.sh python sweep.py -m MT --beta 4 -ds om --seed 1 -trn -ct
# sbatch -J omT execute.sh python sweep.py -m MT --beta 4 -ds om --seed 2 -trn -ct
# sbatch -J omT execute.sh python sweep.py -m MT --beta 4 -ds om --seed 3 -trn -ct
# sbatch -J omT execute.sh python sweep.py -m MT --beta 4 -ds om --seed 4 -trn -ct
# sbatch -J omT execute.sh python sweep.py -m MT --beta 4 -ds om --seed 6 -trn -ct

# sbatch -J qdT execute.sh python sweep.py -m MT --beta 4 -ds qd --seed 0 -trn -ct
# sbatch -J qdT execute.sh python sweep.py -m MT --beta 4 -ds qd --seed 1 -trn -ct
# sbatch -J qdT execute.sh python sweep.py -m MT --beta 4 -ds qd --seed 2 -trn -ct
# sbatch -J qdT execute.sh python sweep.py -m MT --beta 4 -ds qd --seed 3 -trn -ct
# sbatch -J qdT execute.sh python sweep.py -m MT --beta 4 -ds qd --seed 4 -trn -ct

# sbatch -J kmT execute.sh python sweep.py -m MT --beta 4 -ds km --seed 0 -trn -ct
# sbatch -J kmT execute.sh python sweep.py -m MT --beta 4 -ds km --seed 1 -trn -ct
# sbatch -J kmT execute.sh python sweep.py -m MT --beta 4 -ds km --seed 2 -trn -ct
# sbatch -J kmT execute.sh python sweep.py -m MT --beta 4 -ds km --seed 3 -trn -ct
# sbatch -J kmT execute.sh python sweep.py -m MT --beta 4 -ds km --seed 4 -trn -ct

# sbatch -J emT execute.sh python sweep.py -m MT --beta 4 -ds em --seed 0 -trn -ct
# sbatch -J emT execute.sh python sweep.py -m MT --beta 4 -ds em --seed 1 -trn -ct
# sbatch -J emT execute.sh python sweep.py -m MT --beta 4 -ds em --seed 2 -trn -ct
# sbatch -J emT execute.sh python sweep.py -m MT --beta 4 -ds em --seed 3 -trn -ct
# sbatch -J emT execute.sh python sweep.py -m MT --beta 4 -ds em --seed 4 -trn -ct

# sbatch -J mnT execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 0 -trn -ct
# sbatch -J mnT execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 1 -trn -ct
# sbatch -J mnT execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 2 -trn -ct
# sbatch -J mnT execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 3 -trn -ct
# sbatch -J mnT execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 4 -trn -ct

# sbatch -J syT execute.sh python sweep.py -m MT --beta 4 -ds sy --seed 0 -trn
# sbatch -J syT execute.sh python sweep.py -m MT --beta 4 -ds sy --seed 1 -trn
# sbatch -J syT execute.sh python sweep.py -m MT --beta 4 -ds sy --seed 2 -trn
# sbatch -J syT execute.sh python sweep.py -m MT --beta 4 -ds sy --seed 3 -trn
# sbatch -J syT execute.sh python sweep.py -m MT --beta 4 -ds sy --seed 4 -trn

sbatch -J MTmlp execute.sh python sweep.py -m MTmlp --beta 4 -ds mn --seed 0 -trn
sbatch -J MTmlp execute.sh python sweep.py -m MTmlp --beta 4 -ds mn --seed 1 -trn
sbatch -J MTmlp execute.sh python sweep.py -m MTmlp --beta 4 -ds mn --seed 2 -trn

# AIR
# sbatch -J Amn execute.sh python sweep.py -m AIR_l --beta 5 -ds mn --seed 0 -trn -ct
# sbatch -J Amn execute.sh python sweep.py -m AIR_l --beta 5 -ds mn --seed 1 -trn -ct
# sbatch -J Amn execute.sh python sweep.py -m AIR_l --beta 5 -ds mn --seed 2 -trn -ct

# sbatch -J Akm execute.sh python sweep.py -m AIR_l --beta 5 -ds km --seed 0 -trn -ct
# sbatch -J Akm execute.sh python sweep.py -m AIR_l --beta 5 -ds km --seed 1 -trn -ct
# sbatch -J Akm execute.sh python sweep.py -m AIR_l --beta 5 -ds km --seed 2 -trn -ct

# sbatch -J Aem execute.sh python sweep.py -m AIR_l --beta 5 -ds em --seed 0 -trn -ct
# sbatch -J Aem execute.sh python sweep.py -m AIR_l --beta 5 -ds em --seed 1 -trn -ct
# sbatch -J Aem execute.sh python sweep.py -m AIR_l --beta 5 -ds em --seed 2 -trn -ct

# sbatch -J Aqd execute.sh python sweep.py -m AIR_l --beta 5 -ds qd --seed 0 -trn
# sbatch -J Aqd execute.sh python sweep.py -m AIR_l --beta 5 -ds qd --seed 1 -trn
# sbatch -J Aqd execute.sh python sweep.py -m AIR_l --beta 5 -ds qd --seed 2 -trn

# sbatch -J Aom execute.sh python sweep.py -m AIR_g --beta 5 -ds om --seed 0 -trn
# sbatch -J Aom execute.sh python sweep.py -m AIR_g --beta 5 -ds om --seed 1 -trn
# sbatch -J Aom execute.sh python sweep.py -m AIR_g --beta 5 -ds om --seed 2 -trn

# sbatch -J Dmn execute.sh python sweep.py -m DAIR_l --beta 5 -ds mn --seed 0 -trn -ct
# sbatch -J Dmn execute.sh python sweep.py -m DAIR_l --beta 5 -ds mn --seed 1 -trn -ct
# sbatch -J Dmn execute.sh python sweep.py -m DAIR_l --beta 5 -ds mn --seed 2 -trn -ct
# sbatch -J Dmn execute.sh python sweep.py -m DAIR_l --beta 5 -ds mn --seed 3 -trn
# sbatch -J Dmn execute.sh python sweep.py -m DAIR_l --beta 5 -ds mn --seed 4 -trn

# sbatch -J Dmn2 execute.sh python sweep.py -m DAIR_g --beta 5 -ds mn --seed 0 -trn -ct
# sbatch -J Dmn2 execute.sh python sweep.py -m DAIR_g --beta 5 -ds mn --seed 1 -trn -ct
# sbatch -J Dmn2 execute.sh python sweep.py -m DAIR_g --beta 5 -ds mn --seed 2 -trn -ct

# sbatch -J Dem execute.sh python sweep.py -m DAIR_l --beta 5 -ds em --seed 0 -trn -ct
# sbatch -J Dem execute.sh python sweep.py -m DAIR_l --beta 5 -ds em --seed 1 -trn -ct
# sbatch -J Dem execute.sh python sweep.py -m DAIR_l --beta 5 -ds em --seed 2 -trn -ct
# sbatch -J Dem execute.sh python sweep.py -m DAIR_l --beta 4 -ds em --seed 3 -trn
# sbatch -J Dem execute.sh python sweep.py -m DAIR_l --beta 4 -ds em --seed 4 -trn

# sbatch -J Dem2 execute.sh python sweep.py -m DAIR_g --beta 5 -ds em --seed 0 -trn -ct
# sbatch -J Dem2 execute.sh python sweep.py -m DAIR_g --beta 5 -ds em --seed 1 -trn -ct
# sbatch -J Dem2 execute.sh python sweep.py -m DAIR_g --beta 5 -ds em --seed 2 -trn -ct

# sbatch -J Dkm execute.sh python sweep.py -m DAIR_l --beta 5 -ds km --seed 0 -trn
# sbatch -J Dkm execute.sh python sweep.py -m DAIR_l --beta 5 -ds km --seed 1 -trn
# sbatch -J Dkm execute.sh python sweep.py -m DAIR_l --beta 5 -ds km --seed 2 -trn
# sbatch -J Dkm execute.sh python sweep.py -m DAIR_l --beta 4 -ds km --seed 3 -trn
# sbatch -J Dkm execute.sh python sweep.py -m DAIR_l --beta 4 -ds km --seed 4 -trn

# sbatch -J Dkm2 execute.sh python sweep.py -m DAIR_g --beta 5 -ds km --seed 0 -trn -ct
# sbatch -J Dkm2 execute.sh python sweep.py -m DAIR_g --beta 5 -ds km --seed 1 -trn -ct
# sbatch -J Dkm2 execute.sh python sweep.py -m DAIR_g --beta 5 -ds km --seed 2 -trn -ct

# sbatch -J Dqd execute.sh python sweep.py -m DAIR_l --beta 5 -ds qd --seed 0 -trn -ct
# sbatch -J Dqd execute.sh python sweep.py -m DAIR_l --beta 5 -ds qd --seed 1 -trn -ct
# sbatch -J Dqd execute.sh python sweep.py -m DAIR_l --beta 5 -ds qd --seed 2 -trn -ct
# sbatch -J Dqd execute.sh python sweep.py -m DAIR_l --beta 4 -ds qd --seed 3 -trn
# sbatch -J Dqd execute.sh python sweep.py -m DAIR_l --beta 4 -ds qd --seed 4 -trn

# sbatch -J Dqd2 execute.sh python sweep.py -m DAIR_g --beta 5 -ds qd --seed 0 -trn -ct
# sbatch -J Dqd2 execute.sh python sweep.py -m DAIR_g --beta 5 -ds qd --seed 1 -trn -ct
# sbatch -J Dqd2 execute.sh python sweep.py -m DAIR_g --beta 5 -ds qd --seed 2 -trn -ct
 
# sbatch -J Dom execute.sh python sweep.py -m DAIR_g --beta 5 -ds om --seed 0 -trn -ct
# sbatch -J Dom execute.sh python sweep.py -m DAIR_g --beta 5 -ds om --seed 1 -trn -ct
# sbatch -J Dom execute.sh python sweep.py -m DAIR_g --beta 5 -ds om --seed 2 -trn -ct
# sbatch -J Dom execute.sh python sweep.py -m DAIR_g --beta 4 -ds om --seed 3 -trn
# sbatch -J Dom execute.sh python sweep.py -m DAIR_g --beta 4 -ds om --seed 4 -trn

# Ablations
# sbatch -J mnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 0 -trn -ct
# sbatch -J mnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 1 -trn -ct
# sbatch -J mnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 2 -trn -ct
# sbatch -J mnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 3 -trn -ct
# sbatch -J mnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 4 -trn -ct

# sbatch -J mnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 0 -trn -ct
# sbatch -J mnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 1 -trn -ct
# sbatch -J mnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 2 -trn -ct
# sbatch -J mnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 5 -trn -ct
# sbatch -J mnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 6 -trn -ct

# sbatch -J mnAb3 execute.sh python sweep.py -m Ma3 --beta 4 -ds mn --seed 0 -trn -ct
# sbatch -J mnAb3 execute.sh python sweep.py -m Ma3 --beta 4 -ds mn --seed 1 -trn -ct
# sbatch -J mnAb3 execute.sh python sweep.py -m Ma3 --beta 4 -ds mn --seed 2 -trn -ct
# sbatch -J mnAb3 execute.sh python sweep.py -m Ma3 --beta 4 -ds mn --seed 3 -trn -ct
# sbatch -J mnAb3 execute.sh python sweep.py -m Ma3 --beta 4 -ds mn --seed 4 -trn -ct