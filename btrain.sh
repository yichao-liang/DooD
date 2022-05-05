#!/bin/bash  

# e.g.:
# sbatch execute.sh python sweep.py -ds qd -m 20MT --seed 0 --beta 2.5
# Train Full model
# sbatch -J om execute.sh python sweep.py -m M --beta 4 -ds om --seed 0 -trn -ct
# sbatch -J om execute.sh python sweep.py -m M --beta 4 -ds om --seed 1 -trn -ct
# sbatch -J om execute.sh python sweep.py -m M --beta 4 -ds om --seed 2 -trn -ct
# sbatch -J om execute.sh python sweep.py -m M --beta 4 -ds om --seed 3 -trn -ct
# sbatch -J om execute.sh python sweep.py -m M --beta 4 -ds om --seed 4 -trn -ct

# sbatch -J qd execute.sh python sweep.py -m M --beta 4 -ds qd --seed 0 -trn -ct
# sbatch -J qd execute.sh python sweep.py -m M --beta 4 -ds qd --seed 1 -trn -ct
# sbatch -J qd execute.sh python sweep.py -m M --beta 4 -ds qd --seed 2 -trn -ct
# sbatch -J qd execute.sh python sweep.py -m M --beta 4 -ds qd --seed 3 -trn -ct
# sbatch -J qd execute.sh python sweep.py -m M --beta 4 -ds qd --seed 4 -trn -ct

# sbatch -J km execute.sh python sweep.py -m M --beta 4 -ds km --seed 0 -trn -ct
# sbatch -J km execute.sh python sweep.py -m M --beta 4 -ds km --seed 1 -trn -ct
# sbatch -J km execute.sh python sweep.py -m M --beta 4 -ds km --seed 2 -trn -ct
# sbatch -J km execute.sh python sweep.py -m M --beta 4 -ds km --seed 3 -trn -ct
# sbatch -J km execute.sh python sweep.py -m M --beta 4 -ds km --seed 4 -trn -ct

# sbatch -J em execute.sh python sweep.py -m M --beta 4 -ds em --seed 0 -trn -ct
# sbatch -J em execute.sh python sweep.py -m M --beta 4 -ds em --seed 1 -trn -ct
# sbatch -J em execute.sh python sweep.py -m M --beta 4 -ds em --seed 2 -trn -ct
# sbatch -J em execute.sh python sweep.py -m M --beta 4 -ds em --seed 3 -trn -ct
# sbatch -J em execute.sh python sweep.py -m M --beta 4 -ds em --seed 4 -trn -ct

# sbatch -J mn execute.sh python sweep.py -m M --beta 4 -ds mn --seed 0 -trn
# sbatch -J mn execute.sh python sweep.py -m M --beta 4 -ds mn --seed 1 -trn
sbatch -J mn execute.sh python sweep.py -m M --beta 4 -ds mn --seed 2 -trn -ct
# sbatch -J mn execute.sh python sweep.py -m M --beta 4 -ds mn --seed 3 -trn
sbatch -J mn execute.sh python sweep.py -m M --beta 4 -ds mn --seed 4 -trn -ct

# AIR (Baseline)
# sbatch -J Amn execute.sh python sweep.py -m AIR --beta 4 -ds mn --seed 0 -trn
# sbatch -J Amn execute.sh python sweep.py -m AIR --beta 4 -ds mn --seed 1 -trn
# sbatch -J Amn execute.sh python sweep.py -m AIR --beta 4 -ds mn --seed 2 -trn
# sbatch -J Amn execute.sh python sweep.py -m AIR --beta 4 -ds mn --seed 3 -trn
# sbatch -J Amn execute.sh python sweep.py -m AIR --beta 4 -ds mn --seed 4 -trn

# sbatch -J Dmn execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 0 -trn -ct
# sbatch -J Dmn execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 1 -trn -ct
# sbatch -J Dmn execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 2 -trn -ct
# sbatch -J Dmn execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 3 -trn -ct
# sbatch -J Dmn execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 4 -trn -ct

# Ablations
# sbatch -J mnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 0 -trn -ct
# sbatch -J mnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 1 -trn -ct
# sbatch -J mnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 2 -trn -ct
# sbatch -J mnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 3 -trn -ct
# sbatch -J mnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 4 -trn -ct

# sbatch -J mnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 0 -trn -ct
# sbatch -J mnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 1 -trn -ct
# sbatch -J mnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 2 -trn -ct
# sbatch -J mnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 3 -trn -ct
# sbatch -J mnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 4 -trn -ct