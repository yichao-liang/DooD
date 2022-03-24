#!/bin/bash  
# exp with linear sum

sbatch execute.sh python sweep.py --beta 2.5 -m omv1 --seed 0
# sbatch execute.sh python sweep.py --beta 2.5 -m omv1 --seed 1
sbatch execute.sh python sweep.py --beta 2.5 -m omv1 --seed 2

# sbatch execute.sh python sweep.py --beta 1.5 -m om5 --seed 0
sbatch execute.sh python sweep.py --beta 1.5 -m om5 --seed 1
sbatch execute.sh python sweep.py --beta 1.5 -m om5 --seed 2

sbatch execute.sh python sweep.py --beta 2 -m omv2 --seed 0
sbatch execute.sh python sweep.py --beta 2 -m omv2 --seed 1
# sbatch execute.sh python sweep.py --beta 2 -m omv2 --seed 2

sbatch execute.sh python sweep.py --beta 2 -m omv3 --seed 0
# sbatch execute.sh python sweep.py --beta 2 -m omv3 --seed 1
sbatch execute.sh python sweep.py --beta 2 -m omv3 --seed 2
