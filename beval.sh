#!/bin/bash  

# setting the checkpoint iteration for eval
IT=320000
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m em --seed 0
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m em --seed 1
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m em --seed 3
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m em --seed 4
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m em --seed 5

sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m km --seed 1
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m km --seed 2
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m km --seed 3
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m km --seed 4
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m km --seed 5

sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m mn --seed 0
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m mn --seed 1
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m mn --seed 2
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m mn --seed 3
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m mn --seed 4

sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m om --seed 0
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m om --seed 1
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m om --seed 2
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m om --seed 3
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m om --seed 4

sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m qd --seed 5
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m qd --seed 6
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m qd --seed 7
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m qd --seed 8
sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m qd --seed 9