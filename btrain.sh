#!/bin/bash  

# e.g.:
# sbatch execute.sh python sweep.py -ds qd -m 20MT --seed 0 --beta 2.5

# sbatch execute.sh python sweep.py -m MT --beta 2.5 -ds qd --seed 0 
# sbatch execute.sh python sweep.py -m MT --beta 2.5 -ds qd --seed 1
# sbatch execute.sh python sweep.py -m MT --beta 2.5 -ds qd --seed 2
# sbatch execute.sh python sweep.py -m MT --beta 2.5 -ds qd --seed 3
# sbatch execute.sh python sweep.py -m MT --beta 2.5 -ds qd --seed 4

# sbatch execute.sh python sweep.py -m M --beta 2.5 -ds qd --seed 0 
# sbatch execute.sh python sweep.py -m M --beta 2.5 -ds qd --seed 1
# sbatch execute.sh python sweep.py -m M --beta 2.5 -ds qd --seed 2
# sbatch execute.sh python sweep.py -m M --beta 2.5 -ds qd --seed 3
# sbatch execute.sh python sweep.py -m M --beta 2.5 -ds qd --seed 4

# Full
# sbatch -J fkm execute.sh python sweep.py -m MT --beta 2.5 -ds km --seed 0 
# sbatch -J fkm execute.sh python sweep.py -m MT --beta 2.5 -ds km --seed 1
# sbatch -J fkm execute.sh python sweep.py -m MT --beta 2.5 -ds km --seed 2
# sbatch -J fkm execute.sh python sweep.py -m MT --beta 2.5 -ds km --seed 3
# sbatch -J fkm execute.sh python sweep.py -m MT --beta 2.5 -ds km --seed 4

# AIR
# sbatch -J akm execute.sh python sweep.py -m AIR --beta 2.5 -ds km --seed 0 -ct 
# sbatch -J akm execute.sh python sweep.py -m AIR --beta 2.5 -ds km --seed 1 -ct
# sbatch -J akm execute.sh python sweep.py -m AIR --beta 2.5 -ds km --seed 2
# sbatch -J akm execute.sh python sweep.py -m AIR --beta 2.5 -ds km --seed 3 -ct
# sbatch -J akm execute.sh python sweep.py -m AIR --beta 2.5 -ds km --seed 4 -ct

# sbatch -J akm1 execute.sh python sweep.py -m AIR --beta 1 -ds km --seed 5 -ct
# sbatch -J akm1 execute.sh python sweep.py -m AIR --beta 1 -ds km --seed 6 -ct
# sbatch -J akm1 execute.sh python sweep.py -m AIR --beta 1 -ds km --seed 7 -ct
# sbatch -J akm1 execute.sh python sweep.py -m AIR --beta 1 -ds km --seed 8 -ct
# sbatch -J akm1 execute.sh python sweep.py -m AIR --beta 1 -ds km --seed 9 -ct