#!/bin/bash  

# e.g.:
# sbatch execute.sh python sweep.py -ds qd -m 20MT --seed 0 --beta 2.5

# Train
# sbatch -J om execute.sh python sweep.py -m MT --beta 2.5 -ds om --seed 0 -trn -ct
# sbatch -J om execute.sh python sweep.py -m MT --beta 2.5 -ds om --seed 1 -trn -ct
# sbatch -J om execute.sh python sweep.py -m MT --beta 2.5 -ds om --seed 2 -trn -ct
# sbatch -J om execute.sh python sweep.py -m MT --beta 2.5 -ds om --seed 3 -trn -ct
# sbatch -J om execute.sh python sweep.py -m MT --beta 2.5 -ds om --seed 4 -trn -ct

# sbatch -J qd execute.sh python sweep.py -m MT --beta 2.5 -ds qd --seed 0 -trn -ct
# sbatch -J qd execute.sh python sweep.py -m MT --beta 2.5 -ds qd --seed 1 -trn -ct
# sbatch -J qd execute.sh python sweep.py -m MT --beta 2.5 -ds qd --seed 2 -trn -ct
# sbatch -J qd execute.sh python sweep.py -m MT --beta 2.5 -ds qd --seed 3 -trn -ct
# sbatch -J qd execute.sh python sweep.py -m MT --beta 2.5 -ds qd --seed 4 -trn -ct

# sbatch -J km execute.sh python sweep.py -m MT --beta 2.5 -ds km --seed 0 -trn -ct
# sbatch -J km execute.sh python sweep.py -m MT --beta 2.5 -ds km --seed 1 -trn -ct
# sbatch -J km execute.sh python sweep.py -m MT --beta 2.5 -ds km --seed 2 -trn -ct
# sbatch -J km execute.sh python sweep.py -m MT --beta 2.5 -ds km --seed 3 -trn -ct
# sbatch -J km execute.sh python sweep.py -m MT --beta 2.5 -ds km --seed 4 -trn -ct

# sbatch -J em execute.sh python sweep.py -m MT --beta 2.5 -ds em --seed 0 -trn -ct
# sbatch -J em execute.sh python sweep.py -m MT --beta 2.5 -ds em --seed 1 -trn -ct
# sbatch -J em execute.sh python sweep.py -m MT --beta 2.5 -ds em --seed 2 -trn -ct
# sbatch -J em execute.sh python sweep.py -m MT --beta 2.5 -ds em --seed 3 -trn -ct
# sbatch -J em execute.sh python sweep.py -m MT --beta 2.5 -ds em --seed 4 -trn -ct

# sbatch -J mn1 execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 5 -trn
# sbatch -J mn1 execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 6 -trn
# sbatch -J mn1 execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 7 -trn
# sbatch -J mn1 execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 8 -trn
# sbatch -J mn1 execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 9 -trn

# sbatch -J mn2 execute.sh python sweep.py -m MT5 --beta 4 -ds mn --seed 5 -trn
# sbatch -J mn2 execute.sh python sweep.py -m MT5 --beta 4 -ds mn --seed 6 -trn
# sbatch -J mn2 execute.sh python sweep.py -m MT5 --beta 4 -ds mn --seed 7 -trn
# sbatch -J mn2 execute.sh python sweep.py -m MT5 --beta 4 -ds mn --seed 8 -trn
# sbatch -J mn2 execute.sh python sweep.py -m MT5 --beta 4 -ds mn --seed 9 -trn

# sbatch -J mn3 execute.sh python sweep.py -m MTnT --beta 3 -ds mn --seed 5 -trn
# sbatch -J mn3 execute.sh python sweep.py -m MTnT --beta 3 -ds mn --seed 6 -trn
# sbatch -J mn3 execute.sh python sweep.py -m MTnT --beta 3 -ds mn --seed 7 -trn
# sbatch -J mn3 execute.sh python sweep.py -m MTnT --beta 3 -ds mn --seed 8 -trn
# sbatch -J mn3 execute.sh python sweep.py -m MTnT --beta 3 -ds mn --seed 9 -trn

# sbatch -J mn4 execute.sh python sweep.py -m MTnT --beta 3.5 -ds mn --seed 5 -trn
# sbatch -J mn4 execute.sh python sweep.py -m MTnT --beta 3.5 -ds mn --seed 6 -trn
# sbatch -J mn4 execute.sh python sweep.py -m MTnT --beta 3.5 -ds mn --seed 7 -trn
# sbatch -J mn4 execute.sh python sweep.py -m MTnT --beta 3.5 -ds mn --seed 8 -trn
# sbatch -J mn4 execute.sh python sweep.py -m MTnT --beta 3.5 -ds mn --seed 9 -trn

# sbatch -J mn5 execute.sh python sweep.py -m MTnT --beta 4 -ds mn --seed 5 -trn
# sbatch -J mn5 execute.sh python sweep.py -m MTnT --beta 4 -ds mn --seed 6 -trn
# sbatch -J mn5 execute.sh python sweep.py -m MTnT --beta 4 -ds mn --seed 7 -trn
# sbatch -J mn5 execute.sh python sweep.py -m MTnT --beta 4 -ds mn --seed 8 -trn
# sbatch -J mn5 execute.sh python sweep.py -m MTnT --beta 4 -ds mn --seed 9 -trn

# sbatch -J mn6 execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 10 -trn
# sbatch -J mn6 execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 11 -trn
# sbatch -J mn6 execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 12 -trn
# sbatch -J mn6 execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 13 -trn
# sbatch -J mn6 execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 14 -trn

# sbatch -J mn7 execute.sh python sweep.py -m MTS --beta 4 -ds mn --seed 10 -trn
# sbatch -J mn7 execute.sh python sweep.py -m MTS --beta 4 -ds mn --seed 11 -trn
# sbatch -J mn7 execute.sh python sweep.py -m MTS --beta 4 -ds mn --seed 12 -trn
# sbatch -J mn7 execute.sh python sweep.py -m MTS --beta 4 -ds mn --seed 13 -trn
# sbatch -J mn7 execute.sh python sweep.py -m MTS --beta 4 -ds mn --seed 14 -trn


# AIR
# sbatch -J Amn execute.sh python sweep.py -m AIR --beta 4 -ds mn --seed 0 -trn
# sbatch -J Amn execute.sh python sweep.py -m AIR --beta 4 -ds mn --seed 1 -trn
# sbatch -J Amn execute.sh python sweep.py -m AIR --beta 4 -ds mn --seed 2 -trn
# sbatch -J Amn execute.sh python sweep.py -m AIR --beta 4 -ds mn --seed 3 -trn
# sbatch -J Amn execute.sh python sweep.py -m AIR --beta 4 -ds mn --seed 4 -trn

# sbatch -J Dmn1 execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 0 -trn -ct
# sbatch -J Dmn1 execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 1 -trn -ct
# sbatch -J Dmn1 execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 2 -trn -ct
# sbatch -J Dmn1 execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 3 -trn -ct
# sbatch -J Dmn1 execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 4 -trn -ct
# sbatch -J Dmn1 execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 5 -trn -ct
# sbatch -J Dmn1 execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 6 -trn -ct
# sbatch -J Dmn2 execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 7 -trn
# sbatch -J Dmn2 execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 8 -trn
# sbatch -J Dmn2 execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 9 -trn