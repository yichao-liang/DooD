#!/bin/bash  

# setting the checkpoint iteration for eval
IT=320000
# sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m em --seed 0
# sbatch -J omEv execute.sh python sweep.py -m MT --beta 2.5 -ds om --seed 0 it 500000
# sbatch -J omEv execute.sh python sweep.py -m MT --beta 2.5 -ds om --seed 1 it 500000
# sbatch -J omEv execute.sh python sweep.py -m MT --beta 2.5 -ds om --seed 2 it 500000
# sbatch -J omEv execute.sh python sweep.py -m MT --beta 2.5 -ds om --seed 3 it 500000
# sbatch -J omEv execute.sh python sweep.py -m MT --beta 2.5 -ds om --seed 4 it 500000

# sbatch -J qdEv execute.sh python sweep.py -m MT --beta 2.5 -ds qd --seed 0 -it 500000
# sbatch -J qdEv execute.sh python sweep.py -m MT --beta 2.5 -ds qd --seed 1 -it 500000
# sbatch -J qdEv execute.sh python sweep.py -m MT --beta 2.5 -ds qd --seed 2 -it 500000
# sbatch -J qdEv execute.sh python sweep.py -m MT --beta 2.5 -ds qd --seed 3 -it 500000
# sbatch -J qdEv execute.sh python sweep.py -m MT --beta 2.5 -ds qd --seed 4 -it 500000

# sbatch -J kmEv execute.sh python sweep.py -m MT --beta 2.5 -ds km --seed 0 -it 500000
# sbatch -J kmEv execute.sh python sweep.py -m MT --beta 2.5 -ds km --seed 1 -it 500000
# sbatch -J kmEv execute.sh python sweep.py -m MT --beta 2.5 -ds km --seed 2 -it 500000
# sbatch -J kmEv execute.sh python sweep.py -m MT --beta 2.5 -ds km --seed 3 -it 500000
# sbatch -J kmEv execute.sh python sweep.py -m MT --beta 2.5 -ds km --seed 4 -it 500000

# sbatch -J emEv execute.sh python sweep.py -m MT --beta 2.5 -ds em --seed 0 -it 400000
# sbatch -J emEv execute.sh python sweep.py -m MT --beta 2.5 -ds em --seed 1 -it 400000
# sbatch -J emEv execute.sh python sweep.py -m MT --beta 2.5 -ds em --seed 2 -it 400000
# sbatch -J emEv execute.sh python sweep.py -m MT --beta 2.5 -ds em --seed 3 -it 400000
# sbatch -J emEv execute.sh python sweep.py -m MT --beta 2.5 -ds em --seed 4 -it 400000

IT=500000
sbatch -J EVmn7 execute.sh python sweep.py -m M --beta 4 -ds mn --seed 0 -it $IT
sbatch -J EVmn7 execute.sh python sweep.py -m M --beta 4 -ds mn --seed 1 -it $IT
sbatch -J EVmn7 execute.sh python sweep.py -m M --beta 4 -ds mn --seed 2 -it $IT
sbatch -J EVmn7 execute.sh python sweep.py -m M --beta 4 -ds mn --seed 3 -it $IT
sbatch -J EVmn7 execute.sh python sweep.py -m M --beta 4 -ds mn --seed 4 -it $IT

sbatch -J EVmn8 execute.sh python sweep.py -m MS --beta 4 -ds mn --seed 0 -it $IT
sbatch -J EVmn8 execute.sh python sweep.py -m MS --beta 4 -ds mn --seed 1 -it $IT
sbatch -J EVmn8 execute.sh python sweep.py -m MS --beta 4 -ds mn --seed 2 -it $IT
sbatch -J EVmn8 execute.sh python sweep.py -m MS --beta 4 -ds mn --seed 3 -it $IT
sbatch -J EVmn8 execute.sh python sweep.py -m MS --beta 4 -ds mn --seed 4 -it $IT

# AIR
# sbatch -J AmnEv execute.sh python sweep.py -m AIR --beta 4 -ds mn --seed 0 -it 500000
# sbatch -J AmnEv execute.sh python sweep.py -m AIR --beta 4 -ds mn --seed 1 -it 500000
# sbatch -J AmnEv execute.sh python sweep.py -m AIR --beta 4 -ds mn --seed 2 -it 500000
# sbatch -J AmnEv execute.sh python sweep.py -m AIR --beta 4 -ds mn --seed 3 -it 500000
# sbatch -J AmnEv execute.sh python sweep.py -m AIR --beta 4 -ds mn --seed 4 -it 500000

# sbatch -J Dmn1Ev execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 0 -it 300000
# sbatch -J Dmn1Ev execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 1 -it 300000
# sbatch -J Dmn1Ev execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 2 -it 300000
# sbatch -J Dmn1Ev execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 3 -it 300000
# sbatch -J Dmn1Ev execute.sh python sweep.py -m DAIR --beta 4 -ds mn --seed 4 -it 300000