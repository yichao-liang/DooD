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

IT=300000
sbatch -J mn1E execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 5 -it $IT
sbatch -J mn1E execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 6 -it $IT
sbatch -J mn1E execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 7 -it $IT
sbatch -J mn1E execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 8 -it $IT
sbatch -J mn1E execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 9 -it $IT

sbatch -J mn2E execute.sh python sweep.py -m MT5 --beta 4 -ds mn --seed 5 -it $IT
sbatch -J mn2E execute.sh python sweep.py -m MT5 --beta 4 -ds mn --seed 6 -it $IT
sbatch -J mn2E execute.sh python sweep.py -m MT5 --beta 4 -ds mn --seed 7 -it $IT
sbatch -J mn2E execute.sh python sweep.py -m MT5 --beta 4 -ds mn --seed 8 -it $IT
sbatch -J mn2E execute.sh python sweep.py -m MT5 --beta 4 -ds mn --seed 9 -it $IT

sbatch -J mn3E execute.sh python sweep.py -m MTnT --beta 3 -ds mn --seed 5 -it $IT
sbatch -J mn3E execute.sh python sweep.py -m MTnT --beta 3 -ds mn --seed 6 -it $IT
sbatch -J mn3E execute.sh python sweep.py -m MTnT --beta 3 -ds mn --seed 7 -it $IT
sbatch -J mn3E execute.sh python sweep.py -m MTnT --beta 3 -ds mn --seed 8 -it $IT
sbatch -J mn3E execute.sh python sweep.py -m MTnT --beta 3 -ds mn --seed 9 -it $IT

sbatch -J mn4E execute.sh python sweep.py -m MTnT --beta 3.5 -ds mn --seed 5 -it $IT
sbatch -J mn4E execute.sh python sweep.py -m MTnT --beta 3.5 -ds mn --seed 6 -it $IT
sbatch -J mn4E execute.sh python sweep.py -m MTnT --beta 3.5 -ds mn --seed 7 -it $IT
sbatch -J mn4E execute.sh python sweep.py -m MTnT --beta 3.5 -ds mn --seed 8 -it $IT
sbatch -J mn4E execute.sh python sweep.py -m MTnT --beta 3.5 -ds mn --seed 9 -it $IT

sbatch -J mn5E execute.sh python sweep.py -m MTnT --beta 4 -ds mn --seed 5 -it $IT
sbatch -J mn5E execute.sh python sweep.py -m MTnT --beta 4 -ds mn --seed 6 -it $IT
sbatch -J mn5E execute.sh python sweep.py -m MTnT --beta 4 -ds mn --seed 7 -it $IT
sbatch -J mn5E execute.sh python sweep.py -m MTnT --beta 4 -ds mn --seed 8 -it $IT
sbatch -J mn5E execute.sh python sweep.py -m MTnT --beta 4 -ds mn --seed 9 -it $IT

sbatch -J mn6E execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 10 -it $IT
sbatch -J mn6E execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 11 -it $IT
sbatch -J mn6E execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 12 -it $IT
sbatch -J mn6E execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 13 -it $IT
sbatch -J mn6E execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 14 -it $IT

sbatch -J mn7E execute.sh python sweep.py -m MTS --beta 4 -ds mn --seed 10 -it $IT
sbatch -J mn7E execute.sh python sweep.py -m MTS --beta 4 -ds mn --seed 11 -it $IT
sbatch -J mn7E execute.sh python sweep.py -m MTS --beta 4 -ds mn --seed 12 -it $IT
sbatch -J mn7E execute.sh python sweep.py -m MTS --beta 4 -ds mn --seed 13 -it $IT
sbatch -J mn7E execute.sh python sweep.py -m MTS --beta 4 -ds mn --seed 14 -it $IT

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