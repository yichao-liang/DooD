#!/bin/bash  

# setting the checkpoint iteration for eval
IT=500000
# sbatch execute.sh python sweep.py --beta 2.5 -it $IT -m em --seed 0
# sbatch -J EVom execute.sh python sweep.py -m M --beta 4 -ds om --seed 0 -it $IT
# sbatch -J EVom execute.sh python sweep.py -m M --beta 4 -ds om --seed 1 -it $IT
# sbatch -J EVom execute.sh python sweep.py -m M --beta 4 -ds om --seed 2 -it $IT
# sbatch -J EVom execute.sh python sweep.py -m M --beta 4 -ds om --seed 3 -it $IT
# sbatch -J EVom execute.sh python sweep.py -m M --beta 4 -ds om --seed 4 -it $IT

# sbatch -J EVqd execute.sh python sweep.py -m M --beta 4 -ds qd --seed 0 -it $IT
# sbatch -J EVqd execute.sh python sweep.py -m M --beta 4 -ds qd --seed 1 -it $IT
# sbatch -J EVqd execute.sh python sweep.py -m M --beta 4 -ds qd --seed 2 -it $IT
# sbatch -J EVqd execute.sh python sweep.py -m M --beta 4 -ds qd --seed 3 -it $IT
# sbatch -J EVqd execute.sh python sweep.py -m M --beta 4 -ds qd --seed 4 -it $IT

# sbatch -J EVkm execute.sh python sweep.py -m M --beta 4 -ds km --seed 0 -it $IT
# sbatch -J EVkm execute.sh python sweep.py -m M --beta 4 -ds km --seed 1 -it $IT
# sbatch -J EVkm execute.sh python sweep.py -m M --beta 4 -ds km --seed 2 -it $IT
# sbatch -J EVkm execute.sh python sweep.py -m M --beta 4 -ds km --seed 3 -it $IT
# sbatch -J EVkm execute.sh python sweep.py -m M --beta 4 -ds km --seed 4 -it $IT

# sbatch -J EVem execute.sh python sweep.py -m M --beta 4 -ds em --seed 0 -it $IT
# sbatch -J EVem execute.sh python sweep.py -m M --beta 4 -ds em --seed 1 -it $IT
# sbatch -J EVem execute.sh python sweep.py -m M --beta 4 -ds em --seed 2 -it $IT
# sbatch -J EVem execute.sh python sweep.py -m M --beta 4 -ds em --seed 3 -it $IT
# sbatch -J EVem execute.sh python sweep.py -m M --beta 4 -ds em --seed 4 -it $IT

# sbatch -J EVmn execute.sh python sweep.py -m M --beta 4 -ds mn --seed 0 -it $IT
# sbatch -J EVmn execute.sh python sweep.py -m M --beta 4 -ds mn --seed 1 -it $IT
# sbatch -J EVmn execute.sh python sweep.py -m M --beta 4 -ds mn --seed 2 -it $IT
# sbatch -J EVmn execute.sh python sweep.py -m M --beta 4 -ds mn --seed 3 -it $IT
# sbatch -J EVmn execute.sh python sweep.py -m M --beta 4 -ds mn --seed 4 -it $IT

# sbatch -J EVDmn execute.sh python sweep.py -m DAIR_l --beta 5 -ds mn --seed 0 -it $IT
# sbatch -J EVDmn execute.sh python sweep.py -m DAIR_l --beta 5 -ds mn --seed 1 -it $IT
# sbatch -J EVDmn execute.sh python sweep.py -m DAIR_l --beta 5 -ds mn --seed 2 -it $IT
# sbatch -J EVDmn execute.sh python sweep.py -m DAIR_l --beta 5 -ds mn --seed 3 -it $IT
# sbatch -J EVDmn execute.sh python sweep.py -m DAIR_l --beta 5 -ds mn --seed 4 -it $IT

# sbatch -J EVDmn2 execute.sh python sweep.py -m DAIR_g --beta 5 -ds mn --seed 0 -it $IT
# sbatch -J EVDmn2 execute.sh python sweep.py -m DAIR_g --beta 5 -ds mn --seed 1 -it $IT
# sbatch -J EVDmn2 execute.sh python sweep.py -m DAIR_g --beta 5 -ds mn --seed 2 -it $IT

# sbatch -J EVDem execute.sh python sweep.py -m DAIR_l --beta 5 -ds em --seed 0 -it $IT
# sbatch -J EVDem execute.sh python sweep.py -m DAIR_l --beta 5 -ds em --seed 1 -it $IT
# sbatch -J EVDem execute.sh python sweep.py -m DAIR_l --beta 5 -ds em --seed 2 -it $IT

# sbatch -J EVDkm execute.sh python sweep.py -m DAIR_l --beta 5 -ds km --seed 0 -it $IT
# sbatch -J EVDkm execute.sh python sweep.py -m DAIR_l --beta 5 -ds km --seed 1 -it $IT
# sbatch -J EVDkm execute.sh python sweep.py -m DAIR_l --beta 5 -ds km --seed 2 -it $IT

# sbatch -J EVDqd execute.sh python sweep.py -m DAIR_l --beta 5 -ds qd --seed 0 -it $IT
# sbatch -J EVDqd execute.sh python sweep.py -m DAIR_l --beta 5 -ds qd --seed 1 -it $IT
# sbatch -J EVDqd execute.sh python sweep.py -m DAIR_l --beta 5 -ds qd --seed 2 -it $IT

# sbatch -J EVDom execute.sh python sweep.py -m DAIR_g --beta 5 -ds om --seed 0 -it $IT
# sbatch -J EVDom execute.sh python sweep.py -m DAIR_g --beta 5 -ds om --seed 1 -it $IT
# sbatch -J EVDom execute.sh python sweep.py -m DAIR_g --beta 5 -ds om --seed 2 -it $IT

# sbatch -J EVmnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 0 -it $IT
# sbatch -J EVmnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 1 -it $IT
# sbatch -J EVmnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 2 -it $IT
# sbatch -J EVmnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 3 -it $IT
# sbatch -J EVmnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 4 -it $IT

# sbatch -J EVmnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 0 -it $IT
# sbatch -J EVmnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 1 -it $IT
# sbatch -J EVmnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 2 -it $IT
# sbatch -J EVmnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 5 -it $IT
# sbatch -J EVmnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 6 -it $IT

# sbatch -J EVmnAb3 execute.sh python sweep.py -m Ma3 --beta 4 -ds mn --seed 0 -it $IT
# sbatch -J EVmnAb3 execute.sh python sweep.py -m Ma3 --beta 4 -ds mn --seed 1 -it $IT
# sbatch -J EVmnAb3 execute.sh python sweep.py -m Ma3 --beta 4 -ds mn --seed 2 -it $IT
# sbatch -J EVmnAb3 execute.sh python sweep.py -m Ma3 --beta 4 -ds mn --seed 3 -it 530000
# sbatch -J EVmnAb3 execute.sh python sweep.py -m Ma3 --beta 4 -ds mn --seed 4 -it $IT