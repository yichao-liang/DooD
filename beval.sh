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

# sbatch -J EVom2 execute.sh python sweep.py -m MT --beta 4 -ds om --seed 0 -it $IT
# sbatch -J EVom2 execute.sh python sweep.py -m MT --beta 4 -ds om --seed 1 -it 510000
# sbatch -J EVom2 execute.sh python sweep.py -m MT --beta 4 -ds om --seed 2 -it 510000
# sbatch -J EVom2 execute.sh python sweep.py -m MT --beta 4 -ds om --seed 3 -it $IT
# sbatch -J EVom2 execute.sh python sweep.py -m MT --beta 4 -ds om --seed 4 -it $IT
# sbatch -J EVom2 execute.sh python sweep.py -m MT --beta 4 -ds om --seed 6 -it $IT

# sbatch -J EVqd2 execute.sh python sweep.py -m MT --beta 4 -ds qd --seed 0 -it $IT
# sbatch -J EVqd2 execute.sh python sweep.py -m MT --beta 4 -ds qd --seed 1 -it 550000
# sbatch -J EVqd2 execute.sh python sweep.py -m MT --beta 4 -ds qd --seed 2 -it 550000
# sbatch -J EVqd2 execute.sh python sweep.py -m MT --beta 4 -ds qd --seed 3 -it 600000
# sbatch -J EVqd2 execute.sh python sweep.py -m MT --beta 4 -ds qd --seed 4 -it 550000

# sbatch -J EVkm2 execute.sh python sweep.py -m MT --beta 4 -ds km --seed 0 -it $IT
# sbatch -J EVkm2 execute.sh python sweep.py -m MT --beta 4 -ds km --seed 1 -it $IT
# sbatch -J EVkm2 execute.sh python sweep.py -m MT --beta 4 -ds km --seed 2 -it $IT
# sbatch -J EVkm2 execute.sh python sweep.py -m MT --beta 4 -ds km --seed 3 -it $IT
# sbatch -J EVkm2 execute.sh python sweep.py -m MT --beta 4 -ds km --seed 4 -it $IT

# sbatch -J EVem2 execute.sh python sweep.py -m MT --beta 4 -ds em --seed 0 -it $IT
# sbatch -J EVem2 execute.sh python sweep.py -m MT --beta 4 -ds em --seed 1 -it $IT
# sbatch -J EVem2 execute.sh python sweep.py -m MT --beta 4 -ds em --seed 2 -it $IT
# sbatch -J EVem2 execute.sh python sweep.py -m MT --beta 4 -ds em --seed 3 -it $IT
# sbatch -J EVem2 execute.sh python sweep.py -m MT --beta 4 -ds em --seed 4 -it $IT

# sbatch -J EVmn2 execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 0 -it $IT
# sbatch -J EVmn2 execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 1 -it $IT
# sbatch -J EVmn2 execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 2 -it $IT
# sbatch -J EVmn2 execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 3 -it $IT
# sbatch -J EVmn2 execute.sh python sweep.py -m MT --beta 4 -ds mn --seed 4 -it $IT

# sbatch -J EVsy2 execute.sh python sweep.py -m MT --beta 4 -ds sy --seed 0 -it $IT
# sbatch -J EVsy2 execute.sh python sweep.py -m MT --beta 4 -ds sy --seed 1 -it $IT
# sbatch -J EVsy2 execute.sh python sweep.py -m MT --beta 4 -ds sy --seed 2 -it $IT
# sbatch -J EVsy2 execute.sh python sweep.py -m MT --beta 4 -ds sy --seed 3 -it $IT
# sbatch -J EVsy2 execute.sh python sweep.py -m MT --beta 4 -ds sy --seed 4 -it $IT

# # sbatch -J EVAmn execute.sh python sweep.py -m AIR_l --beta 5 -ds mn --seed 0 -it $IT
# # sbatch -J EVAmn execute.sh python sweep.py -m AIR_l --beta 5 -ds mn --seed 1 -it $IT
# # sbatch -J EVAmn execute.sh python sweep.py -m AIR_l --beta 5 -ds mn --seed 2 -it $IT

# sbatch -J EVDmn execute.sh python sweep.py -m DAIR_l --beta 5 -ds mn --seed 0 -it $IT
# sbatch -J EVDmn execute.sh python sweep.py -m DAIR_l --beta 5 -ds mn --seed 1 -it $IT
# sbatch -J EVDmn execute.sh python sweep.py -m DAIR_l --beta 5 -ds mn --seed 2 -it $IT

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

sbatch -J EVmnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 0 -it $IT
# sbatch -J EVmnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 1 -it $IT
# sbatch -J EVmnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 2 -it $IT
# sbatch -J EVmnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 3 -it $IT
# sbatch -J EVmnAb1 execute.sh python sweep.py -m Ma1 --beta 4 -ds mn --seed 4 -it $IT

sbatch -J EVomAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds om --seed 0 -it $IT
# sbatch -J EVomAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds om --seed 1 -it $IT
# sbatch -J EVomAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds om --seed 2 -it $IT
# sbatch -J EVomAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds om --seed 3 -it $IT
# sbatch -J EVomAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds om --seed 4 -it $IT
 
sbatch -J EVqdAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds qd --seed 0 -it $IT
# sbatch -J EVqdAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds qd --seed 1 -it $IT
# sbatch -J EVqdAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds qd --seed 2 -it $IT
# sbatch -J EVqdAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds qd --seed 3 -it $IT
# sbatch -J EVqdAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds qd --seed 4 -it $IT
 
sbatch -J EVkmAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds km --seed 0 -it $IT
# sbatch -J EVkmAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds km --seed 1 -it $IT
# sbatch -J EVkmAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds km --seed 2 -it $IT
# sbatch -J EVkmAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds km --seed 3 -it $IT
# sbatch -J EVkmAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds km --seed 4 -it $IT
 
sbatch -J EVemAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds em --seed 0 -it $IT
# sbatch -J EVemAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds em --seed 1 -it $IT
# sbatch -J EVemAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds em --seed 2 -it $IT
# sbatch -J EVemAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds em --seed 3 -it $IT
# sbatch -J EVemAb1 execute.sh python sweep.py -m MTa1 --beta 4 -ds em --seed 4 -it $IT

# sbatch -J EVmnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 0 -it $IT
# sbatch -J EVmnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 1 -it $IT
# sbatch -J EVmnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 2 -it $IT
# sbatch -J EVmnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 5 -it $IT
# sbatch -J EVmnAb2 execute.sh python sweep.py -m Ma2 --beta 4 -ds mn --seed 6 -it $IT

sbatch -J EVmnAb3 execute.sh python sweep.py -m Ma3 --beta 4 -ds mn --seed 0 -it $IT
# sbatch -J EVmnAb3 execute.sh python sweep.py -m Ma3 --beta 4 -ds mn --seed 1 -it $IT
# sbatch -J EVmnAb3 execute.sh python sweep.py -m Ma3 --beta 4 -ds mn --seed 2 -it $IT
# sbatch -J EVmnAb3 execute.sh python sweep.py -m Ma3 --beta 4 -ds mn --seed 3 -it 530000
# sbatch -J EVmnAb3 execute.sh python sweep.py -m Ma3 --beta 4 -ds mn --seed 4 -it $IT

sbatch -J EvomAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds om --seed 0 -it $IT
# sbatch -J EvomAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds om --seed 1 -it $IT
# sbatch -J EvomAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds om --seed 2 -it $IT
# sbatch -J EvomAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds om --seed 3 -it $IT
# sbatch -J EvomAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds om --seed 4 -it $IT
 
sbatch -J EvqdAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds qd --seed 0 -it $IT
# sbatch -J EvqdAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds qd --seed 1 -it $IT
# sbatch -J EvqdAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds qd --seed 2 -it $IT
# sbatch -J EvqdAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds qd --seed 3 -it $IT
# sbatch -J EvqdAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds qd --seed 4 -it $IT
 
sbatch -J EvkmAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds km --seed 0 -it $IT
# sbatch -J EvkmAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds km --seed 1 -it $IT
# sbatch -J EvkmAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds km --seed 2 -it $IT
# sbatch -J EvkmAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds km --seed 3 -it $IT
# sbatch -J EvkmAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds km --seed 4 -it $IT
 
sbatch -J EvemAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds em --seed 0 -it $IT
# sbatch -J EvemAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds em --seed 1 -it $IT
# sbatch -J EvemAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds em --seed 2 -it $IT
# sbatch -J EvemAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds em --seed 3 -it $IT
# sbatch -J EvemAn3 execute.sh python sweep.py -m MTa3 --beta 4 -ds em --seed 4 -it $IT