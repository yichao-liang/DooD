#!/bin/bash  

# sbatch execute.sh python sweep.py --beta 2.5 -m emM --seed 0
# sbatch execute.sh python sweep.py --beta 2.5 -m emM --seed 1
# sbatch execute.sh python sweep.py --beta 2.5 -m emM --seed 2
# sbatch execute.sh python sweep.py --beta 2.5 -m emM --seed 3
# sbatch execute.sh python sweep.py --beta 2.5 -m emM --seed 4

# sbatch execute.sh python sweep.py --beta 2.5 -m kmM --seed 0
# sbatch execute.sh python sweep.py --beta 2.5 -m kmM --seed 1
# sbatch execute.sh python sweep.py --beta 2.5 -m kmM --seed 2
# sbatch execute.sh python sweep.py --beta 2.5 -m kmM --seed 3
# sbatch execute.sh python sweep.py --beta 2.5 -m kmM --seed 4

# sbatch execute.sh python sweep.py --beta 2.5 -m mnM --seed 0
# sbatch execute.sh python sweep.py --beta 2.5 -m mnM --seed 1
# sbatch execute.sh python sweep.py --beta 2.5 -m mnM --seed 2
# sbatch execute.sh python sweep.py --beta 2.5 -m mnM --seed 3
# sbatch execute.sh python sweep.py --beta 2.5 -m mnM --seed 4

# sbatch execute.sh python sweep.py --beta 2.5 -m omBzR --seed 0
# sbatch execute.sh python sweep.py --beta 2.5 -m omBzR --seed 1
# sbatch execute.sh python sweep.py --beta 2.5 -m omBzR --seed 2
# sbatch execute.sh python sweep.py --beta 2.5 -m omBzR --seed 3
# sbatch execute.sh python sweep.py --beta 2.5 -m omBzR --seed 4

sbatch execute.sh python sweep.py --beta 2.5 -m omBRT --seed 1
sbatch execute.sh python sweep.py --beta 2.5 -m omBRT --seed 0
sbatch execute.sh python sweep.py --beta 2.5 -m omBRT --seed 2

# sbatch execute.sh python sweep.py --beta 2.5 -m omIm --seed 0
# sbatch execute.sh python sweep.py --beta 2.5 -m omIm --seed 1
# sbatch execute.sh python sweep.py --beta 2.5 -m omIm --seed 2
# sbatch execute.sh python sweep.py --beta 2.5 -m omIm --seed 3
# sbatch execute.sh python sweep.py --beta 2.5 -m omIm --seed 4

# sbatch execute.sh python sweep.py --beta 4 -m mnIm --seed 0 
# sbatch execute.sh python sweep.py --beta 4 -m mnIm --seed 1 
# sbatch execute.sh python sweep.py --beta 4 -m mnIm10 --seed 2 
# sbatch execute.sh python sweep.py --beta 4 -m mnIm10 --seed 3 
# sbatch execute.sh python sweep.py --beta 4 -m mnIm10 --seed 4

# sbatch execute.sh python sweep.py --beta 4 -m mnImT --seed 0 
# sbatch execute.sh python sweep.py --beta 4 -m mnImT --seed 1 
# sbatch execute.sh python sweep.py --beta 4 -m mnImT --seed 2 
# sbatch execute.sh python sweep.py --beta 4 -m mnImT --seed 3 
# sbatch execute.sh python sweep.py --beta 4 -m mnImT --seed 4

# sbatch execute.sh python sweep.py --beta 4 -m mnBzR --seed 0
# sbatch execute.sh python sweep.py --beta 4 -m mnBzR --seed 1
# sbatch execute.sh python sweep.py --beta 4 -m mnBzR --seed 2 
# sbatch execute.sh python sweep.py --beta 4 -m mnBzR --seed 3 
# sbatch execute.sh python sweep.py --beta 4 -m mnBzR --seed 4 