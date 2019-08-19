#!/bin/sh

python rlalgos/sac_mult_opt.py --env_name HalfCheetah-v2 --exp_name sac_mult_opt --dir data --clear --seed 0 --remote --backend gce
python rlalgos/sac_mult_opt.py --env_name HalfCheetah-v2 --exp_name sac_mult_opt --dir data --clear --seed 11 --remote --backend gce
python rlalgos/sac_mult_opt.py --env_name HalfCheetah-v2 --exp_name sac_mult_opt --dir data --clear --seed 21 --remote --backend gce