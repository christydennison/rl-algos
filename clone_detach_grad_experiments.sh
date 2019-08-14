#!/bin/sh

python rlalgos/sac_no_grad_no_clone.py --env_name HalfCheetah-v2 --exp_name sac_no_grad_no_clone_0 --dir data --clear --seed 0 --remote --backend gce
python rlalgos/sac_no_grad_no_clone.py --env_name HalfCheetah-v2 --exp_name sac_no_grad_no_clone_11 --dir data --clear --seed 11 --remote --backend gce
python rlalgos/sac_no_grad_no_clone.py --env_name HalfCheetah-v2 --exp_name sac_no_grad_no_clone_21 --dir data --clear --seed 21 --remote --backend gce

python rlalgos/sac_no_grad_clone.py --env_name HalfCheetah-v2 --exp_name sac_no_grad_clone_0 --dir data --clear --seed 0 --remote --backend gce
python rlalgos/sac_no_grad_clone.py --env_name HalfCheetah-v2 --exp_name sac_no_grad_clone_11 --dir data --clear --seed 11 --remote --backend gce
python rlalgos/sac_no_grad_clone.py --env_name HalfCheetah-v2 --exp_name sac_no_grad_clone_21 --dir data --clear --seed 21 --remote --backend gce

python rlalgos/sac_detach_no_clone.py --env_name HalfCheetah-v2 --exp_name sac_detach_no_clone_0 --dir data --clear --seed 0 --remote --backend gce
python rlalgos/sac_detach_no_clone.py --env_name HalfCheetah-v2 --exp_name sac_detach_no_clone_11 --dir data --clear --seed 11 --remote --backend gce
python rlalgos/sac_detach_no_clone.py --env_name HalfCheetah-v2 --exp_name sac_detach_no_clone_21 --dir data --clear --seed 21 --remote --backend gce

python rlalgos/sac_detach_clone.py --env_name HalfCheetah-v2 --exp_name sac_detach_clone_0 --dir data --clear --seed 0 --remote --backend gce
python rlalgos/sac_detach_clone.py --env_name HalfCheetah-v2 --exp_name sac_detach_clone_11 --dir data --clear --seed 11 --remote --backend gce
python rlalgos/sac_detach_clone.py --env_name HalfCheetah-v2 --exp_name sac_detach_clone_21 --dir data --clear --seed 21 --remote --backend gce