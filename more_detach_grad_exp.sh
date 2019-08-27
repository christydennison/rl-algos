#!/bin/sh

# python rlalgos/sac.py --env_name HalfCheetah-v2 --exp_name sac_3e6_steps_1000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 1000
# python rlalgos/sac.py --env_name HalfCheetah-v2 --exp_name sac_3e6_steps_2000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 2000
# python rlalgos/sac.py --env_name HalfCheetah-v2 --exp_name sac_3e6_steps_3000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 3000
# python rlalgos/sac.py --env_name HalfCheetah-v2 --exp_name sac_3e6_steps_4000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 4000
# python rlalgos/sac.py --env_name HalfCheetah-v2 --exp_name sac_3e6_steps_5000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 5000
# python rlalgos/sac.py --env_name HalfCheetah-v2 --exp_name sac_3e6_steps_6000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 6000


python rlalgos/sac_detach.py --env_name HalfCheetah-v2 --exp_name sac_detach_3e6_steps_1000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 1000
python rlalgos/sac_detach.py --env_name HalfCheetah-v2 --exp_name sac_detach_3e6_steps_2000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 2000
python rlalgos/sac_detach.py --env_name HalfCheetah-v2 --exp_name sac_detach_3e6_steps_3000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 3000
python rlalgos/sac_detach.py --env_name HalfCheetah-v2 --exp_name sac_detach_3e6_steps_4000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 4000
python rlalgos/sac_detach.py --env_name HalfCheetah-v2 --exp_name sac_detach_3e6_steps_5000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 5000
python rlalgos/sac_detach.py --env_name HalfCheetah-v2 --exp_name sac_detach_3e6_steps_6000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 6000


# g rsync sac-detach-0:/root/code/rl-algos/rlalgos/data/sac_detach_data.csv ~/Downloads/sac_detach_0.csv
# g rsync sac-detach-11:/root/code/rl-algos/rlalgos/data/sac_detach_data.csv ~/Downloads/sac_detach_11.csv
# g rsync sac-detach-21:/root/code/rl-algos/rlalgos/data/sac_detach_data.csv ~/Downloads/sac_detach_21.csv


# g rsync sac-single-opt-0:/root/code/rl-algos/rlalgos/data/sac_no_grad_data.csv ~/Downloads/sac_0.csv
# g rsync sac-single-opt-11:/root/code/rl-algos/rlalgos/data/sac_no_grad_data.csv ~/Downloads/sac_11.csv
# g rsync sac-single-opt-21:/root/code/rl-algos/rlalgos/data/sac_no_grad_data.csv ~/Downloads/sac_21.csv

# rg rsync sac-detach-3-31:/root/code/rl-algos/rlalgos/data/sac_detach_3_data.csv ~/Downloads/sac_detach_31.csv
# rg rsync sac-detach-3-41:/root/code/rl-algos/rlalgos/data/sac_detach_3_data.csv ~/Downloads/sac_detach_41.csv
# rg rsync sac-detach-3-51:/root/code/rl-algos/rlalgos/data/sac_detach_3_data.csv ~/Downloads/sac_detach_51.csv


# rg rsync sac-no-grad-3-31:/root/code/rl-algos/rlalgos/data/sac_no_grad_3_data.csv ~/Downloads/sac_31.csv
# rg rsync sac-no-grad-3-41:/root/code/rl-algos/rlalgos/data/sac_no_grad_3_data.csv ~/Downloads/sac_41.csv
# rg rsync sac-no-grad-3-51:/root/code/rl-algos/rlalgos/data/sac_no_grad_3_data.csv ~/Downloads/sac_51.csv


# python rlalgos/plot.py --filenames ~/Downloads/sac_detach_0.csv ~/Downloads/sac_detach_11.csv ~/Downloads/sac_detach_21.csv ~/Downloads/sac_0.csv ~/Downloads/sac_11.csv ~/Downloads/sac_21.csv --savefile ~/Downloads/sac_detach_vs_no_grad_epoch.png --title "SAC Detach vs No Grad" --labels "Detach" "No Grad"


