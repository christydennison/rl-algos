#!/bin/sh

# python rlalgos/sac.py --env_name HalfCheetah-v2 --exp_name sac_3e6_steps_1000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 1000
# python rlalgos/sac.py --env_name HalfCheetah-v2 --exp_name sac_3e6_steps_2000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 2000
# python rlalgos/sac.py --env_name HalfCheetah-v2 --exp_name sac_3e6_steps_3000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 3000
# python rlalgos/sac.py --env_name HalfCheetah-v2 --exp_name sac_3e6_steps_4000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 4000
# python rlalgos/sac.py --env_name HalfCheetah-v2 --exp_name sac_3e6_steps_5000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 5000
# python rlalgos/sac.py --env_name HalfCheetah-v2 --exp_name sac_3e6_steps_6000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 6000


# python rlalgos/sac_detach.py --env_name HalfCheetah-v2 --exp_name sac_detach_3e6_steps_1000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 1000
# python rlalgos/sac_detach.py --env_name HalfCheetah-v2 --exp_name sac_detach_3e6_steps_2000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 2000
# python rlalgos/sac_detach.py --env_name HalfCheetah-v2 --exp_name sac_detach_3e6_steps_3000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 3000
# python rlalgos/sac_detach.py --env_name HalfCheetah-v2 --exp_name sac_detach_3e6_steps_4000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 4000
# python rlalgos/sac_detach.py --env_name HalfCheetah-v2 --exp_name sac_detach_3e6_steps_5000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 5000
# python rlalgos/sac_detach.py --env_name HalfCheetah-v2 --exp_name sac_detach_3e6_steps_6000 --dir data --clear --epochs 600 --backend gce --remote --ncpu 8 --seeds 6000


rg rsync sac-3e6-steps-1000:/root/code/rl-algos/rlalgos/data/sac_3e6_steps_1000_data.csv ~/Downloads/sac_3e6_steps_1000_data.csv
rg rsync sac-3e6-steps-2000:/root/code/rl-algos/rlalgos/data/sac_3e6_steps_2000_data.csv ~/Downloads/sac_3e6_steps_2000_data.csv
rg rsync sac-3e6-steps-3000:/root/code/rl-algos/rlalgos/data/sac_3e6_steps_3000_data.csv ~/Downloads/sac_3e6_steps_3000_data.csv
rg rsync sac-3e6-steps-4000:/root/code/rl-algos/rlalgos/data/sac_3e6_steps_4000_data.csv ~/Downloads/sac_3e6_steps_4000_data.csv
rg rsync sac-3e6-steps-5000:/root/code/rl-algos/rlalgos/data/sac_3e6_steps_5000_data.csv ~/Downloads/sac_3e6_steps_5000_data.csv
rg rsync sac-3e6-steps-6000:/root/code/rl-algos/rlalgos/data/sac_3e6_steps_6000_data.csv ~/Downloads/sac_3e6_steps_6000_data.csv

rg rsync sac-detach-3e6-steps-1000:/root/code/rl-algos/rlalgos/data/sac_detach_3e6_steps_1000_data.csv ~/Downloads/sac_detach_3e6_steps_1000_data.csv
rg rsync sac-detach-3e6-steps-2000:/root/code/rl-algos/rlalgos/data/sac_detach_3e6_steps_2000_data.csv ~/Downloads/sac_detach_3e6_steps_2000_data.csv
rg rsync sac-detach-3e6-steps-3000:/root/code/rl-algos/rlalgos/data/sac_detach_3e6_steps_3000_data.csv ~/Downloads/sac_detach_3e6_steps_3000_data.csv
rg rsync sac-detach-3e6-steps-4000:/root/code/rl-algos/rlalgos/data/sac_detach_3e6_steps_4000_data.csv ~/Downloads/sac_detach_3e6_steps_4000_data.csv
rg rsync sac-detach-3e6-steps-5000:/root/code/rl-algos/rlalgos/data/sac_detach_3e6_steps_5000_data.csv ~/Downloads/sac_detach_3e6_steps_5000_data.csv
rg rsync sac-detach-3e6-steps-6000:/root/code/rl-algos/rlalgos/data/sac_detach_3e6_steps_6000_data.csv ~/Downloads/sac_detach_3e6_steps_6000_data.csv


# python rlalgos/plot.py --filenames ~/Downloads/sac_detach_0.csv ~/Downloads/sac_detach_11.csv ~/Downloads/sac_detach_21.csv ~/Downloads/sac_0.csv ~/Downloads/sac_11.csv ~/Downloads/sac_21.csv --savefile ~/Downloads/sac_detach_vs_no_grad_epoch.png --title "SAC Detach vs No Grad" --labels "Detach" "No Grad"


