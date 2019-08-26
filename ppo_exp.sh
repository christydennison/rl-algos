#!/bin/sh

# python rlalgos/ppo.py --env_name Hopper-v2 --exp_name ppo --dir data --clear --seed 0 --remote --backend gce
# python rlalgos/ppo.py --env_name Hopper-v2 --exp_name ppo --dir data --clear --seed 11 --remote --backend gce
# python rlalgos/ppo.py --env_name Hopper-v2 --exp_name ppo --dir data --clear --seed 21 --remote --backend gce

# python rlalgos/ppo_sep_std.py --env_name Hopper-v2 --exp_name ppo_sep_std --dir data --clear --seed 0 --remote --backend gce
# python rlalgos/ppo_sep_std.py --env_name Hopper-v2 --exp_name ppo_sep_std --dir data --clear --seed 11 --remote --backend gce
# python rlalgos/ppo_sep_std.py --env_name Hopper-v2 --exp_name ppo_sep_std --dir data --clear --seed 21 --remote --backend gce


# g rsync ppo-0:/root/code/rl-algos/rlalgos/data/ppo_data.csv ~/Downloads/ppo_0.csv
# g rsync ppo-11:/root/code/rl-algos/rlalgos/data/ppo_data.csv ~/Downloads/ppo_11.csv
# g rsync ppo-21:/root/code/rl-algos/rlalgos/data/ppo_data.csv ~/Downloads/ppo_21.csv

# g rsync ppo-sep-std-0:/root/code/rl-algos/rlalgos/data/ppo_sep_std_data.csv ~/Downloads/ppo_sep_std_0.csv
# g rsync ppo-sep-std-11:/root/code/rl-algos/rlalgos/data/ppo_sep_std_data.csv ~/Downloads/ppo_sep_std_11.csv
# g rsync ppo-sep-std-21:/root/code/rl-algos/rlalgos/data/ppo_sep_std_data.csv ~/Downloads/ppo_sep_std_21.csv


python rlalgos/plot.py --filenames ~/Downloads/ppo_sep_std_0.csv ~/Downloads/ppo_sep_std_11.csv ~/Downloads/ppo_sep_std_21.csv ~/Downloads/ppo_0.csv ~/Downloads/ppo_11.csv ~/Downloads/ppo_21.csv --savefile ~/Downloads/ppo_stddev_comp.png --title "PPO Trained Variable vs Network Output STDDEV" --labels "Trained Variable" "Network Output"



python rlalgos/plot.py --filenames /Users/christy/projects/spinningup/data/spinup_ppo_3/spinup_ppo_3_s0/progress.txt /Users/christy/projects/spinningup/data/spinup_ppo_3/spinup_ppo_3_s11/progress.txt /Users/christy/projects/spinningup/data/spinup_ppo_3/spinup_ppo_3_s21/progress.txt ~/Downloads/ppo_sep_std_0.csv ~/Downloads/ppo_sep_std_11.csv ~/Downloads/ppo_sep_std_21.csv --savefile ~/Downloads/ppo_vs_spinup.png --title "PPO Spinup vs My Impl" --labels "Spinup" "My Impl"