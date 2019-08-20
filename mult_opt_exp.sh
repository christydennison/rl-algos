#!/bin/sh

# python rlalgos/sac_mult_opt.py --env_name HalfCheetah-v2 --exp_name sac_mult_opt --dir data --clear --seed 0 --remote --backend gce
# python rlalgos/sac_mult_opt.py --env_name HalfCheetah-v2 --exp_name sac_mult_opt --dir data --clear --seed 11 --remote --backend gce
# python rlalgos/sac_mult_opt.py --env_name HalfCheetah-v2 --exp_name sac_mult_opt --dir data --clear --seed 21 --remote --backend gce

# python rlalgos/sac.py --env_name HalfCheetah-v2 --exp_name sac_single_opt --dir data --clear --seed 0 --remote --backend gce
# python rlalgos/sac.py --env_name HalfCheetah-v2 --exp_name sac_single_opt --dir data --clear --seed 11 --remote --backend gce
# python rlalgos/sac.py --env_name HalfCheetah-v2 --exp_name sac_single_opt --dir data --clear --seed 21 --remote --backend gce

# python rlalgos/sac_detach_ind.py --env_name HalfCheetah-v2 --exp_name sac_detach_ind --dir data --clear --seed 0 --remote --backend gce
# python rlalgos/sac_detach_ind.py --env_name HalfCheetah-v2 --exp_name sac_detach_ind --dir data --clear --seed 11 --remote --backend gce
# python rlalgos/sac_detach_ind.py --env_name HalfCheetah-v2 --exp_name sac_detach_ind --dir data --clear --seed 21 --remote --backend gce

# python rlalgos/sac_detach.py --env_name HalfCheetah-v2 --exp_name sac_detach --dir data --clear --seed 0 --remote --backend gce
# python rlalgos/sac_detach.py --env_name HalfCheetah-v2 --exp_name sac_detach --dir data --clear --seed 11 --remote --backend gce
# python rlalgos/sac_detach.py --env_name HalfCheetah-v2 --exp_name sac_detach --dir data --clear --seed 21 --remote --backend gce



# g rsync sac-mult-opt-0:/root/code/rl-algos/rlalgos/data/sac_mult_opt_data.csv ~/Downloads/sac_mult_opt_0.csv
# g rsync sac-mult-opt-11:/root/code/rl-algos/rlalgos/data/sac_mult_opt_data.csv ~/Downloads/sac_mult_opt_11.csv
# g rsync sac-mult-opt-21:/root/code/rl-algos/rlalgos/data/sac_mult_opt_data.csv ~/Downloads/sac_mult_opt_21.csv


# g rsync sac-detach-ind-0:/root/code/rl-algos/rlalgos/data/sac_detach_ind_data.csv ~/Downloads/sac_detach_ind_0.csv
# g rsync sac-detach-ind-11:/root/code/rl-algos/rlalgos/data/sac_detach_ind_data.csv ~/Downloads/sac_detach_ind_11.csv
# g rsync sac-detach-ind-21:/root/code/rl-algos/rlalgos/data/sac_detach_ind_data.csv ~/Downloads/sac_detach_ind_21.csv


# g rsync sac-detach-0:/root/code/rl-algos/rlalgos/data/sac_detach_data.csv ~/Downloads/sac_detach_0.csv
# g rsync sac-detach-11:/root/code/rl-algos/rlalgos/data/sac_detach_data.csv ~/Downloads/sac_detach_11.csv
# g rsync sac-detach-21:/root/code/rl-algos/rlalgos/data/sac_detach_data.csv ~/Downloads/sac_detach_21.csv


# g rsync sac-single-opt-0:/root/code/rl-algos/rlalgos/data/sac_single_opt_data.csv ~/Downloads/sac_0.csv
# g rsync sac-single-opt-11:/root/code/rl-algos/rlalgos/data/sac_single_opt_data.csv ~/Downloads/sac_11.csv
# g rsync sac-single-opt-21:/root/code/rl-algos/rlalgos/data/sac_single_opt_data.csv ~/Downloads/sac_21.csv


# python rlalgos/plot.py --filenames ~/Downloads/sac_mult_opt_0.csv ~/Downloads/sac_mult_opt_21.csv ~/Downloads/sac_mult_opt_21.csv ~/Downloads/sac_0.csv ~/Downloads/sac_11.csv ~/Downloads/sac_21.csv --savefile ~/Downloads/sac_vs_mult_opt.png --title "SAC Mult Opt Vs Single" --labels "Mult Opt" "Single Opt"

# python rlalgos/plot.py --filenames ~/Downloads/sac_detach_ind_0.csv ~/Downloads/sac_detach_ind_11.csv ~/Downloads/sac_detach_ind_21.csv ~/Downloads/sac_detach_0.csv ~/Downloads/sac_detach_11.csv ~/Downloads/sac_detach_21.csv --savefile ~/Downloads/sac_detach_vs_ind_epoch.png --title "SAC Detach Individual vs Detach All" --labels "Detach Individual" "Detach All"
# python rlalgos/plot.py --filenames ~/Downloads/sac_detach_0.csv ~/Downloads/sac_detach_11.csv ~/Downloads/sac_detach_21.csv ~/Downloads/sac_0.csv ~/Downloads/sac_11.csv ~/Downloads/sac_21.csv --savefile ~/Downloads/sac_detach_vs_no_grad_epoch.png --title "SAC Detach vs No Grad" --labels "Detach" "No Grad"


