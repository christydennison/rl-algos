import sys, subprocess

names = """ppo-pd-pointgrid1-std
ppo-pd-pointgrid2
ppo-sep-std-1-3e6-1seed
ppo-sep-std-1-3e6-1seed-nof
ppo-sep-std-1-3e6-1seed-sa
ppo-sep-std-1-3e6-2
ppo-sep-std-1-3e6-3seed-sa
ppo-sep-std-4-3e6
ppo-sep-std-4-3e6-1seed
ppo-sep-std-4-3e6-3seed-nof
ppo-sep-std-4-3e6-3seed-sa"""


names_tuples = [(name, '_'.join(name.split('-'))) for name in names.split('\n')]

for hyphen_name, under_name in names_tuples:
    cmd = f"source activate safeexp; rcall-gce rsync {hyphen_name}:/root/code/rl-algos/rlalgos/data/{under_name}_data.csv /Users/christy/Downloads/{under_name}_data.csv"
    print(under_name)
    subprocess.check_call(cmd, shell=True)

