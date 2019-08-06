import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from base import DataLogger

def plot(args):
    data = [DataLogger(filename).get_data() for filename in args.filenames]
    data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid")
    sns.relplot(x="Steps", y=args.key, kind="line", hue="ExpName", data=data)
    plt.legend(loc='best').set_draggable(state=True)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--filenames", nargs='*')
    parser.add_argument("--key", type=str, default="AverageReturn")
    args = parser.parse_args()

    plot(args)
