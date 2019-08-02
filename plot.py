import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from base import DataLogger

def plot(filenames):
    data = [DataLogger(filename).get_data() for filename in filenames]
    data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid")
    sns.relplot(x="Steps", y="AverageReturn", kind="line", hue="ExpName", data=data)
    plt.legend(loc='best').set_draggable(state=True)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", nargs='*')
    args = parser.parse_args()

    plot(args.logdir)
