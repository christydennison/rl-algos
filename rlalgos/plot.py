import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from base import DataLogger

def plot(args):
    data = [DataLogger(filename, args).get_data() for filename in args.filenames]
    data = pd.concat(data, ignore_index=True, sort=False)
    sns.set(style="darkgrid")
    legend = len(args.filenames) > 1
    g = sns.relplot(x="Epoch", y=args.key, kind="line", hue="ExpName", data=data, legend="brief" if legend else False)
    if legend:
        g._legend.set_draggable(state=True)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(args.title)

    if args.ylabel is not None:
        g.set_ylabels(args.ylabel)

    # replace labels
    if args.labels is not None and len(args.labels) > 0:
        for t, l in zip(g._legend.texts, [""] + args.labels): t.set_text(l)

    plt.savefig(args.savefile)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--filenames", nargs='*')
    parser.add_argument("--savefile", type=str)
    parser.add_argument("--labels", nargs='*')
    parser.add_argument("--ylabel", type=str)
    parser.add_argument("--title", type=str)
    parser.add_argument("--key", type=str, default="AverageReturn")
    parsed_args = parser.parse_args()

    plot(parsed_args)
