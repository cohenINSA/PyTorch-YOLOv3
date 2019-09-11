# https://python-graph-gallery.com/122-multiple-lines-chart/
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import os

MARKERS = ['.', 'o', '-', '^', '1', 's', '*', 'p', 'P', 'x']
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
LINESTYLES = [':', '-.', '--', '-']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pandas", type=str, help="Path to the Pandas file containing the data to plot.")
    parser.add_argument("--save", type=str, help="Path and filename to save the plot.")
    parser.add_argument("--xlabel", type=str, help="Label of the x axis (default=batches)", default="batches")
    opt = parser.parse_args()
    print(opt)

    assert os.path.exists(opt.pandas)

    if opt.save is not None:
        save_path, save_name = os.path.split(opt.save)
        if os.path.exists(save_path):
            if save_name.endswith(".jpg") or save_name.endswith(".png"):
                save_plot = opt.save
            else:
                save_plot = os.path.join(os.path.splitext(opt.pandas)[0]+".png")
        else:
            save_plot = os.path.join(save_path, os.path.splitext(save_name)[0]+".png")
    else:
        save_plot = os.path.join(os.path.splitext(opt.pandas)[0]+".png")

    data_df = pd.read_csv(opt.pandas, index_col=0)

    x = data_df.index.values
    for i, c in enumerate(data_df.columns.values):
        y = data_df[c].values
        plt.plot(x, c, data=data_df, marker=MARKERS[i%len(MARKERS)], color=COLORS[i%len(COLORS)],
                 linestyle=LINESTYLES[i%len(LINESTYLES)])
    plt.legend()
    plt.show()

    plt.savefig(save_plot)
    print("Saved at %s" % save_plot)
