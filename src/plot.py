"""Create plots

Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from utils import make_dir
from itertools import product

sns.set(
    context="paper",
    style="whitegrid",
    palette="RdBu",
    color_codes=True
)


def plot_motiv_weight(data):
    data = data.reshape((1,-1))
    print(data)

    ax = sns.heatmap(
        data=data,
        vmin=0,
        vmax=1,
        linewidths=0.01,
        cmap="Blues",
        cbar=False,
        annot=True,
        annot_kws={
            "fontsize": 20
        },
        square=True,
        xticklabels=['SC', 'NI', 'AT'],
        yticklabels=False,
    )
    ax.tick_params(labelsize=20)

    plt.show()


def plot_geo_weight(user, poi, m,n):
    user = user.reshape((m,n))
    poi = poi.reshape((m,n))
    fig, axs = plt.subplots(ncols=2)
    ax1 = sns.heatmap(
        data=user,
        vmin=0,
        vmax=1,
        linewidths=0.01,
        cmap="Blues",
        cbar=True,
        cbar_kws={"shrink": .5},
        annot=False,
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=axs[0]
    )
    ax1.set(title="User Geographical Preference")

    ax2 = sns.heatmap(
        data=poi,
        vmin=0,
        vmax=1,
        linewidths=0.01,
        cmap="Greens",
        cbar=True,
        cbar_kws={"shrink": .5},
        annot=False,
        square=True,
        xticklabels=False,
        # xticklabels=['SC', 'NI', 'AT'],
        yticklabels=False,
        ax=axs[1]
    )
    ax2.set(title="POI Geolocation Influence")

    plt.show()


def plot_attr_weights(data):
    data = data.reshape((1, -1))
    print(data)

    ax = sns.heatmap(
        data=data,
        vmin=0, vmax=1,
        linewidths=0.01,
        cmap="Blues",
        cbar=True,
        annot=False,
        square=True
    )
    ax.tick_params(labelsize=20)

    plt.show()


def plot_friend_weights(data):
    # TODO:
    pass


def curve_performance(metric, city):
    baseline_perf = "/Users/zyli/Research/interptable-recsys-friend-network/baseline_perf/"
    assert metric in ["prec", "recall", "map"]
    assert city in ["lv", "tor", "phx"]
    title = {
        "prec": "Precision",
        "recall": "Recall",
        "map": "MAP"
    }

    cities = {
        "lv": "Las Vegas",
        "tor": "Toronto",
        "phx": "Phoenix"
    }

    fname = baseline_perf + metric + ".csv"
    df = pd.read_csv(fname)

    print(df.columns)

    df = df.drop([3]).reset_index(drop=True)


    models = ['USG', 'GeoSoCa', "MF", "WRMF", "GeoMF", "LORE", "GEARP"]
    # tor_df['model'] = models

    if city == "tor":
        sub_df = df.iloc[1:, 1: 11]
    elif city == "phx":
        sub_df = df.iloc[1:, 11: 21]
    else:
        sub_df = df.iloc[1:, 21: 31]

    sub_df.columns = [str(x) for x in range(10, 101, 10)]

    # ax = sns.lineplot(data=tor_df.iloc[:, 1:5])
    x_axis = list(range(10, 101, 10))
    usg, = plt.plot(x_axis, sub_df.iloc[0, :], "--o", color="darkviolet")
    geosoca, = plt.plot(x_axis, sub_df.iloc[1, :], "--go")
    mf, = plt.plot(x_axis, sub_df.iloc[2, :], "--ro")
    wrmf, = plt.plot(x_axis, sub_df.iloc[3, :], "--co")
    geomf, = plt.plot(x_axis, sub_df.iloc[4, :], "--o", color="chocolate")
    lore, = plt.plot(x_axis, sub_df.iloc[5, :], "--o", color="orange")
    geapr, = plt.plot(x_axis, sub_df.iloc[6, :], "--o", color="dodgerblue")


    usg.set_label("USG")
    geosoca.set_label("GeoSoCa")
    mf.set_label("MF")
    wrmf.set_label("WRMF")
    geomf.set_label("GeoMF")
    lore.set_label("LORE")
    geapr.set_label("GEAPR")

    if metric == "prec":
        loc = "upper right"
    else:
        loc = "upper left"


    plt.xticks(x_axis, fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("k", fontsize=20)
    plt.ylabel(title[metric], fontsize=20)
    plt.legend(fontsize=12, loc=loc)

    # plt.show()
    make_dir("/Users/zyli/Research/interptable-recsys-friend-network/baseline_perf/image/")
    plt.savefig("/Users/zyli/Research/interptable-recsys-friend-network/baseline_perf/image/{}_{}.pdf".format(city, metric),
                format="pdf")
    plt.clf()


def curve_ablation(metric, city):
    baseline_perf = "/Users/zyli/Research/interptable-recsys-friend-network/baseline_perf/"
    assert metric in ["prec", "recall", "map"]
    assert city in ["lv", "tor", "phx"]
    title = {
        "prec": "Precision",
        "recall": "Recall",
        "map": "MAP"
    }

    fname = baseline_perf + metric + "_abla.csv"
    df = pd.read_csv(fname)

    print(df.columns)

    if city == "tor":
        sub_df = df.iloc[1:, 1: 11]
    elif city == "phx":
        sub_df = df.iloc[1:, 11: 21]
    else:
        sub_df = df.iloc[1:, 21: 31]

    sub_df.columns = [str(x) for x in range(10, 101, 10)]

    # ax = sns.lineplot(data=tor_df.iloc[:, 1:5])
    x_axis = list(range(10, 101, 10))
    sc, = plt.plot(x_axis, sub_df.iloc[0, :], "--o", color="darkviolet")
    ni, = plt.plot(x_axis, sub_df.iloc[1, :], "--go")
    at, = plt.plot(x_axis, sub_df.iloc[2, :], "--ro")
    geo, = plt.plot(x_axis, sub_df.iloc[3, :], "--co")
    pa, = plt.plot(x_axis, sub_df.iloc[4, :], "--o", color="chocolate")
    geapr, = plt.plot(x_axis, sub_df.iloc[5, :], "--o", color="dodgerblue")


    sc.set_label("GEAPR-SC")
    ni.set_label("GEAPR-NI")
    at.set_label("GEAPR-AT")
    geo.set_label("GEAPR-GEO")
    pa.set_label("GEAPR-PA")
    geapr.set_label("GEAPR")

    if metric == "map":
        loc = "lower right"
    elif metric == "prec":
        loc = "upper right"
    else:
        loc = "upper left"


    plt.xticks(x_axis, fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("k", fontsize=20)
    plt.ylabel(title[metric], fontsize=20)
    plt.legend(fontsize=12, loc=loc)

    # plt.show()
    make_dir("/Users/zyli/Research/interptable-recsys-friend-network/baseline_perf/abla/")
    plt.savefig("/Users/zyli/Research/interptable-recsys-friend-network/baseline_perf/abla/{}_{}.pdf".format(city, metric),
                format="pdf")
    plt.clf()


if __name__ == "__main__":

    # Plotting motivation weights
    data = np.random.uniform(0, 10, [3])
    print(data.shape)
    # plot_motiv_weight(data)

    # Plotting geolocation weight
    user = np.random.rand(900)
    poi = np.random.rand(900)
    # plot_geo_weight(user, poi, 30, 30)

    # Plotting performance curves
    metrics = ["map", "prec", "recall"]
    # cities = ["lv", "tor", "phx"]
    cities = ["phx", "tor"]
    all_plots = product(metrics, cities)
    for m, c in all_plots:
        print(m, c)
        curve_performance(m, c)
        curve_ablation(m, c)


"""
Possible values are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r
"""