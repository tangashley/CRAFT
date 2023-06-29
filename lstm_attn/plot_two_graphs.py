from cProfile import label
from pdb import main
from unicodedata import name
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

path = 'checkpoint/'

# amp = np.load(path+'test_amp.npy')
w = np.load(path + 'test_weights.npy')
w2 = np.load(path + 'attn_scores.npy')
# print(w[1,:,:]);exit()
# m = np.load(path+'test_measurements.npy')

w = np.abs(w)

# print("shape of amp", amp.shape)
print("shape of weight", w.shape)
# print("shape of measurement", m.shape)

w_avg = np.mean(w, axis=0)
w_avg = w_avg / np.sum(w_avg)
print("average score ", w_avg)
print("sum average score ", np.sum(w_avg))

np.save('w_avg', w_avg)

# amp_norm = np.linalg.norm(amp, axis=2)
# print("amp norm ", amp_norm.shape)

# amp_norm_avg = np.mean(amp_norm, axis = 0)

# print("amp norm_avg ", amp_norm_avg.shape)
# print(amp_norm_avg)

# print(np.linalg.norm(amp[0,10,:]))
# print(amp[0,10,:].shape)
feature = ['Limit balance', 'gender', 'education', 'marriage', 'age',
           'pay0', 'pay2', 'pay3', 'pay4', 'pay5', 'pay6',
           'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
           'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']

dict_feature = {}

for i, f in enumerate(feature):
    dict_feature[f] = w_avg[i]

s_d = dict(sorted(dict_feature.items(), key=lambda item: item[1]))

for k in s_d:
    print(k, ":", s_d[k])


# fig, ax = plt.subplots()

# im = ax.imshow(w_avg)

# print(len(feature))

# ax.set_xticks([1])
# ax.set_yticks(np.arange(len(feature)))
# ax.set_xticklabels(["feature"])
# ax.set_yticklabels(labels = feature)
# for i in range(len(feature)):
#     for j in range(1):
#         text = ax.text(j, i, w_avg[i, j], ha="center", va="center", color="w")

# ax.set_title("credit features")
# fig.tight_layout()
# # plt.savefig("w.png")
# plt.show()

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # ax.ticklabel_format(style='sci')

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


if __name__ == "__main__":
    # plt.rcParams['text.usetex'] = True

    plt.rcParams.update({'font.size': 14})
    # matplotlib.rcParams['font.family'] = "Arial"

    # fig1, ax1 = plt.subplots(figsize=(20,3))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 6))

    ax1.title.set_text('Attention weights of Bi-LSTM')
    ax2.title.set_text('Weights of CVN')

    # w_select = w[3:8]

    im1, cbar1 = heatmap(np.transpose(w_avg * 100), ["class"], feature, ax=ax1,
                         cmap="YlGn", cbarlabel=r"weight *1e-2")
    texts1 = annotate_heatmap(im1, valfmt="{x:.3f}")

    # ax.set_title("credit features")
    fig.tight_layout()

    # w_select = w[3:8]

    im2, cbar2 = heatmap(np.transpose(w_avg * 100), ["class"], feature, ax=ax2,
                         cmap="YlGn", cbarlabel=r"weight *1e-2")
    texts2 = annotate_heatmap(im2, valfmt="{x:.3f}")

    # ax.set_title("credit features")
    fig.tight_layout()

    plt.savefig("w.png")
    # plt.show()

    # im, cbar = heatmap(np.transpose(w*100), ["class"], feature, ax=ax,
    #                 cmap="YlGn", cbarlabel= r"weight *1e-2")
    # texts = annotate_heatmap(im, valfmt="{x:.3f}")

    # ax.set_title("credit features")
    # fig.tight_layout()
    # plt.savefig("w.png")