# Adapted from https://github.com/brentp/bio-playground/blob/master/plots/manhattan-plot.py

"""
MIT License

Copyright (c) 2009-2011 Brent Pedersen, Haibao Tang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import matplotlib, matplotlib.pyplot as plt, numpy as np, pandas as pd, seaborn, warnings
from itertools import groupby, cycle
from matplotlib.ticker import FormatStrFormatter
from operator import itemgetter

# noinspection PyUnresolvedReferences
matplotlib.style.use('default')
warnings.filterwarnings('ignore', category=UserWarning)  # ignore Matplotlib missing font warnings

def manhattan(chroms, bps, ps, save_to, chrom_colors='bk', gene_colors=None,
              title=None, lines=False, ymax=None, log_scale=True, despine=True, labels=None,
              num_tests=None, start_on_right=False, highlight_adjacent_hits=False,
              width_scale=1.0, height_scale=1.0):
    if isinstance(chroms, pd.Series): chroms = chroms.values
    if isinstance(bps, pd.Series): bps = bps.values
    if isinstance(ps, pd.Series): ps = ps.values
    assert isinstance(chroms, np.ndarray)
    assert isinstance(bps, np.ndarray)
    # noinspection PyUnresolvedReferences
    assert issubclass(bps.dtype.type, np.integer)
    assert isinstance(ps, np.ndarray)
    assert len(chroms) == len(bps) == len(ps)
    if labels is not None:
        assert len(labels) == len(ps)
    xs, ys, cs = [], [], []
    xs_by_chr = {}
    last_x = 0
    chrom_to_order = {f'chr{chrom}': chrom for chrom in range(1, 23)}
    chrom_to_order.update({'chrX': 23, 'chrY': 24})
    key_func = lambda triple: (chrom_to_order[triple[0]], triple[1])
    data = sorted(zip(chroms, bps, ps), key=key_func)
    for (seqid, rlist), color in zip(groupby(data, key=itemgetter(0)), cycle(chrom_colors)):
        rlist = list(rlist)
        region_xs = [last_x + r[1] for r in rlist]
        xs.extend(region_xs)
        ys.extend([r[2] for r in rlist])
        cs.extend([color] * len(rlist))
        xs_by_chr[seqid] = (region_xs[0] + region_xs[-1]) / 2
        # keep track so that chrs don't overlap.
        last_x = xs[-1]
    xs_by_chr = [(k, xs_by_chr[k]) for k in sorted(xs_by_chr)]
    xs = np.array(xs)
    if len(np.unique(chroms)) == 1: xs = xs * 1e-6
    ys = np.array(ys) if not log_scale else \
        -np.log10(np.maximum(ys, np.nextafter(0, 1)))  # clip 0 to +eps
    plt.close()
    ax = plt.gca()
    if gene_colors is not None:
        # Override cs with gene_colors
        cs = gene_colors
    if title is not None:
        plt.title(title)
    ax.set_ylabel('$-log_{10}(p)$')
    if lines:
        ax.vlines(xs, 0, ys, colors=cs, alpha=0.5)
    else:
        ax.scatter(xs, ys, s=2, c=cs, alpha=0.8, edgecolors='none')
    # plot 0.05 line after multiple testing.
    if num_tests is None: num_tests = len(data)
    log_significance_threshold = -np.log10(0.05 / num_tests)
    ax.axhline(y=log_significance_threshold, color='0.5', linewidth=1)
    if highlight_adjacent_hits:
        hits = ys > log_significance_threshold
        # hits_in_clusters is the same as hits but with singletons removed
        hits_in_clusters = hits.copy()
        for index in range(len(hits)):
            is_hit = hits[index]
            if not is_hit: continue
            has_hit_to_left_on_same_chrom = \
                index != 0 and hits[index - 1] and chroms[index] == chroms[index - 1]
            has_hit_to_right_on_same_chrom = \
                index != len(hits) - 1 and hits[index + 1] and chroms[index] == chroms[index + 1]
            is_singleton = not (has_hit_to_left_on_same_chrom or has_hit_to_right_on_same_chrom)
            if is_singleton:
                hits_in_clusters[index] = False
        hit_cluster_indices = np.flatnonzero(hits_in_clusters)
        hit_clusters = np.split(hit_cluster_indices,
                                np.flatnonzero(np.diff(hit_cluster_indices) != 1) + 1)
        # print(f'[{save_to} Manhattan]: There are {len(hit_clusters)} hit clusters\n')
        for hit_cluster in hit_clusters:
            start_index, end_index = hit_cluster[0], hit_cluster[-1]
            padding_w = 10000000
            padding_h = 2
            rect_left, rect_right = xs[start_index] - padding_w, xs[end_index] + padding_w
            # noinspection PyUnresolvedReferences
            plt.gca().add_patch(
                matplotlib.patches.Rectangle(
                    (rect_left, 0),  # (x,y)
                    rect_right - rect_left,  # width
                    ys[start_index : end_index + 1].max() + padding_h,  # height
                    edgecolor='r', lw=1, fill=None))
    if labels is not None:
        labels = np.array([label for chrom, bp, p, label in
                           sorted(zip(chroms, bps, ps, labels), key=key_func)])
        # Annotate all significant hits with their label
        for chrom in np.unique(chroms):  # hack: assume all hit blocks on diff. chromosomes
            significant = (chroms == chrom) & (ys > log_significance_threshold)
            right = start_on_right
            for index, (label, x, y) in enumerate(
                    sorted(zip(labels[significant], xs[significant], ys[significant]),
                           key=lambda x: x[1])):
                actual_right = True if index == 0 else False \
                    if index == len(labels[significant]) - 1\
                    else right
                plt.annotate(label, xy=(x, y), xytext=(
                    3 if actual_right else -3, 0 if actual_right else -10),
                             horizontalalignment='left' if actual_right else 'right',
                             textcoords='offset points',
                             size='medium' if len(labels[significant]) > 5 else 'large')
                right = not right
    plt.ylim(ymin=0)
    if log_scale:
        plt.yscale('symlog')
        plt.gca().get_yaxis().set_major_formatter(FormatStrFormatter('%.0f'))  # no sci notation
        if ys.max() < 10:
            plt.ylim(ymax=11)  # make sure at least the 1 and 10 marks are displayed
    if ymax is not None: plt.ylim(ymax=ymax)
    if width_scale != 1.0 or height_scale != 1.0:
        w, h = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches(width_scale * w, height_scale * h)
    if len(np.unique(chroms)) > 1:
        plt.xlim(0, xs[-1])
        plt.xticks([c[1] for c in xs_by_chr],
                   [c[0][3:] if int(c[0][3:]) < 10 or int(c[0][3:]) % 2 == 0 else ''
                    for c in xs_by_chr],  # only do even-numbered double-digit chromosomes
                   size='medium')
    else:
        plt.xlabel(f'{chroms[0]} (MB)')
    if despine:
        seaborn.despine()
    plt.savefig(save_to, dpi=600)
    plt.close()