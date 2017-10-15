import matplotlib, numpy as np, pandas as pd, parse
matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from dotdict import dotdict
from itertools import islice
from matplotlib.patches import Patch

for text_type in 'xtick', 'ytick', 'axes':
    plt.rc(text_type, labelsize='x-large')

def parse_line(parser, line):
    return dotdict(parser.parse(line).__dict__['named'])

def grouped_barplot(groups, save_to, ylabel='', colors=None, bar_labels=None,
                    width_scale=1.0, height_scale=1.0,
                    dummy_width=1, group_labels=None, legend_labels=None):
    # groups is a list of lists of values in the group
    # colors has the same structure as groups and gives color names for each bar
    # bar_labels also has the same structure as groups and gives labels for each bar
    # legend_labels is a dict mapping colors to labels
    # group_labels is a len(groups)-sized list with labels for each group
    # Hack: join groups with a dummy-width-sized 0-height bar
    assert not (bar_labels is not None and
                group_labels is not None)  # can't plot both types of labels
    y = np.concatenate([elem for index, group in enumerate(groups)
                        for elem in ((group,) if index == len(colors) - 1 else
                                     (group, ('0',) * dummy_width))])
    x = np.arange(len(y))
    if colors is None:
        plt.bar(x, y, width=1)
    else:
        c = np.concatenate([elem for index, group in enumerate(colors)
                            for elem in ((group,) if index == len(colors) - 1 else
                                         (group, ('k',) * dummy_width))])
        plt.bar(x, y, width=1, color=c)
    group_lengths = np.array([len(group) for group in groups])
    if group_labels is not None:
        cumsum = np.cumsum(group_lengths + dummy_width) - dummy_width + 1
        # +dummy_width for dummy width; -dummy_width because first bar doesn't have dummy width,
        # + 1 because we start at 1
        left_edges = np.concatenate(((0,), cumsum[:-1]))
        # noinspection PyTypeChecker
        right_edges = np.concatenate((cumsum[:-1], (len(x),)))
        midpoint = (left_edges + right_edges) / 2
        plt.xticks(midpoint, group_labels)
    if bar_labels is not None:
        bar_labels = np.concatenate([elem for index, group in enumerate(bar_labels)
                        for elem in ((group,) if index == len(colors) - 1 else
                                     (group, ('',) * dummy_width))])
        plt.xticks(x[bar_labels != ''], bar_labels[bar_labels != ''], rotation=90)
    if width_scale != 1.0 or height_scale != 1.0:
        w, h = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches(width_scale * w, height_scale * h)
    if legend_labels is not None:
        plt.legend(handles=[Patch(color=color, label=label)
                            for color, label in legend_labels.items()], loc='best')
    plt.ylabel(ylabel)
    plt.gcf().set_tight_layout(True)
    plt.savefig(save_to, dpi=600)
    plt.close()

line_format = '{gene} ({chrom}:{start}-{end}): p < {p:.0g}, Bonf q < {q:.0g}'
parser = parse.compile(line_format)
for correct_tissue, incorrect_tissue, trait in \
        (('LIV', 'Blood', 'LDL'), ('Blood', 'LIV', 'Crohns')):
    causal_genes = {'SORT1', 'IRF2BP2', 'PPARG', 'LPA', 'TNKS',
                    'FADS1', 'FADS2', 'FADS3', 'ALDH2', 'KPNB1', 'LPIN3'} \
        if trait == 'LDL' else {'IRF1', 'SLC22A4', 'SLC22A5', 'CARD9', 'NOD2', 'ACDY7', 'STAT3'}
    correct_tissue_hits = defaultdict(list)
    results_dir = f'fusion_twas/results/{correct_tissue}_{trait}'
    TWAS_results_file = f'{results_dir}/TWAS_results.txt'
    block_number = 0
    for line in islice(open(TWAS_results_file), 1, None):
        line = line.rstrip()
        if not line:
            block_number += 1
            continue
        parsed_line = parse_line(parser, line)
        for key in 'start', 'end':
            parsed_line[key] = int(parsed_line[key].replace(',', ''))
        for key, value in parsed_line.items():
            correct_tissue_hits[key].append(value)
        correct_tissue_hits['block'].append(block_number)
    correct_tissue_hits = pd.DataFrame(correct_tissue_hits, columns=correct_tissue_hits)
    correct_tissue_hits.set_index('gene', inplace=True)
    # Remove blocks that do not contain one of the causal genes
    blocks_with_causal_gene = np.unique(correct_tissue_hits['block'][
        correct_tissue_hits.index.isin(causal_genes)].values)
    correct_tissue_hits = correct_tissue_hits[
        correct_tissue_hits['block'].isin(blocks_with_causal_gene)]
    # Sort by log10(p)
    correct_tissue_hits.sort_values('p', inplace=True)
    # Make box plot for correct tissue
    ordered_causal_genes = correct_tissue_hits.index[
        correct_tissue_hits.index.isin(causal_genes)].values
    genes = [correct_tissue_hits[correct_tissue_hits['block'] == block].index
             for block, gene in zip(blocks_with_causal_gene, ordered_causal_genes)]
    groups = [-np.log10(correct_tissue_hits[correct_tissue_hits['block'] == block]['p'])
              for block, gene in zip(blocks_with_causal_gene, ordered_causal_genes)]
    colors = [['r' if gene in causal_genes else 'k' for gene in gene_group]
              for gene_group in genes]
    bar_labels = [[gene if gene in causal_genes else '' for gene in gene_group]
                  for gene_group in genes]
    grouped_barplot(groups,
                    save_to=f'{results_dir}/multi_gene_blocks',
                    ylabel='$-log_{10}(p)$',
                    colors=colors,
                    bar_labels=bar_labels,
                    width_scale=1.25,
                    legend_labels={'r': 'Candidate causal genes',
                                   'k': 'Potential false positive genes'}
                    if trait == 'Crohns' else None)
    # Make box plot for incorrect tissue
    wrong_tissue_results_dir = f'fusion_twas/results/{incorrect_tissue}_{trait}'
    wrong_tissue_results = pd.concat([pd.read_table(f'{wrong_tissue_results_dir}/chr{chrom}.txt',
                                                    usecols=['ID', 'TWAS.P'],
                                                    index_col='ID', delim_whitespace=True)
                                      for chrom in range(1, 23)])
    # noinspection PyUnresolvedReferences
    correct_tissue_hits['p_wrong_tissue'] = \
        wrong_tissue_results['TWAS.P'].loc[correct_tissue_hits.index]
    correct_tissue_hits['p_wrong_tissue'].fillna(1, inplace=True)
    wrong_tissue_groups = [-np.log10(correct_tissue_hits[
                                         correct_tissue_hits['block'] == block]['p_wrong_tissue'])
                           for block, gene in zip(blocks_with_causal_gene, ordered_causal_genes)]
    grouped_barplot(wrong_tissue_groups,
                    save_to=f'{wrong_tissue_results_dir}/multi_gene_blocks',
                    ylabel='$-log_{10}(p)$',
                    colors=colors,
                    bar_labels=bar_labels,
                    width_scale=1.25)

