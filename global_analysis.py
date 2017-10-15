import matplotlib, numpy as np, pandas as pd, parse, subprocess
matplotlib.use('agg')
import matplotlib.pyplot as plt, seaborn
plt.style.use('default')
from collections import defaultdict
from dotdict import dotdict
from utils import get_gene_model
from functools import lru_cache
from itertools import islice
from matplotlib.ticker import FuncFormatter
from plinkio import plinkfile
from pybedtools import BedTool
from scipy.stats import pearsonr

def parse_line(parser, line):
    return dotdict(parser.parse(line).__dict__['named'])

def load_TWAS_hits(ref_panel, multi_hit_blocks_only=True):
    line_format = '{gene} ({chrom}:{start}-{end}): p < {p:.0g}, Bonf q < {q:.0g}'
    parser = parse.compile(line_format)
    TWAS_hits = defaultdict(list)
    TWAS_results_file = f'fusion_twas/results/{ref_panel}/TWAS_results.txt'
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
            TWAS_hits[key].append(value)
        TWAS_hits['block'].append(block_number)
    TWAS_hits = pd.DataFrame(TWAS_hits, columns=TWAS_hits)
    TWAS_hits.set_index('gene', inplace=True)
    if multi_hit_blocks_only:
        TWAS_hits['block_size'] = TWAS_hits.block.map(TWAS_hits.block.value_counts())
        TWAS_hits = TWAS_hits[TWAS_hits['block_size'] > 1]
    return TWAS_hits

@lru_cache(maxsize=None)
def get_pred_exp_correlation_matrix(ref_panel, locus_genes):
    # Get predicted expression correlations across STARNET individuals
    assert top_gene in locus_genes
    model_weights = {}
    for gene in locus_genes:
        model_weights[gene] = get_gene_model(gene, ref_panel)
    model_weights = pd.DataFrame(model_weights, columns=model_weights)
    model_weights.fillna(0, inplace=True)
    rs_numbers_in_block = model_weights.index.values
    model_weights = model_weights.values
    STARNET_rs_number_file = 'STARNET_rs_numbers.txt'
    open(STARNET_rs_number_file, 'w').write('\n'.join(rs_numbers_in_block))
    weight_dir = f'fusion_twas/WEIGHTS/{ref_panel}'
    plink_file = f'{weight_dir}/{top_gene}_locus'
    individuals_with_expression_file = f'{weight_dir}/individuals_with_expression_thread_0.txt'
    subprocess.check_call(f'plink --bfile data/STARNET/genotypes/STARNET '
                          f'--extract {STARNET_rs_number_file} '
                          f'--keep-fam {individuals_with_expression_file} '
                          f'--maf 1e-10 '  # re-filter on MAF because we've removed some indivs
                          f'--make-bed --out {plink_file} > /dev/null', shell=True)
    plinkio_file = plinkfile.open(plink_file)
    variants = np.array(tuple(plinkio_file))
    plinkio_rs_numbers = np.array([locus.name for locus in plinkio_file.get_loci()])
    sorted_order = np.argsort(plinkio_rs_numbers)[np.argsort(np.argsort(rs_numbers_in_block))]
    assert (plinkio_rs_numbers[sorted_order] == rs_numbers_in_block).all()
    variants = variants[sorted_order]
    predicted_expression = model_weights.T.dot(variants)
    pred_exp_correlation_matrix = np.corrcoef(predicted_expression)
    return pred_exp_correlation_matrix

def get_abs_predicted_expression_correlations(ref_panel, top_gene, locus_genes):
    pred_exp_correlation_matrix = get_pred_exp_correlation_matrix(ref_panel, locus_genes)
    corr_with_top_gene = np.abs(pred_exp_correlation_matrix[locus_genes.index(top_gene)])
    return dict(zip(locus_genes, corr_with_top_gene))

def STARNET_expression_ID_to_genotype_ID(expression_ID, STARNET_number_to_genotype_ID={
        int(genotype_ID.split('.')[0].split('t')[2]): genotype_ID for genotype_ID in
        pd.read_table('data/STARNET/genotypes/STARNET.fam', delimiter=' ',
                      usecols=[0], header=None)[0].values}):
    STARNET_number = int(expression_ID.split('_')[1])
    return STARNET_number_to_genotype_ID[STARNET_number]

def get_abs_expression_correlations(
        ref_panel, top_gene, locus_genes, expression={},
        ensembl_ID_to_gene={row.attrs['gene_id']: row.attrs['gene_name']
                            for row in BedTool('data/ensembl/ensembl_genes.gtf.gz')}):
    assert top_gene in locus_genes
    tissue = ref_panel.split('_')[0]
    # Load expression for this tissue
    if tissue not in expression:
        STARNET_expression_file = f'data/STARNET/expression/STARNET.{tissue}.exp.for.stanford.gz'
        expression[tissue] = pd.read_table(STARNET_expression_file, delimiter=' ', index_col=0)
        expression[tissue].rename(
            index=lambda ensembl_ID: ensembl_ID_to_gene[ensembl_ID.split('.')[0]],
            columns=STARNET_expression_ID_to_genotype_ID, inplace=True)
        # Very rarely, multiple Ensembl IDs map to the same gene; take the first one's expression
        expression[tissue] = expression[tissue][~expression[tissue].index.duplicated()]
    return {gene: abs(pearsonr(expression[tissue].loc[gene],
                               expression[tissue].loc[top_gene])[0])
            for gene in locus_genes}

def get_num_overlapping_variants(ref_panel, top_gene, locus_genes):
    gene_models = {gene: get_gene_model(gene, ref_panel).index for gene in locus_genes}
    return {gene: len(gene_models[gene] & gene_models[top_gene]) for gene in locus_genes}

if __name__ == '__main__':

    for ref_panel in 'LIV_LDL', 'Blood_Crohns':

        print(f'{ref_panel}:')

        # Load TWAS hits

        TWAS_hits = load_TWAS_hits(ref_panel)

        # For genes that are not the top hit in a multi-hit block, quantify their
        # overlap with the top hit in 3 different ways

        top_genes = set()
        overlap_types = (
            ('|Predicted expression correlation|', get_abs_predicted_expression_correlations),
            ('|Expression correlation|', get_abs_expression_correlations),
            ('Number of overlapping variants', get_num_overlapping_variants))
        for overlap_type, overlap_func in overlap_types:
            TWAS_hits[overlap_type] = np.empty_like(TWAS_hits.index, dtype=float)
        for block_index in TWAS_hits['block'].unique():
            locus_genes = TWAS_hits.index[TWAS_hits['block'] == block_index]
            top_gene = TWAS_hits['p'][locus_genes].idxmin()
            top_genes.add(top_gene)
            for overlap_type, overlap_func in overlap_types:
                gene_to_overlap = overlap_func(ref_panel, top_gene, tuple(locus_genes))
                TWAS_hits.ix[locus_genes, overlap_type] = \
                    np.array([gene_to_overlap[gene] for gene in locus_genes])
        is_top_gene = TWAS_hits.index.isin(top_genes)
        TWAS_hits = TWAS_hits[~is_top_gene]

        # Fig. S1: CDF of each type of overlap

        def CDF_plot(array, xlabel, save_to, log_scale=False):
            with plt.rc_context(rc={'xtick.labelsize': 'x-large',
                                    'ytick.labelsize': 'x-large',
                                    'axes.labelsize': 'x-large'}):
                plt.plot(np.sort(array), np.linspace(0, 1, len(array), endpoint=False))
                plt.xlabel(xlabel)
                if log_scale:
                    plt.xscale('log')
                    # Remove scientific notation
                    plt.gca().xaxis.set_major_formatter(FuncFormatter(
                        lambda y, pos: (
                            '{{:.{:1d}f}}'.format(int(max(-np.log10(y), 0)))).format(y)))
                plt.ylabel('CDF')
                plt.ylim(ymin=0, ymax=1)
                seaborn.despine()
                plt.savefig(save_to)
                plt.close()

        for overlap_type, overlap_func in overlap_types:
            threshold = 1 if overlap_type == 'Number of overlapping variants' else 0.2
            print(f'{overlap_type}: {100 * (TWAS_hits[overlap_type] >= threshold).mean():.0f}% '
                  f'>= {threshold} ({(TWAS_hits[overlap_type] >= threshold).sum()} of '
                  f'{len(TWAS_hits)})')
            CDF_plot(TWAS_hits[overlap_type], xlabel=overlap_type,
                     save_to=f'fusion_twas/results/{ref_panel}/{ref_panel}_'
                             f'{overlap_type.replace(" ", "_").replace("|", "").lower()}_CDF',
                     log_scale=overlap_type == 'Number of overlapping variants')

        # Fig. S4: total vs predicted expression correlation scatter plot

        plt.scatter(TWAS_hits['|Expression correlation|'],
                    TWAS_hits['|Predicted expression correlation|'])
        plt.plot((0, 1), (0, 1), 'r-')  # plot y = x
        plt.xlabel('|Expression corr.| with top gene')
        plt.ylabel('|Pred. expression corr.| with top gene')
        seaborn.despine()
        plt.savefig(f'fusion_twas/results/{ref_panel}/{ref_panel}_total_vs_pred_exp_corr')
        plt.close()