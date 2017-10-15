import matplotlib, numpy as np, pandas as pd, regex, subprocess
from utils import get_gene_model
from plinkio import plinkfile
from pybedtools import BedTool
from rpy2.robjects import pandas2ri, r
from scipy.stats import norm, pearsonr

# Load gene locations

def get_TWAS_results(results_dir, gene_to_location={
        row.attrs['gene_name']: (f'chr{row[0]}', int(row[3]), int(row[4]), row[6])
        for row in BedTool('data/ensembl/ensembl_genes.gtf.gz')
        # e.g. PNRC2 appears twice, once on "1" and once on "GL000191.1"
        if row[0].isnumeric() or row[0] == 'X' or row[0] == 'Y'}):
    TWAS_results = pd.concat([pd.read_table(f'{results_dir}/chr{chrom}.txt',
                                            index_col='ID', delim_whitespace=True)
                               for chrom in range(1, 23)])
    # noinspection PyUnresolvedReferences
    TWAS_results = TWAS_results[(TWAS_results.index.str.len() > 0) &
                                ~pd.isnull(TWAS_results['TWAS.P'])]
    TWAS_results['CHR'] = 'chr' + TWAS_results['CHR'].astype(int).astype(str)
    locations = [gene_to_location[gene] for gene in TWAS_results.index]
    chroms, TWAS_results['start'], TWAS_results['end'], strands = \
        (np.array([location[i] for location in locations]) for i in range(4))
    assert (chroms == TWAS_results['CHR'].values).all()
    TWAS_results['TSS'] = np.where(strands == '+', TWAS_results['start'], TWAS_results['end'])
    TWAS_results.sort_values(['CHR', 'TSS'], inplace=True)
    return TWAS_results

def get_LD(chrom, rs1, rs2, use_STARNET=False):
    if rs1 == rs2: return 1
    if use_STARNET:
        bfile = 'data/STARNET/genotypes/STARNET'
    else:
        bfile = f'fusion_twas/LDREF/1000G.EUR.{chrom[3:]}'
    try:
        plink_output = subprocess.check_output(
            f'plink --bfile {bfile} --ld {rs1} {rs2} 2> /dev/null',
            shell=True)
    except subprocess.CalledProcessError:
        return np.nan
    return float(regex.search(b'R-sq = (.*?)\s', plink_output).group(1))

if __name__ == '__main__':
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from manhattan import manhattan
    # Note: have to set text size after importing manhattan because manhattan imports seaborn
    for text_type in 'xtick', 'ytick', 'axes':
        plt.rc(text_type, labelsize='x-large')
    pandas2ri.activate()
    pd.set_option('display.max_rows', 9999)  # display all rows of dataframe
    pd.set_option('display.expand_frame_repr', False)  # don't wrap dfs with many columns

    for trait_name, tissue, chrom, causal_gene, other_genes in (
            ('LDL', 'LIV', 'chr1', 'SORT1', ('SARS', 'CELSR2', 'PSRC1', 'PSMA5',
                                             'SYPL2', 'ATXN7L2', 'AMIGO1', 'GSTM4')),
            ('LDL', 'LIV', 'chr1', 'IRF2BP2', ('RP4-781K5.7',)),
            ('Crohns', 'Blood', 'chr16', 'NOD2', ('HEATR3', 'ADCY7', 'BRD7', 'SNX20'))):
        locus_genes = (causal_gene,) + other_genes

        # TWAS Manhattan plot of case study locus

        ref_panel = f'{tissue}_{trait_name}'
        results_dir = f'fusion_twas/results/{ref_panel}'
        TWAS_results = get_TWAS_results(results_dir)
        locus_gene_indices = np.flatnonzero(TWAS_results.index.isin(locus_genes))
        lowest_gene_index, highest_gene_index = locus_gene_indices[0], locus_gene_indices[-1]
        locus_TWAS_results = TWAS_results.iloc[lowest_gene_index : highest_gene_index + 1]
        for start_on_right in False, True:
            manhattan_file = f'{results_dir}/TWAS_manhattan_{causal_gene}_locus' \
                             f'{"_2" if start_on_right else ""}.png'
            manhattan(locus_TWAS_results['CHR'], locus_TWAS_results['TSS'], locus_TWAS_results['TWAS.P'],
                      manhattan_file, labels=locus_TWAS_results.index, num_tests=len(TWAS_results),
                      start_on_right=start_on_right)

        # Report SNPs tagged by each model, from lowest to highest p

        weight_dir = f'fusion_twas/WEIGHTS/{ref_panel}'
        trait_to_sumstats = {
            'CAD': 'fusion_twas/cardiogramplusc4d/cad.sumstats',
            'LDL': 'fusion_twas/LDL/ldl.sumstats',
            'Crohns': 'fusion_twas/crohns/crohns.sumstats',
        }
        sumstats_file = trait_to_sumstats[trait_name]
        rs_to_z = pd.read_table(sumstats_file, usecols=['SNP', 'Z'], index_col='SNP')['Z']
        rs_to_p = pd.Series(2 * np.where(rs_to_z > 0, norm.sf(rs_to_z), norm.cdf(rs_to_z)),
                            index=rs_to_z.index)
        dfs = {}
        models = 'blup', 'lasso', 'top1', 'enet', 'prs'
        for gene in locus_genes:
            weight_file = f'{weight_dir}/{gene}_500kb.wgt.RDat'
            r.load(weight_file)
            performance = r['cv.performance'][0]
            sorted_order = np.argsort(performance)
            best_model_index = sorted_order[-1]  # model with highest performance
            if best_model_index == 2: best_model_index = sorted_order[-2]  # if top1 is best, take 2nd-best
            df = pd.DataFrame({'weight': r['wgt.matrix'][:, best_model_index]}, index=r.snps['V2'].values)
            df = df[df['weight'] != 0]
            df['p'] = df.index.map(rs_to_p.__getitem__)  # map() doesn't work with dicts when used on indices
            df = df.sort_values('p')
            if gene == causal_gene:
                GWAS_hits = df.index[:6]
            # noinspection PyUnboundLocalVariable
            for GWAS_hit in GWAS_hits:
                df[f'LD with {GWAS_hit}'] = df.index.map(lambda rs: get_LD(
                    chrom, rs, GWAS_hit, use_STARNET=True))
            dfs[gene] = df
            print(f'{gene} ({models[best_model_index]}):')
            print(df)
            print()

        if len(other_genes) == 0: continue

        # Get predicted expression correlations across STARNET individuals
        model_weights = {}
        for gene in locus_genes:
            model_weights[gene] = get_gene_model(gene, ref_panel)
        model_weights = pd.DataFrame(model_weights, columns=model_weights)
        model_weights.fillna(0, inplace=True)
        rs_numbers_in_block = model_weights.index.values
        model_weights = model_weights.values
        STARNET_rs_number_file = 'STARNET_rs_numbers.txt'
        open(STARNET_rs_number_file, 'w').write('\n'.join(rs_numbers_in_block))
        plink_file = f'{weight_dir}/{causal_gene}_case_study'
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
        corr_with_causal_gene = pred_exp_correlation_matrix[locus_genes.index(causal_gene)]
        print(f'Predicted expression correlations with {causal_gene} from STARNET:')
        # noinspection PyRedeclaration
        for corr, gene in zip(corr_with_causal_gene, locus_genes):
            if gene == causal_gene: continue
            print(f'{gene}: {corr:.2f}')
        print()

        # Get true expression correlations

        def STARNET_expression_ID_to_genotype_ID(expression_ID, STARNET_number_to_genotype_ID={
                int(genotype_ID.split('.')[0].split('t')[2]): genotype_ID for genotype_ID in
                pd.read_table('data/STARNET/genotypes/STARNET.fam', delimiter=' ',
                              usecols=[0], header=None)[0].values}):
            STARNET_number = int(expression_ID.split('_')[1])
            return STARNET_number_to_genotype_ID[STARNET_number]
        # Load Ensembl ID to gene name mapping
        ensembl_ID_to_gene = {row.attrs['gene_id']: row.attrs['gene_name']
                              for row in BedTool('data/ensembl/ensembl_genes.gtf.gz')}
        # Load expression for this tissue
        STARNET_expression_file = f'data/STARNET/expression/STARNET.{tissue}.exp.for.stanford.gz'
        expression = pd.read_table(STARNET_expression_file, delimiter=' ', index_col=0)
        expression.rename(index=lambda ensembl_ID: ensembl_ID_to_gene[ensembl_ID.split('.')[0]],
                          columns=STARNET_expression_ID_to_genotype_ID, inplace=True)
        # Very rarely, multiple Ensembl IDs map to the same gene; take the first one's expression
        expression = expression[~expression.index.duplicated()]

        print(f'True expression correlations with {causal_gene}:')
        for gene in locus_genes:
            if gene == causal_gene: continue
            # noinspection PyRedeclaration
            corr = pearsonr(expression.loc[gene], expression.loc[causal_gene])[0]
            print(f'{gene}: {corr:.2f}')
        print()
        print('=' * 30)
        print()
