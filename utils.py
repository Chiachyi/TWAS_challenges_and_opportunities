import numpy as np, os, pandas as pd, sys, warnings
from functools import lru_cache
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
sys.path.append('.')
warnings.filterwarnings('error', category=RuntimeWarning)

@lru_cache(maxsize=None)
def get_gene_model(gene, ref_panel, weights_dir='fusion_twas/WEIGHTS'):
    weight_file = f'{weights_dir}/{ref_panel}/{gene}_500kb.wgt.RDat'
    assert os.path.exists(os.path.dirname(weight_file))
    if not os.path.exists(weight_file):
        print(f'WARNING: weight file {weight_file} missing for gene {gene}!  '
              f'Removing from the analysis.')
        return None
    r.load(weight_file)
    performance = r['cv.performance'][0]
    sorted_order = np.argsort(performance)
    best_model_index = sorted_order[-1]  # model with highest performance
    if best_model_index == 2:
        best_model_index = sorted_order[-2]  # if top1 is best, take 2nd-best
    model_weights = r['wgt.matrix'][:, best_model_index]
    if np.isnan(model_weights).all():
        print(f'WARNING: Best model for gene {gene} has all-nan weights!  '
              f'Removing from the analysis.')
        return None
    if (model_weights == 0).all():
        print(f'WARNING: Best model for gene {gene} has all-0 weights!  '
              f'Removing from the analysis.')
        return None
    rs_numbers = r.snps['V2'].values
    assert len(model_weights) == len(rs_numbers)
    model_weights = pd.Series(data=model_weights, index=rs_numbers)
    # Remove SNPs with 0 weight (or nan weight)
    model_weights = model_weights[model_weights != 0]
    return model_weights