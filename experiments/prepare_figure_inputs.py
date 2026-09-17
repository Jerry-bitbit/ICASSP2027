"""Extract the paper's fixed kodim23 example from a completed Kodak rerun."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.paths import DATA, RESULTS


def prepare(cache, results):
    cache, results = Path(cache), Path(results)
    if not cache.is_file():
        raise FileNotFoundError(f'Missing Kodak DRUNet cache: {cache}; run the Kodak budget experiment first')
    with np.load(cache, allow_pickle=False) as stored:
        arrays = {f'kodim23_{key}':stored[f'kodim23_{key}'] for key in ['clean','noisy','proposal']}
    frame = pd.read_csv(results/'icassp_budget_audit.csv',float_precision='round_trip')
    rows = frame[(frame.image=='kodim23') & (frame.method=='adaptive') & frame.beta.isin([.2,.6,.8])]
    if len(rows) != 3:
        raise ValueError('Expected the three fixed kodim23 budget rows, with no image selection')
    folder = results/'figure_inputs'
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'overview_sample.npz',**arrays)
    rows.to_csv(folder/'overview_row.csv',index=False)
    print(f'Prepared Figure 1/3 inputs in {folder}')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=DATA/'cache/icassp_drunet_sigma25.npz')
    parser.add_argument('--results',type=Path,default=RESULTS)
    args=parser.parse_args()
    prepare(args.cache,args.results)
