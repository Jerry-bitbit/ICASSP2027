"""Rebuild manuscript Figure 4 and its statistics from released reference records.

This command reads frozen selections; it neither runs the detector nor tunes.
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.paths import ROOT, OUTPUT
from .statistics import METRICS, bootstrap_mean, geometry_means, summarize

COLORS = {'noisy':'#7b8794', 'proposal':'#c75b39', 'fixed':'#2d7bb6',
          'tv':'#279267', 'clean':'#333333', 'beta_0.2':'#a766ad'}
LABELS = {'noisy':'Noisy', 'proposal':'Proposal', 'fixed':'Fixed (val-selected)',
          'tv':'TV (val-selected)', 'clean':'Clean reference', 'beta_0.2':'TV beta=0.2'}


def export_records(run, records):
    """Export completed-run records without changing its frozen configuration."""
    run, records = Path(run).resolve(), Path(records).resolve()
    reference = ROOT/'results/weak_structure'
    if records == reference.resolve() or records == run:
        raise ValueError('Export to a new records directory, separate from the released reference/run')
    complete = json.loads((run/'test_complete.json').read_text())
    if not complete['complete'] or not (run/'report/report_complete.json').exists():
        raise ValueError('Finish test evaluation and reporting before exporting paper records')
    records.mkdir(parents=True,exist_ok=True)
    for name in ['run_config.json','environment.json','manifest.json','selected_operating_points.json',
                 'validation_selection_scores.csv','development_complete.json','test_complete.json']:
        shutil.copyfile(run/name,records/name)
    for name in ['selected_per_sample.csv','geometry_means.csv','main_table.csv',
                 'main_table_with_paired_ci.csv','paired_comparisons.csv','all_beta_paired_comparisons.csv',
                 'geometry_paired_differences.csv','test_tradeoff_curve.csv','validation_tradeoff_curve.csv',
                 'geometric_region_validity.csv','interpretation.json','report.md']:
        shutil.copyfile(run/'report'/name,records/name)
    for split in ['validation','test']:
        frame=pd.read_csv(run/f'{split}_per_sample.csv',float_precision='round_trip')
        frame=frame[frame.method.isin(['fixed_scan','tv_scan'])]
        geometry_means(frame,['method','parameter']).to_csv(records/f'{split}_scan_geometry_means.csv',index=False)
    shutil.copyfile(reference/'paper_values.json',records/'paper_values.json')
    # The auto-generated report's image links remain usable in this compact export.
    for name in ['test_tradeoff.png','test_robustness.png']:
        shutil.copyfile(run/'report'/name,records/name)
    print(f'Exported frozen paper records to {records}')


def read_records(records):
    records = Path(records)
    config = json.loads((records/'run_config.json').read_text())
    frozen = json.loads((records/'selected_operating_points.json').read_text())
    selected = pd.read_csv(records/'selected_per_sample.csv', float_precision='round_trip')
    return config, frozen, selected


def verify(records):
    records = Path(records)
    config, frozen, selected = read_records(records)
    assert frozen['frozen'] and frozen['source_split'] == 'validation'
    assert frozen['config'] == config
    assert selected.geometry.nunique() == 100
    assert set(selected.method) == set(LABELS)
    assert not selected.duplicated(['geometry','sigma255','repeat','a','method']).any()
    assert (selected.groupby(['geometry','method']).size() == 18).all()
    assert set(selected.sigma255) == set(config['sigma255'])
    assert set(selected.a) == set(config['strengths'])
    assert set(selected['repeat']) == set(range(config['noise_repeats']))
    for method, parameter in [('fixed',frozen['eta']),('tv',frozen['beta']),('beta_0.2',.2)]:
        assert (selected.loc[selected.method == method,'parameter'] == parameter).all()
    audited = selected[selected.method.isin(['tv','beta_0.2'])]
    assert (audited.budget_residual <= 1e-12).all()
    assert audited.audit_pass.all()
    main, comparisons, differences, geometry = summarize(selected, config)
    for name, frame, keys, columns in [
        ('main_table.csv',main,['a','method'],METRICS),
        ('paired_comparisons.csv',comparisons,['stratum','target','reference','metric'],['difference','ci_low','ci_high']),
        ('geometry_means.csv',geometry,['a','method','geometry'],METRICS),
    ]:
        reference = pd.read_csv(records/name, float_precision='round_trip')
        a = frame.set_index(keys).sort_index()
        b = reference.set_index(keys).sort_index()
        assert a.index.equals(b.index)
        np.testing.assert_allclose(a[columns], b[columns], rtol=0, atol=2e-12)

    curves = {}
    for split, count in [('validation',40),('test',100)]:
        scan = pd.read_csv(records/f'{split}_scan_geometry_means.csv', float_precision='round_trip')
        assert not scan.duplicated(['method','parameter','geometry']).any()
        assert (scan.groupby(['method','parameter']).size() == count).all()
        assert scan.geometry.nunique() == count
        assert set(scan.loc[scan.method=='fixed_scan','parameter']) == set(np.arange(101)/100)
        assert set(scan.loc[scan.method=='tv_scan','parameter']) == set(config['betas'])
        curve = scan.groupby(['method','parameter'])[METRICS].mean().reset_index()
        released = pd.read_csv(records/f'{split}_tradeoff_curve.csv', float_precision='round_trip')
        np.testing.assert_allclose(curve[METRICS], released[METRICS], rtol=0, atol=2e-12)
        curves[split] = curve
    validation = curves['validation']
    for method, value in [('fixed_scan',frozen['eta']),('tv_scan',frozen['beta'])]:
        rows = validation[validation.method==method]
        maximum = rows.ap.max()
        tied = rows.loc[maximum-rows.ap <= config['selection']['tie_tolerance'],'parameter']
        assert value == tied.min()  # Verify the saved choice; never write a new choice.
    manifest = json.loads((records/'manifest.json').read_text())
    ids = [row['id'] for row in manifest['geometries']]
    assert len(ids) == len(set(ids)) == 142
    overall = geometry_means(selected,['method']).groupby('method')[METRICS].mean()
    primary = comparisons[(comparisons.stratum=='overall') & (comparisons.target=='tv') &
                          (comparisons.reference=='proposal') & (comparisons.metric=='ap')].iloc[0]
    values = {
        'proposal_ap':float(overall.loc['proposal','ap']),
        'tv_ap':float(overall.loc['tv','ap']),
        'fixed_ap':float(overall.loc['fixed','ap']),
        'tv_minus_proposal_ap':float(primary.difference),
        'primary_ci_low':float(primary.ci_low), 'primary_ci_high':float(primary.ci_high),
        'tv_background_ratio':float(overall.loc['tv','background_rmse_ratio']),
        'fixed_background_ratio':float(overall.loc['fixed','background_rmse_ratio']),
    }
    for a in config['strengths']:
        row = comparisons[(comparisons.stratum==f'a={a:g}') & (comparisons.target=='tv') &
                          (comparisons.reference=='proposal') & (comparisons.metric=='ap')].iloc[0]
        values[f'ap_gain_a_{a:g}'] = float(row.difference)
    # Rounded values transcribed from the supplied revised PDF, Section 3.3.
    expected = json.loads((records/'paper_values.json').read_text())
    for name, printed in expected.items():
        assert f'{values[name]:.4f}' == printed, (name, values[name], printed)
    print('Verified Section 3.3: 100 geometry clusters, frozen selections, scans and paired intervals.')
    return config, frozen, selected, curves, main, comparisons, values


def build(records, output):
    output = Path(output)
    figures, stats = output/'figures', output/'statistics'
    figures.mkdir(parents=True, exist_ok=True)
    stats.mkdir(parents=True, exist_ok=True)
    config, frozen, selected, curves, main, comparisons, values = verify(records)
    main.to_csv(stats/'weak_structure_main.csv',index=False)
    comparisons.to_csv(stats/'weak_structure_paired.csv',index=False)
    (stats/'weak_structure_paper_values.json').write_text(json.dumps(values,indent=2)+'\n')
    aggregates = geometry_means(selected,['a','method'])
    points = geometry_means(selected,['method']).groupby('method')[METRICS].mean()
    with plt.rc_context(plt.rcParamsDefault):
        plt.rcParams.update({'pdf.fonttype':42, 'ps.fonttype':42})
        fig, ax = plt.subplots(figsize=(7,5),layout='constrained')
        for method in LABELS:
            rows = [(a,*bootstrap_mean(group.ap.to_numpy(),config))
                    for a,group in aggregates[aggregates.method==method].groupby('a')]
            data = np.array(rows)
            ax.plot(data[:,0],data[:,1],'o-',color=COLORS[method],label=LABELS[method])
            ax.fill_between(data[:,0],data[:,2],data[:,3],color=COLORS[method],alpha=.1)
        ax.set(xlabel='DRUNet noise-map multiplier a',ylabel='Mean per-image AP',
               xticks=config['strengths'],title='Test robustness (geometry bootstrap 95% CI)')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
        for ext in ['pdf','png']:
            fig.savefig(figures/f'test_robustness.{ext}',dpi=170)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7,5),layout='constrained')
        for method,label,color in [('fixed_scan','All 101 fixed eta values',COLORS['fixed']),
                                   ('tv_scan','All 6 TV beta values',COLORS['tv'])]:
            group = curves['test'][curves['test'].method==method].sort_values('parameter')
            ax.plot(group.background_rmse_ratio,group.ap,'-',color=color,label=label,linewidth=1.8)
            if method == 'tv_scan':
                ax.scatter(group.background_rmse_ratio,group.ap,color=color,s=23)
                for row in group.itertuples():
                    ax.annotate(f'{row.parameter:g}',(row.background_rmse_ratio,row.ap),
                                xytext=(4,4),textcoords='offset points',fontsize=8)
        for method in ['noisy','proposal','fixed','tv']:
            p=points.loc[method]; chosen=method in ['fixed','tv']
            ax.scatter(p.background_rmse_ratio,p.ap,marker='*' if chosen else 'o',
                       s=160 if chosen else 55,color=COLORS[method],edgecolors='white',
                       zorder=5,label=LABELS[method])
        ax.set(xlabel='Background RMSE / noisy background RMSE',ylabel='Mean per-image AP',
               title='Test trade-off — complete diagnostic scans')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
        for ext in ['pdf','png']:
            fig.savefig(figures/f'test_tradeoff.{ext}',dpi=170)
        plt.close(fig)
    print('Generated Figure 4 panels and regenerated weak-structure statistics.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,default=None)
    parser.add_argument('--output',type=Path,default=OUTPUT/'paper')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--export-run',type=Path,help='Export a completed full run to an explicit --records directory')
    args=parser.parse_args()
    if args.export_run:
        if args.records is None:
            parser.error('--export-run requires an explicit --records destination')
        export_records(args.export_run,args.records)
    else:
        records=args.records or ROOT/'results/weak_structure'
        verify(records) if args.verify_only else build(records,args.output)
