"""Reproduce the figure: python plot.py (requires seaborn and matplotlib)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent

def main():
    data = json.loads((ROOT / 'data.json').read_text())
    sns.set_theme(style='whitegrid', context='talk', font='DejaVu Sans')
    plt.rcParams.update({'svg.fonttype': 'none', 'svg.hashsalt': 'dsv41-performance', 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), gridspec_kw={'width_ratios': [1, 1.65]})
    fig.patch.set_facecolor('#FAFBFD')
    colors = ['#64748B', '#0D9488']
    panels = [(['4 concurrent'], [data['prefill_tokens_per_s']], ['#3B6EA8']),
              (['Decode', 'DSpark'], [data['decode_tokens_per_s'], data['dspark_tokens_per_s']], colors)]
    for ax, (labels, values, palette) in zip(axes, panels):
        ax.set_facecolor('#FAFBFD')
        sns.barplot(x=labels, y=values, hue=labels, palette=palette, legend=False, ax=ax, width=.55)
        ax.set_ylim(0, max(values) * 1.28)
        ax.set_xlabel(''); ax.set_ylabel('Tokens / s', fontsize=12, color='#64748B')
        ax.grid(axis='y', color='#E5EAF0'); ax.grid(axis='x', visible=False)
        ax.tick_params(axis='both', labelsize=12, length=0)
        sns.despine(ax=ax, left=True, bottom=True)
        for bar, value in zip(ax.patches, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+max(values)*.045,
                    f'{value:,.0f}', ha='center', va='bottom', fontsize=23, fontweight='bold', color='#15243B')
    axes[0].set_title('Prefill', loc='left', pad=26, fontsize=18)
    axes[1].set_title('Single-stream decode', loc='left', pad=26, fontsize=18)
    ratio=data['dspark_tokens_per_s']/data['decode_tokens_per_s']
    axes[1].text(.98, 1.055, f'{ratio:.2f}× with DSpark', transform=axes[1].transAxes,
                 ha='right', fontsize=12, color='#0D9488', fontweight='bold')
    fig.text(.075,.93,'DeepSeek V4.1 Flash', fontsize=27, fontweight='bold', color='#15243B')
    fig.text(.075,.875,'4 × NVIDIA GB300  ·  DP4 / EP4  ·  10K-token input', fontsize=13, color='#64748B')
    fig.subplots_adjust(left=.075,right=.965,bottom=.17,top=.74,wspace=.38)
    for extension in ['png','svg']:
        fig.savefig(ROOT/f'performance.{extension}',dpi=240,facecolor=fig.get_facecolor(),
                    metadata={'Date': None} if extension == 'svg' else None)
    svg = ROOT / 'performance.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)

if __name__ == '__main__':
    main()
