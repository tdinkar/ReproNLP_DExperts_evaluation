import matplotlib.pyplot as plt

# Comment providing context about the origin of the code
''' Original code adapted from paper DEXPERTS: Decoding-Time Controlled Text Generation
with Experts and Anti-Experts, link to github page: https://github.com/alisawuffles/DExperts/tree/main'''

# Function to label bars in a bar plot
def label_bars(ax, rects):
    """
    Attach a text label over each bar displaying its height
    """
    for rect in rects:
        width = rect.get_width()
        ax.text(
            rect.get_x() + width / 2.0, rect.get_y() + 0.01,
            f'{width:.2f}',
            ha='center', va='bottom'
        )
        
def plt_figure(res, working_dir):
    plt.figure()
    fig, axes = plt.subplots(2, 2, figsize=(18,4))
    fig.tight_layout()
    plt.style.use('seaborn-whitegrid')
    height = 0.05
    ypos = [height* i for i in range(4)]
    ypos.reverse()

    for i, pair in enumerate(res):
        m1, m2 = pair.split(',')
        m1_bars = [res[pair][c][m1] for c in res[pair]]
        equal_bars = [res[pair][c]['equal'] for c in res[pair]]
        m2_bars = [res[pair][c][m2] for c in res[pair]]
        if m1 == 'Ensemble':
            m1 = 'DExperts'
        
        ax0 = i // 2
        ax1 = i % 2
        ax = axes[ax0, ax1]
        ax.set_yticks(ypos)
        rects = ax.barh(ypos, m1_bars, alpha=0.8, height=height-0.01, label=m1, color='#1c9099')
        label_bars(ax, rects)
        rects = ax.barh(ypos, equal_bars, alpha=0.8, height=height-0.01, left=m1_bars, label='equal', color='#a6bddb')
        label_bars(ax, rects)
        rects = ax.barh(ypos, m2_bars, alpha=0.8, height=height-0.01, left=[a+b for a,b in zip(m1_bars, equal_bars)], label=m2, color='#ece2f0')
        label_bars(ax, rects)
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        ax.set_yticks(ypos)
        ax.set_yticklabels(['Original', 'Reproduced', 'Experiment 1', 'Experiment 3'])
        ax.set_xticklabels([])

        ax.legend(loc='center right', bbox_to_anchor=(1.28, 0.5))
        fig.subplots_adjust(top=0.9, left=0.16, right=0.81, bottom=0.27, wspace=0.5, hspace=0.4)

    # Save the figure as an image file
    plt.savefig(working_dir+'toxicity_human_eval_results.png')
