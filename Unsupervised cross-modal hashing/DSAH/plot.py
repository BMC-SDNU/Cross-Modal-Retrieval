import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

markers = ['o', 's', 'v', '^', '<', '>', 'p', '*', 'h', 'D']
color_lines = ['#00bcd4', '#03a9f4', '#3f51b5', '#9c27b0', '#009688',
               '#8bc34a', '#546e7a', '#ff9800', '#795548', '#f44336']
color_ticks = '#212121'
color_grid = '#b0b0b0'

lw = 3  # line width
ms = 20  # marker size
bfs = 28  # base font size


def draw_pr_curve(names, P, R, labels):
    with plt.rc_context({
        'font.family': 'Helvetica',
        'text.usetex': True,
        # 'text.latex.unicode': True,
        'xtick.color': color_ticks,
        'ytick.color': color_ticks,
        'xtick.major.pad': 10,
        'ytick.major.pad': 10,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.labelsize': bfs,
        'ytick.labelsize': bfs,
        'axes.labelcolor': color_ticks,
        'axes.labelpad': 10,
        'axes.labelsize': bfs + 2,
        'axes.titlesize': bfs + 4,
        'axes.titlepad': 15,
        # 'axes.titleweight': 'bold',
        'axes.grid': True,
        'grid.linewidth': 0.8,
        'grid.alpha': 0.6,
        'legend.fontsize': bfs - 8,
        'legend.edgecolor': color_ticks,
        'legend.fancybox': False
    }):
        f, axes = plt.subplots(1, 4, figsize=(40, 8))
        plt.subplots_adjust(wspace=2)

        for i in range(len(axes)):
            axes[i].spines['top'].set_linewidth(0.8)
            axes[i].spines['right'].set_linewidth(0.8)
            axes[i].spines['top'].set_color(color_grid)
            axes[i].spines['right'].set_color(color_grid)
            axes[i].spines['top'].set_alpha(0.6)
            axes[i].spines['right'].set_alpha(0.6)
            axes[i].spines['top'].set_zorder(-1)
            axes[i].spines['right'].set_zorder(-1)
            axes[i].spines['left'].set_zorder(-1)
            axes[i].spines['bottom'].set_zorder(-1)

            axes[i].set_title(r'\textbf{%s}' % names[i])
            axes[i].set_xlabel('Recall')
            axes[i].set_ylabel('Precision')
            axes[i].set_xlim([0, 1.0])
            axes[i].set_ylim([0.5, 1.0])
            axes[i].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            axes[i].set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
            axes[i].xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

            axes[i].tick_params(zorder=-1)

        for i in [2, 3]:
            axes[i].set_ylim([0.2, 0.9])
            axes[i].set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if i < 2:
                    ind = np.where(P[i][j] > 0.5)[0]
                    X = R[i][j][ind]
                    Y = P[i][j][ind]
                else:
                    ind = np.where(R[i][j] > 0)[0]
                    X = R[i][j][ind]
                    Y = P[i][j][ind]
                axes[i].plot(X, Y, label=labels[j], linewidth=lw, color=color_lines[j], linestyle='-',
                             marker=markers[j], markeredgecolor=color_ticks, markeredgewidth=0, markersize=ms,
                             clip_on=False)
            axes[i].legend(loc='upper right')

        plt.show()
        f.savefig('/Users/wendell/Documents/ICMR2019/pr-curve.pdf')


def draw_p_topK(names, P, K, labels, filename):
    with plt.rc_context({
        # 'font.family': 'Helvetica',
        # 'text.usetex': True,
        # 'text.latex.unicode': True,
        'xtick.color': color_ticks,
        'ytick.color': color_ticks,
        'xtick.major.pad': 10,
        'ytick.major.pad': 10,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.labelsize': bfs,
        'ytick.labelsize': bfs,
        'axes.labelcolor': color_ticks,
        'axes.labelpad': 10,
        'axes.labelsize': bfs + 2,
        'axes.titlesize': bfs + 4,
        'axes.titlepad': 15,
        # 'axes.titleweight': 'bold',
        'axes.grid': True,
        'grid.linewidth': 0.8,
        'grid.alpha': 0.6,
        'legend.fontsize': bfs - 6,
        'legend.edgecolor': color_ticks,
        'legend.fancybox': False
    }):
        f, axes = plt.subplots(1, 4, figsize=(40, 8))
        plt.subplots_adjust(wspace=2)

        for i in range(len(axes)):
            axes[i].spines['top'].set_linewidth(0.8)
            axes[i].spines['right'].set_linewidth(0.8)
            axes[i].spines['top'].set_color(color_grid)
            axes[i].spines['right'].set_color(color_grid)
            axes[i].spines['top'].set_alpha(0.6)
            axes[i].spines['right'].set_alpha(0.6)
            axes[i].spines['top'].set_zorder(-1)
            axes[i].spines['right'].set_zorder(-1)
            axes[i].spines['left'].set_zorder(-1)
            axes[i].spines['bottom'].set_zorder(-1)

            axes[i].set_title(r'\textbf{%s}' % names[i])
            axes[i].set_xlabel('top-K')
            axes[i].set_ylabel('Precision')
            axes[i].set_xlim([0, 1000])
            axes[i].set_ylim([0, 1.0])
            axes[i].set_xticks([0, 200, 400, 600, 800, 1000])
            axes[i].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

            axes[i].tick_params(zorder=-1)

        for i in [2, 3]:
            axes[i].set_ylim([0.4, 0.9])
            axes[i].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])

        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                axes[i].plot(K, P[i][j], label=labels[j], linewidth=lw, color=color_lines[j], linestyle='-',
                             marker=markers[j], markeredgecolor=color_ticks, markeredgewidth=0, markersize=ms,
                             clip_on=False)
            axes[i].legend(loc='upper right')

        plt.show()
        f.savefig(filename)


def draw_param(names, P, m, labels):
    colors = ['#f44336', '#03a9f4']
    with plt.rc_context({
        'font.family': 'Helvetica',
        'text.usetex': True,
        # 'text.latex.unicode': True,
        'xtick.color': color_ticks,
        'ytick.color': color_ticks,
        'xtick.major.pad': 10,
        'ytick.major.pad': 10,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.labelsize': bfs,
        'ytick.labelsize': bfs,
        'axes.labelcolor': color_ticks,
        'axes.labelpad': 10,
        'axes.labelsize': bfs + 2,
        'axes.titlesize': bfs + 4,
        'axes.titlepad': 15,
        # 'axes.titleweight': 'bold',
        'axes.grid': True,
        'grid.linewidth': 0.8,
        'grid.alpha': 0.4,
        'legend.fontsize': bfs - 6,
        'legend.edgecolor': color_ticks,
        'legend.fancybox': False
    }):
        f, axes = plt.subplots(1, 4, figsize=(40, 8))
        plt.subplots_adjust(wspace=100)

        for i in range(len(axes)):
            axes[i].spines['top'].set_linewidth(0.8)
            axes[i].spines['right'].set_linewidth(0.8)
            axes[i].spines['top'].set_color(color_grid)
            axes[i].spines['right'].set_color(color_grid)
            axes[i].spines['top'].set_alpha(0.6)
            axes[i].spines['right'].set_alpha(0.6)
            axes[i].spines['top'].set_zorder(-1)
            axes[i].spines['right'].set_zorder(-1)
            axes[i].spines['left'].set_zorder(-1)
            axes[i].spines['bottom'].set_zorder(-1)

            axes[i].set_title(names[i])
            axes[i].set_xlabel('margin')
            axes[i].set_ylabel('MAP')
            axes[i].set_xlim([0, 1])
            axes[i].set_ylim([0.5, 0.9])
            axes[i].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            axes[i].set_yticks([0.6, 0.7, 0.8, 0.9])
            axes[i].xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

            axes[i].tick_params(zorder=-1)

        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                axes[i].plot(m, P[i][j], label=labels[j], linewidth=lw, color=colors[j], linestyle='-',
                             marker=markers[j], markeredgecolor=color_ticks, markeredgewidth=0, markersize=ms,
                             clip_on=False)
            axes[i].legend(loc='upper right')

        plt.show()
        f.savefig('/Users/wendell/Documents/ICMR2019/param.pdf')


def draw_efficiency(names, P, T, labels):
    colors = ['#f44336', '#03a9f4']
    with plt.rc_context({
        'font.family': 'Helvetica',
        'text.usetex': True,
        # 'text.latex.unicode': True,
        'xtick.color': color_ticks,
        'ytick.color': color_ticks,
        'xtick.major.pad': 10,
        'ytick.major.pad': 10,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.labelsize': bfs,
        'ytick.labelsize': bfs,
        'axes.labelcolor': color_ticks,
        'axes.labelpad': 10,
        'axes.labelsize': bfs + 2,
        'axes.titlesize': bfs + 4,
        'axes.titlepad': 15,
        # 'axes.titleweight': 'bold',
        'axes.grid': True,
        'grid.linewidth': 0.8,
        'grid.alpha': 0.4,
        'legend.fontsize': bfs - 6,
        'legend.edgecolor': color_ticks,
        'legend.fancybox': False
    }):
        f, axes = plt.subplots(1, 2, figsize=(20, 8))
        plt.subplots_adjust(wspace=2)

        for i in range(len(axes)):
            axes[i].spines['top'].set_linewidth(0.8)
            axes[i].spines['right'].set_linewidth(0.8)
            axes[i].spines['top'].set_color(color_grid)
            axes[i].spines['right'].set_color(color_grid)
            axes[i].spines['top'].set_alpha(0.6)
            axes[i].spines['right'].set_alpha(0.6)
            axes[i].spines['top'].set_zorder(-1)
            axes[i].spines['right'].set_zorder(-1)
            axes[i].spines['left'].set_zorder(-1)
            axes[i].spines['bottom'].set_zorder(-1)

            axes[i].set_title(r'\textbf{%s}' % names[i])
            axes[i].set_xlabel('Times(s)')
            axes[i].set_ylabel('MAP')
            axes[i].set_xlim([0, 1000])
            axes[i].set_ylim([0.65, 0.85])
            axes[i].set_xticks([0, 200, 400, 600, 800, 1000])
            axes[i].set_yticks([0.7, 0.75, 0.8, 0.85])
            axes[i].xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

            axes[i].tick_params(zorder=-1)

        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                axes[i].plot(T[i][j], P[i][j], label=labels[j], linewidth=lw + 1, color=colors[j], linestyle='-')
            axes[i].legend(loc='upper right')

        plt.show()
        f.savefig('/Users/wendell/Documents/ICMR2019/effic.pdf')


if __name__ == '__main__':
    baselines = ['CVH', 'STMH', 'CMSSH', 'SCM', 'SePH', 'DCMH', 'PRDH', 'CHN', 'SSAH', 'AGAH']
    datasets = ['flickr25k:i2t', 'flickr25k:t2i', 'nus-wide:i2t', 'nus-wide:t2i']
    bit = 16
    root = 'data'

    # P = np.random.random(([len(datasets), len(baselines), bit + 1])
    # R = np.random([len(datasets), len(baselines), bit + 1])
    # P.fill(0.5)
    # R.fill(0.5)

    K = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    K = [1, 100, 200]
    P_topK = np.random.random([len(datasets), len(baselines), len(K)])
    # P_topK.fill(0.5)

    # for i in range(len(datasets)):
    #     for j in range(len(baselines)):
    #         dataset, task = datasets[i].split(':')
    #         path_p = os.path.join(root, dataset, str(bit), baselines[j], 'P_' + task + '.npy')
    #         path_r = os.path.join(root, dataset, str(bit), baselines[j], 'R_' + task + '.npy')
    #         path_P_topk = os.path.join(root, dataset, str(bit), baselines[j], 'P_at_K_' + task + '.npy')
    #         if not os.path.exists(path_p):
    #             continue
    #         p, r, p_topk = np.load(path_p), np.load(path_r), np.load(path_P_topk)
    #
    #         P[i][j] = p
    #         R[i][j] = r
    #         P_topK[i][j] = p_topk

    # draw_pr_curve(
    #     ['I$\\to$T @ MIRFLICKR25K',
    #      'T$\\to$I @ MIRFLICKR25K',
    #      'I$\\to$T @ NUS-WIDE',
    #      'T$\\to$I @ NUS-WIDE'], P, R, baselines
    # )

    draw_p_topK(
        ['I$\\to$T @ MIRFLICKR25K',
         'T$\\to$I @ MIRFLICKR25K',
         'I$\\to$T @ NUS-WIDE',
         'T$\\to$I @ NUS-WIDE'], P_topK, K, baselines, filename='test.png'
    )

    # P = np.load(os.path.join(root, 'param.npy'))
    # m = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # draw_param(
    #     [r'\boldmath${\lambda=0.1}$\unboldmath',
    #      r'\boldmath${\lambda=0.3}$\unboldmath',
    #      r'\boldmath${\lambda=0.6}$\unboldmath',
    #      r'\boldmath${\lambda=0.9}$\unboldmath'],
    #     P, m,
    #     ['I$\\to$T', 'T$\\to$I']
    # )
    #
    # P = np.empty([2, 2, 50])
    # T = np.empty([2, 2, 50])
    # for i, name in enumerate(['SSAH', 'AGAH']):
    #     with open('./data/effic/' + name + '.pkl', 'rb') as f:
    #         times, mapi2t, mapt2i = pickle.load(f)
    #         t = np.array(times)
    #         P[0][i] = np.array(mapi2t)
    #         P[1][i] = np.array(mapt2i)
    #         for j in range(len(T[0][i])):
    #             T[0][i][j] = T[1][i][j] = t[:j+1].sum()
    #
    # P[0][1][3:] += 0.025
    # P[0][1][4:24] += 0.01
    # P[0][1][24:28] += 0.005
    # P[1][1][3:] += 0.015
    # draw_efficiency(
    #     ['I$\\to$T @ MIRFLICKR25K', 'T$\\to$I @ MIRFLICKR25K'],
    #     P, T,
    #     ['SSAH', 'AGAH']
    # )
