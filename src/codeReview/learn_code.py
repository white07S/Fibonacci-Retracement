from utils import datefmt
from sup_res_preliminary import *
import yfinance as yf 
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import numpy as np


tick = yf.Ticker('AMD') 
hist = tick.history(period="max", rounding=True)


def fig_linregrs():
    plt.clf()
    plt.rcParams.update({'font.size': 14})
    plt.gcf().set_size_inches(1024/plt.gcf().dpi, 768/plt.gcf().dpi) 
    spec = gridspec.GridSpec(ncols=1, nrows=2, figure=plt.gcf(), height_ratios=[3, 1])
    plt.subplot(spec[1, 0])
    plt.axis('off')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.annotate(r'Mean of n-Points along x and y-axes: $\bar{x}=\frac{\sum_{i=1}^n{x_i}}{n}, \bar{y}=\frac{\sum_{i=1}^n{y_i}}{n}$' + '\n' +
                           r'Regression slope: $m=\frac{\sum_{i=1}^n(x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=1}^n(x_i-\bar{x})^2}$' + '  ' +
                           r'Regression intercept: $b=\bar{y}-m\bar{x}$' + '\n' +
                           r'Sum of Squared Residuals for expected $y_i$ $(\hat{y}_i)$: $SSR=\sum_{i=1}^n{(y_i-\hat{y}_i)^2}$' + '\n' +
                           r'Standard Error of Slope: $\sigma_m=\sqrt{\frac{SSR}{(n-2)\sum_{i=1}^n{(x_i-\bar{x})^2}}}$' + '  ' + 
                           r'Standard Error of Intercept: $\sigma_b=\sigma_m\sqrt{\frac{\sum_{i=1}^nx_i^2}{n}}$', (0, 0))
    plt.subplot(spec[0, 0])
    plt.plot(range(len(hist.Close)-3, len(hist.Close)), hist.Close.iloc[-3:], 'bo')
    xbar, ybar = (0 + 1 + 2) / 3, (hist.Close.iloc[-3] + hist.Close.iloc[-2] + hist.Close.iloc[-1]) / 3
    height = hist.Close.iloc[-3:].max() - hist.Close.iloc[-3:].min()
    plt.hlines(ybar, len(hist.Close)-3, len(hist.Close)-1, colors='r', linestyles='--')
    plt.annotate(r'$\bar{{x}}=\frac{{{}+{}+{}}}{{{}}}={}$'.format(0, 1, 2, 3, 1), (xbar + len(hist.Close)-3, (hist.Close.iloc[-3:].min() + hist.Close.iloc[-3:].max()) / 2), (xbar + len(hist.Close)-3, hist.Close.iloc[-3:].min()), color='red', va='center', arrowprops={'arrowstyle':'->', 'color':'red'})
    plt.vlines(xbar + len(hist.Close)-3, hist.Close.iloc[-3:].min(), hist.Close.iloc[-3:].max(), colors='r', linestyles='--')
    plt.annotate(r'$\bar{{y}}=\frac{{{}+{}+{}}}{{{}}}={}$'.format(hist.Close.iloc[-3], hist.Close.iloc[-2], hist.Close.iloc[-1], 3, round(ybar, 2)), (len(hist.Close)-2, ybar), (len(hist.Close)-1, ybar - height * 0.1), color='red', va='top', ha='right', arrowprops={'arrowstyle':'->', 'color':'red'})
    m = ((0 - xbar) * (hist.Close.iloc[-3] - ybar) + (1 - xbar) * (hist.Close.iloc[-2] - ybar) + (2 - xbar) * (hist.Close.iloc[-1] - ybar)) / (np.square(0-xbar)+np.square(1-xbar)+np.square(2-xbar))
    b = ybar - m * xbar
    SSR = np.square(hist.Close.iloc[-3] - (m * 0 + b)) + np.square(hist.Close.iloc[-2] - (m * 1 + b)) + np.square(hist.Close.iloc[-1] - (m * 2 + b))
    err1 = np.sqrt(SSR / ((3 - 2) * (np.square(0-xbar)+np.square(1-xbar)+np.square(2-xbar))))
    err2 = err1*np.sqrt((np.square(0)+np.square(1)+np.square(2))/3)
    plt.annotate(r'$\hat{{y}}_0={}*{}+{}={}$'.format(round(m, 2), 0, round(b, 2), round(m*0+b, 2)), (len(hist.Close) - 3, m*0+b), (len(hist.Close) - 3 + 0.1, m*0+b), va='top', arrowprops={'arrowstyle':'->'})
    plt.annotate(r'$\hat{{y}}_1={}*{}+{}={}$'.format(round(m, 2), 1, round(b, 2), round(m*1+b, 2)), (len(hist.Close) - 2, m*1+b), (len(hist.Close) - 2 + 0.15, m*1+b+height*0.01), arrowprops={'arrowstyle':'->'})
    plt.annotate(r'$\hat{{y}}_2={}*{}+{}={}$'.format(round(m, 2), 2, round(b, 2), round(m*2+b, 2)), (len(hist.Close) - 1, m*2+b), (len(hist.Close) - 1 - 0.1, m*2+b+height*0.1), ha='right', arrowprops={'arrowstyle':'->'})
    plt.plot([len(hist.Close) - 3, len(hist.Close) - 3], [hist.Close.iloc[-3], ybar], color='green')
    plt.plot([len(hist.Close) - 2, len(hist.Close) - 2], [hist.Close.iloc[-2], ybar], color='green')
    plt.plot([len(hist.Close) - 1, len(hist.Close) - 1], [hist.Close.iloc[-1], ybar], color='green')
    plt.annotate(r'$y_0-\bar{{y}}={}$'.format(round(hist.Close.iloc[-3] - ybar, 2)), (len(hist.Close) - 3, (hist.Close.iloc[-3] + ybar) / 2 + height * 0.1), (len(hist.Close) - 3 + 0.1, (hist.Close.iloc[-3] + ybar) / 2 + height * 0.1), color='green', va='center', arrowprops={'arrowstyle':'-[', 'color':'green'})
    plt.annotate(r'$y_1-\bar{{y}}={}$'.format(round(hist.Close.iloc[-2] - ybar, 2)), (len(hist.Close) - 2, (hist.Close.iloc[-2] + ybar) / 2 + height * 0.1), (len(hist.Close) - 2 + 0.1, (hist.Close.iloc[-2] + ybar) / 2 + height * 0.1), color='green', va='center', arrowprops={'arrowstyle':'-[', 'color':'green'})
    plt.annotate(r'$y_2-\bar{{y}}={}$'.format(round(hist.Close.iloc[-1] - ybar, 2)), (len(hist.Close) - 1, (hist.Close.iloc[-1] + ybar) / 2), (len(hist.Close) - 1 - 0.1, (hist.Close.iloc[-1] + ybar) / 2), color='green', va='center', ha='right', arrowprops={'arrowstyle':'-[', 'color':'green'})
    plt.plot([len(hist.Close) - 3, len(hist.Close) - 3], [hist.Close.iloc[-3], m*0+b], color='cyan')
    plt.plot([len(hist.Close) - 2, len(hist.Close) - 2], [hist.Close.iloc[-2], m*1+b], color='cyan')
    plt.plot([len(hist.Close) - 1, len(hist.Close) - 1], [hist.Close.iloc[-1], m*2+b], color='cyan')
    plt.annotate(r'$y_0-\hat{{y}}={}$'.format(round(hist.Close.iloc[-3] - (m*0+b), 2)), (len(hist.Close) - 3, (hist.Close.iloc[-3] + m*0+b) / 2), (len(hist.Close) - 3 + 0.1, (hist.Close.iloc[-3] + m*0+b) / 2), color='cyan', va='center', arrowprops={'arrowstyle':'-[', 'color':'cyan'})
    plt.annotate(r'$y_1-\hat{{y}}={}$'.format(round(hist.Close.iloc[-2] - (m*1+b), 2)), (len(hist.Close) - 2, (hist.Close.iloc[-2] + m*1+b) / 2), (len(hist.Close) - 2 + 0.1, (hist.Close.iloc[-2] + m*1+b) / 2), color='cyan', va='center', arrowprops={'arrowstyle':'-[', 'color':'cyan'})
    plt.annotate(r'$y_2-\hat{{y}}={}$'.format(round(hist.Close.iloc[-1] - (m*2+b), 2)), (len(hist.Close) - 1, (hist.Close.iloc[-1] + m*2+b) / 2 - height * 0.05), (len(hist.Close) - 1 - 0.1, (hist.Close.iloc[-1] + m*2+b) / 2 - height * 0.05), color='cyan', va='center', ha='right', arrowprops={'arrowstyle':'-[', 'color':'cyan'})
    plt.annotate((r'$m=\frac{{({}-{})*{}+({}-{})*{}+({}-{})*{}}}{{({}-{})^2+({}-{})^2+({}-{})^2}}$' + '\n' + '=${}$' + '\n' +
                     r'$SSR={}^2+{}^2+{}^2={}$' + '\n' +
                     r'$\sigma_m=\sqrt{{\frac{{{}}}{{({}-2)(({}-{})^2+({}-{})^2+({}-{})^2)}}}}={}$' + '\n' +
                     r'$\sigma_b={}\sqrt{{\frac{{{}^2+{}^2+{}^2}}{{{}}}}}={}$'
                     ).format(0, round(xbar, 2), round(hist.Close.iloc[-3] - ybar, 2), 1, round(xbar, 2), round(hist.Close.iloc[-2] - ybar, 2), 2, round(xbar, 2), round(hist.Close.iloc[-1] - ybar, 2), 0, round(xbar, 2), 1, round(xbar, 2), 2, round(xbar, 2), round(m, 2),
                             round(hist.Close.iloc[-3] - (m*0+b), 2), round(hist.Close.iloc[-2] - (m*1+b), 2), round(hist.Close.iloc[-1] - (m*2+b), 2), round(SSR, 2),
                             round(SSR, 2), 3, 0, round(xbar, 2), 1, round(xbar, 2), 2, round(xbar, 2), round(err1, 2),
                             round(err1, 2), 0, 1, 2, 3, round(err2, 2)),
                 (len(hist.Close)-2, m * 1 + b), (len(hist.Close)-1, hist.Close.iloc[-3:].min()), color='blue', va='bottom', ha='right', arrowprops={'arrowstyle':'->', 'color':'blue'})
    plt.annotate(r'$b={}-{}*{}={}$'.format(round(ybar, 2), round(m, 2), xbar, round(b, 2)), (len(hist.Close)-3, b), (len(hist.Close)-3+0.1, b), color='blue', ha='left', arrowprops={'arrowstyle':'->', 'color':'blue'})
    plt.plot([len(hist.Close)-3, len(hist.Close)-1], [b, 2 * m + b])
    ax = plt.gca()
    plt.yticks(hist.Close.iloc[-3:])
    plt.title('Closing Price Points Demonstrating Linear Regression')
    plt.xlabel('Date')
    plt.ylabel('Price')
    ax.xaxis.set_major_locator(ticker.IndexLocator(1, 0))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(datefmt(hist.index)))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('LinReg.png')


def fig_hough():
        plt.clf()
        plt.rcParams.update({'font.size': 14})
        plt.gcf().set_size_inches(1280/plt.gcf().dpi, 1024/plt.gcf().dpi) #plt.gcf().dpi=100
        spec = gridspec.GridSpec(ncols=1, nrows=2, figure=plt.gcf(), height_ratios=[3, 1])
        plt.subplot(spec[1, 0])
        plt.axis('off')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.annotate(r'Slope of Perpendicular Line: $m_p=-\frac{1}{m}, mm_p=-1$' + '\n' +
                           r'Perpencicular Line passing through Point: $y=\frac{x_0-x}{m}+y_0$' + '\n' +
                           r'Point $(x\prime, y\prime)$ of Intersection of Lines: $mx+b=\frac{x_0-x}{m}+y_0\equiv x\prime=\frac{x_0+my_0-mb}{m^2+1}, y\prime=mx\prime+b$' + '\n' +
                           r'Distance of Point to Line after simplification: $d=\frac{\left|b+mx_0-y_0\right|}{\sqrt{1 + m^2}}$' + '\n' +
                           r'$\rho=x \cos \theta+y \sin \theta$ where $\sin \theta=\frac{opposite}{hypotenuse}, \cos \theta=\frac{adjacent}{hypotenuse}$ and $y=\frac{\sin \theta}{\cos \theta}x$ while its perpendicular line is $y=-\frac{\cos \theta}{\sin \theta}x+\frac{\rho}{\sin \theta}$', (0, 0))
        plt.subplot(spec[0, 0])
        plt.plot([len(hist.Close)-10, len(hist.Close)-1], [hist.Close.iloc[-10], hist.Close.iloc[-1]], 'ro')
        plt.plot([len(hist.Close)-10, len(hist.Close)-1], [hist.Close.iloc[-10], hist.Close.iloc[-1]], 'k-')
        mn, mx = min(hist.Close.iloc[-10], hist.Close.iloc[-1]), max(hist.Close.iloc[-10], hist.Close.iloc[-1])
        plt.plot([len(hist.Close)-10, len(hist.Close)-1], [mn, mx], 'b--')
        plt.annotate(r'Diagonal length=$\sqrt{{{}^2+{}^2}}={}$'.format(9, round(mx-mn, 2), round(np.sqrt(np.square(9)+np.square(mx-mn)), 2)), (len(hist.Close)-1, mx), (len(hist.Close)-1-1, mx), ha='right', va='top', color='blue', arrowprops={'arrowstyle':'->', 'color':'blue'})
        ax = plt.gca()
        m = (hist.Close.iloc[-10] - hist.Close.iloc[-1]) / (0 - 9)
        b = hist.Close.iloc[-10] - m * 0 - mn #+ height * 0.2
        plt.annotate(r'$y={}x+{}$'.format(round(m, 2), round(b, 2)), (len(hist.Close)-5.5, (mn+mx)/2), (len(hist.Close)-5.5, mn+(mx-mn)*0.7), arrowprops={'arrowstyle':'->'})
        bperp = 0
        x0 = (m * (bperp - b)) / (m*m+1)
        angle = np.arctan((-x0/m+bperp) / (x0))
        plt.annotate('', (len(hist.Close)-10, mn), (len(hist.Close)-10 + x0, mn + -x0/m + bperp), arrowprops=dict(arrowstyle="<|-", color='red'))
        plt.gca().add_patch(mpatches.Wedge((len(hist.Close)-10 + x0, mn + -x0/m + bperp), 1, angle * 180 / np.pi - 180, angle * 180 / np.pi - 90, fill=False))
        plt.gca().add_patch(mpatches.Wedge((len(hist.Close)-10 + x0, mn + -x0/m + bperp), 0.5, angle * 180 / np.pi - 270, angle * 180 / np.pi - 180, fill=False))
        plt.annotate(r'$90\circ$', (len(hist.Close)-10 + x0 - 1, mn + -x0/m + bperp - 2))
        plt.annotate(r'$90\circ$', (len(hist.Close)-10 + x0 - 1, mn + -x0/m + bperp + 1))
        plt.gca().add_patch(mpatches.Wedge((len(hist.Close)-10, mn), 3, 0, angle * 180 / np.pi, fill=False))
        plt.annotate(r'$\theta={}^\circ$'.format(round(angle * 180/np.pi, 2)), (len(hist.Close) - 6.75, mn+0.1))
        plt.gca().add_patch(mpatches.Wedge((len(hist.Close)-10, mx), 3, 270, 270 + angle * 180 / np.pi, fill=False))
        plt.annotate(r'$\theta$', (len(hist.Close)-9.5, mx-5))
        plt.annotate((r'$\rho=\frac{{\left|{}+{}*{}-{}\right|}}{{\sqrt{{1 + {}^2}}}}$' + '\n' + '$={}\cos {}+{}\sin {}$' + '\n' + '$={}\cos {}+{}\sin {}$' + '\n' + '$={}$').format(
            round(b, 2), round(m, 2), 0, 0, round(m, 2),
            0, round(angle*180/np.pi, 2), round(hist.Close.iloc[-10]-mn, 2), round(angle*180/np.pi, 2), 9, round(angle*180/np.pi, 2), hist.Close.iloc[-1]-mn, round(angle*180/np.pi, 2), round(0 * np.cos(angle) + (hist.Close.iloc[-10]-mn) * np.sin(angle), 2)),
                     (len(hist.Close)-10+x0/2, mn + (-x0/m + bperp) / 2), (len(hist.Close)-10+x0/2, mn + (-x0/m + bperp) / 2+0.9), ha='center', color='red', arrowprops=dict(arrowstyle="->", color='red'))
        plt.plot([len(hist.Close)-10, len(hist.Close)-1], [mn, mn], 'k-')
        plt.plot([len(hist.Close)-10, len(hist.Close)-10], [mn, mx], 'k-')
        plt.annotate('{}'.format(9), (len(hist.Close)-5.5, mn), (len(hist.Close)-5.5, mn+0.5), ha='center', arrowprops=dict(arrowstyle="->"))
        plt.annotate('{}'.format(round(mx-mn, 2)), (len(hist.Close)-10, (mn+mx)/2), (len(hist.Close)-10+0.5, (mn+mx)/2), ha='left', arrowprops=dict(arrowstyle="->"))
        plt.yticks([hist.Close.iloc[-10], hist.Close.iloc[-1]])
        plt.title('Closing Price Points Demonstrating Hough transform accumulation of rho-theta for 2 point line')
        plt.xlabel('Date')
        plt.ylabel('Price')
        ax.xaxis.set_major_locator(ticker.IndexLocator(1, 0))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(datefmt(hist.index)))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig('hough.png')

def fig_slopeint():
        plt.clf()
        plt.rcParams.update({'font.size': 14})
        plt.gcf().set_size_inches(1000/plt.gcf().dpi, 1000/plt.gcf().dpi) #plt.gcf().dpi=100
        spec = gridspec.GridSpec(ncols=1, nrows=2, figure=plt.gcf(), height_ratios=[3, 1])
        plt.subplot(spec[1, 0])
        plt.axis('off')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.annotate(r'Standard slope-intercept line equation: $f(x)=y=mx+b$' + '\n'
                           r'For 2 points $(x_0, y_0), (x_1, y_1)$:' + '\n' +
                           r'Slope derived from two points: $m=\frac{\Delta y}{\Delta x}=\frac{y_0-y_1}{x_0-x_1}$' + '\n' +
                           r'Intercept derived from slope and point: $b=y_0-mx_0=y_1-mx_1$' + '\n' +
                           r'Y-axis Distance to point from line: $d=\left|mx_2+b-y_2\right|$' + '\n' +
                           r'''Pythagorean's Theorem for Right Triangles: $c^2=a^2+b^2\equiv$ $d^2=\Delta x^2+\Delta y^2$''' + '\n' +
                           r'Distance between Points: d=$\sqrt{(x_1-x_0)^2+(y_1-y_0)^2}$', (0, 0))
        plt.subplot(spec[0, 0])
        m = (hist.Close.iloc[-3] - hist.Close.iloc[-1]) / -2
        b1, b2 = hist.Close.iloc[-1] - m * 2, hist.Close.iloc[-3] - m * 0
        d = abs(m * 1 + b1 - hist.Close.iloc[-2])
        dist = np.sqrt(np.square(hist.Close.iloc[-3] - hist.Close.iloc[-1]) + np.square(-2))
        height = hist.Close.iloc[-3:].max() - hist.Close.iloc[-3:].min()
        plt.plot(range(len(hist.Close)-3, len(hist.Close)), hist.Close.iloc[-3:])
        plt.yticks(hist.Close.iloc[-3:])
        plt.plot([len(hist.Close)-3, len(hist.Close)-1], [hist.Close.iloc[-3], hist.Close.iloc[-1]], 'g--')
        #perpendicular slope: 1/-m, intercept to midpoint b=y-mx: 
        #intercept = (hist.Close.iloc[-3] + hist.Close.iloc[-1]) / 2 - (-1/m)
        ax = plt.gca()
        plt.ylim(ax.get_ylim()[0] - height * 0.1, ax.get_ylim()[1])
        bbox = ax.get_window_extent()
        drawdim = [bbox.width, bbox.height]
        xaxwdt, yaxhgt = ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_ylim()[1] - ax.get_ylim()[0]
        mvisual = (hist.Close.iloc[-3] - hist.Close.iloc[-1]) * drawdim[1] / yaxhgt / (-2 * drawdim[0] / xaxwdt)
        intcpt = ((hist.Close.iloc[-3] + hist.Close.iloc[-1]) / 2 - ax.get_ylim()[0]) * drawdim[1] / yaxhgt - (-(drawdim[0] / 2) / mvisual)
        ann = plt.annotate(r'$d=\sqrt{{({}-{})^2+({}-{})^2}}={}$'.format(hist.Close.iloc[-3], hist.Close.iloc[-1], 0, 2, round(dist, 2)), (len(hist.Close)-2, (hist.Close.iloc[-3] + hist.Close.iloc[-1]) / 2), ax.transData.inverted().transform(((drawdim[0] * 0.54)+bbox.x0, (-(drawdim[0] * 0.54)/mvisual + intcpt)+bbox.y0)), textcoords='data', color='green', ha='center', va='center', arrowprops={'arrowstyle':'-[', 'color':'green'})
        plt.annotate(r'$b={}-{}*{}={}-{}*{}={}$'.format(hist.Close.iloc[-1], round(m, 2), 2, hist.Close.iloc[-3], round(m, 2), 0, b1), (len(hist.Close)-3, hist.Close.iloc[-3]), (len(hist.Close)-3, hist.Close.iloc[-3] - height*0.1), arrowprops={'arrowstyle':'->'})
        plt.plot([len(hist.Close)-2, len(hist.Close)-2], [m * 1 + b1, hist.Close.iloc[-2]], 'r--')
        plt.annotate((r'$d=$' + '\n' + r'$\left|{}*{}+{}-{}\right|$' + '\n' + r'$={}$').format(round(m, 2), 1, b1, hist.Close.iloc[-2], round(d, 2)), (len(hist.Close)-2, (m * 1 + b1 + hist.Close.iloc[-2]) / 2), (len(hist.Close)-2+0.1, (m * 1 + b1 + hist.Close.iloc[-2]) / 2), va='center', color='red', arrowprops={'arrowstyle':'-[', 'color':'red'})
        plt.annotate(r'$m=\frac{{{}}}{{{}}}={}$'.format(round(hist.Close.iloc[-3] - hist.Close.iloc[-1], 2), 0 - 2, round(m, 2)), (len(hist.Close)-2, (hist.Close.iloc[-3] + hist.Close.iloc[-1]) / 2), (len(hist.Close)-2+0.2, (hist.Close.iloc[-3] + hist.Close.iloc[-1]) / 2 - height * 0.1), color='black', arrowprops={'arrowstyle':'->'})
        plt.plot([len(hist.Close)-3, len(hist.Close)-1], [hist.Close.iloc[-3], hist.Close.iloc[-3]], 'c--')
        plt.annotate(r'$\Delta x={}-{}={}$'.format(0, 2, 0 - 2), (len(hist.Close)-2, hist.Close.iloc[-3]), (len(hist.Close)-2, hist.Close.iloc[-3] + height * 0.10), color='cyan', ha='center', va='center', arrowprops={'arrowstyle':'-[', 'color':'cyan'})
        plt.plot([len(hist.Close)-1, len(hist.Close)-1], [hist.Close.iloc[-3], hist.Close.iloc[-1]], 'c--')
        plt.annotate(r'$\Delta y={}-{}={}$'.format(hist.Close.iloc[-3], hist.Close.iloc[-1], round(hist.Close.iloc[-3] - hist.Close.iloc[-1], 2)), (len(hist.Close)-1, (hist.Close.iloc[-3] + hist.Close.iloc[-1]) / 2), (len(hist.Close)-2+0.5, (hist.Close.iloc[-3] + hist.Close.iloc[-1]) / 2), color='cyan', ha='center', va='center', arrowprops={'arrowstyle':'-[', 'color':'cyan'})

        plt.title('Closing Price Points Demonstrating Line Calculations')
        plt.xlabel('Date')
        plt.ylabel('Price')
        ax.xaxis.set_major_locator(ticker.IndexLocator(1, 0))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(datefmt(hist.index)))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        def redraw(event):
            bbox = ax.get_window_extent()
            drawdim = [bbox.width, bbox.height]
            mvisual = (hist.Close.iloc[-3] - hist.Close.iloc[-1]) * drawdim[1] / yaxhgt / (-2 * drawdim[0] / xaxwdt)
            intcpt = ((hist.Close.iloc[-3] + hist.Close.iloc[-1]) / 2 - ax.get_ylim()[0]) * drawdim[1] / yaxhgt - (-(drawdim[0] / 2) / mvisual)
            ann.xyann = ax.transData.inverted().transform(((drawdim[0] * 0.54)+bbox.x0, (-(drawdim[0] * 0.54)/mvisual + intcpt)+bbox.y0))
            plt.gcf().canvas.draw_idle()
        cid = plt.gcf().canvas.mpl_connect('resize_event', redraw)
        plt.tight_layout()
        plt.savefig('line.png')



def fig_reimann():
        mins, maxs = calc_support_resistance(hist[-250:].Close, sortError = True,accuracy=8)
        minimaIdxs, pmin, mintrend, minwindows = mins
        maximaIdxs, pmax, maxtrend, maxwindows = maxs
        plt.clf()
        plt.gcf().set_size_inches(800/plt.gcf().dpi, 720/plt.gcf().dpi) #plt.gcf().dpi=100
        plt.subplot(211)
        plt.title('Closing Price with Resistance and Area')
        plt.xlabel('Date')
        plt.ylabel('Price')
        trendline = maxtrend[0]
        base = trendline[0][0]
        m, b, ser = trendline[1][0], trendline[1][1], hist[-250:][base:trendline[0][-1]+1].Close
        plt.plot(range(base, trendline[0][-1]+1), hist[-250:][base:trendline[0][-1]+1].Close, 'b-', label='Price')
        plt.plot((base, trendline[0][-1]+1), (m * base + b, m * (trendline[0][-1]+1) + b), 'r-', label='Resistance')
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(6))
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(datefmt(hist[-250:].index)))
        plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right')
        plt.legend()
        plt.subplot(212)
        plt.ylabel('Price Difference from Trend')
        isMin = False
        S = sum([max(0, (m * (x+base) + b) - y if isMin else y - (m * (x+base) + b)) for x, y in enumerate(ser)])
        area = S / len(ser)
        for x, y in enumerate(ser):
            plt.bar(x, (m * (x+base) + b) - y if isMin else y - (m * (x+base) + b), color='r' if (y < (m * (x+base) + b) if isMin else y > (m * (x+base) + b)) else 'gray')

        plt.annotate(r'S={}, $\frac{{{}}}{{{}}}$={}$\frac{{\$}}{{day}}$'.format(round(S, 2), round(S, 2), len(range(base, trendline[0][-1]+1)), round(area, 2)) + '\n' + r'Reimann Sum where $\Delta x=x_i-x_{i-1}$,' + '\n' + '$x_i^* \in [x_{i-1}, x_i]$: $S=\sum_{i=1}^n{f(x_i^*)\Delta x_i}$', (0, plt.gca().get_ylim()[0]+5), va='bottom')
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(6))
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(datefmt(hist[-250:].index)))
        plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right')
        plt.savefig('reimann.png')



if __name__ == "__main__":
    fig_slopeint()
    fig_linregrs()
    fig_hough()
    fig_reimann()