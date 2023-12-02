import numpy as np
import matplotlib.pyplot as plt

from linora.chart._config import Options
from linora.chart._base import Coordinate
from linora.chart._arrow import Arrow
from linora.chart._bar import Bar
from linora.chart._boxplot import Boxplot
from linora.chart._circle import Circle
from linora.chart._coherence import Coherence
from linora.chart._csd import Csd
from linora.chart._ellipse import Ellipse
from linora.chart._errorbar import Errorbar
from linora.chart._fillline import Fillline
from linora.chart._hist import Hist
from linora.chart._hist2d import Hist2d
from linora.chart._hlines import Hlines
from linora.chart._line import Line
from linora.chart._line3D import Line3D
from linora.chart._pie import Pie
from linora.chart._polygon import Polygon
from linora.chart._radar import Radar
from linora.chart._rectangle import Rectangle
from linora.chart._regularpolygon import RegularPolygon
from linora.chart._scatter import Scatter
from linora.chart._scatter3D import Scatter3D
from linora.chart._vlines import Vlines
from linora.chart._wedge import Wedge


__all__ = ['Plot']


classlist = [
    Coordinate, Arrow, Bar, Boxplot, Circle, Coherence, Csd, Ellipse, Errorbar, Fillline, 
    Hist, Hist2d, Hlines, Line, Line3D,
    Pie, Polygon, Radar, Rectangle, RegularPolygon, Scatter, Scatter3D, Vlines, Wedge
]

class Plot(*classlist):
    def __init__(self, config=None):
        super(Plot, self).__init__()
        if config is not None:
            self.set_config(config)
        key = np.random.choice(list(Options.color), size=len(Options.color), replace=False)
        t = {i:np.random.choice(Options.color[i], size=len(Options.color[i]), replace=False) for i in key}
        self._params.color = sorted([[r+k*7, j, i] for r, i in enumerate(key) for k, j in enumerate(t[i])])
#         if len(args)!=0:
#             if isinstance(args[0], dict):
#                 for i,j in args[0].items():
#                     setattr(self._params, i, j)
#         if kwargs:
#             for i,j in kwargs.items():
#                 setattr(self._params, i, j)
                
    def _execute(self):
        fig = plt.figure(**self._params.figure)
        with plt.style.context(self._params.theme):
            mode = set([j['plotmode'].split('_')[-1] if '_' in j['plotmode'] else 'rectilinear' for i,j in self._params.ydata.items()])
            if len(mode)==1:
                projection = list(mode)[0]
            elif len(mode)==2 and 'rectilinear' in mode and '3d' in mode:
                projection = '3d'
            else:
                raise ValueError('There are two different coordinate systems.')
            ax = fig.add_subplot(projection=projection)
            
            for i,j in self._params.ydata.items():
                if 'transform' in j['kwargs']:
                    if j['kwargs']['transform'] in ['transData', 'Data']:
                        j['kwargs']['transform'] = ax.transData
                    elif j['kwargs']['transform'] in ['transAxes', 'Axes']:
                        j['kwargs']['transform'] = ax.transAxes
                    elif j['kwargs']['transform'] in ['transFigure', 'Figure']:
                        j['kwargs']['transform'] = fig.transFigure
                        j['transform'] = 'fig'
            
        ax = self._execute_ax(fig, ax)
        return fig
            
    def _execute_ax(self, fig, ax):
        for i,j in self._params.ydata.items():
            if i in self._params.twin:
                if self._params.twin[i]=='x':
                    ax_new = ax.twinx()
                    j['plotfunc'](fig, ax_new, i, j)
                    if 'x' in self._params.label:
                        self._execute_label(ax_new, 'x')
                    if 'x' in self._params.axis:
                        self._execute_axis(ax_new, 'x')
                else:
                    ax_new = ax.twiny()
                    j['plotfunc'](fig, ax_new, i, j)
                    if 'y' in self._params.label:
                        self._execute_label(ax_new, 'y')
                    if 'y' in self._params.axis:
                        self._execute_axis(ax_new, 'y')
            else:
                j['plotfunc'](fig, ax, i, j)
                self._execute_label(ax, 'normal')
                self._execute_axis(ax, 'normal')
            
        #legend
        if len(self._params.legend)>0:
            if self._params.legend['loc'] not in [None, 'None', 'none']:
                ax.legend(**self._params.legend)
        else:
            t = ['ellipse', 'regularpolygon', 'rectangle', 'circle', 'polygon', 'boxplot', 'wedge', 'arrow']
            if len([1 for i,j in self._params.ydata.items() if j['plotmode'] not in t])>1:
                ax.legend(loc='best')
        #spines
        if len(self._params.spine['alpha'])>0:
            for i,j in self._params.spine['alpha'].items():
                ax.spines[i].set_alpha(j)
        if len(self._params.spine['color'])>0:
            for i,j in self._params.spine['color'].items():
                ax.spines[i].set_color(j)
        if len(self._params.spine['width'])>0:
            for i,j in self._params.spine['width'].items():
                ax.spines[i].set_linewidth(j)
        if len(self._params.spine['style'])>0:
            for i,j in self._params.spine['style'].items():
                ax.spines[i].set_linestyle(j)
        if len(self._params.spine['position'])>0:
            for i,j in self._params.spine['position'].items():
                ax.spines[i].set_position(j)
        if len(self._params.spine['show'])>0:
            for i,j in self._params.spine['show'].items():
                ax.spines[i].set_visible(j)
        #title
        if self._params.title['label'] is not None:
            ax.set_title(**self._params.title)
        #text
        if len(self._params.text)>0:
            for i in self._params.text:
                ax.text(**self._params.text[i])
        #annotate
        if len(self._params.annotate)>0:
            for i in self._params.annotate:
                ax.annotate(**self._params.annotate[i])
        return ax
    
    def _execute_label(self, ax, mode):
        if self._params.label[mode]['xlabel']['xlabel'] is not None:
            ax.set_xlabel(**self._params.label[mode]['xlabel'])
        if self._params.label[mode]['ylabel']['ylabel'] is not None:
            ax.set_ylabel(**self._params.label[mode]['ylabel'])
            
    def _execute_axis(self, ax, mode):
        if self._params.axis[mode]['axis'] is not None:
            ax.axis(self._params.axis[mode]['axis'])
        if self._params.axis[mode]['xlabel'] is not None:
            if len(self._params.axis[mode]['xlabel'])==0:
                ax.set_xticks(self._params.axis[mode]['xlabel'])
            elif isinstance(self._params.axis[mode]['xlabel'][0], (list, tuple, np.ndarray)):
                ax.set_xticks(self._params.axis[mode]['xlabel'][0])
                ax.set_xticklabels(self._params.axis[mode]['xlabel'][1])
            else:
                ax.set_xticks(self._params.axis[mode]['xlabel'])
        if self._params.axis[mode]['ylabel'] is not None:
            if len(self._params.axis[mode]['ylabel'])==0:
                ax.set_yticks(self._params.axis[mode]['ylabel'])
            elif isinstance(self._params.axis[mode]['ylabel'][0], (list, tuple, np.ndarray)):
                ax.set_yticks(self._params.axis[mode]['ylabel'][0])
                ax.set_yticklabels(self._params.axis[mode]['ylabel'][1])
            else:
                ax.set_yticks(self._params.axis[mode]['ylabel'])
        ax.tick_params(axis='x', **self._params.axis[mode]['xtick'])
        ax.tick_params(axis='y', **self._params.axis[mode]['ytick'])
        if self._params.axis[mode]['xinvert']:
            ax.invert_xaxis()
        if self._params.axis[mode]['yinvert']:
            ax.invert_yaxis()
        if self._params.axis[mode]['xtickposition'] is not None:
            ax.xaxis.set_ticks_position(self._params.axis[mode]['xtickposition'])
        if self._params.axis[mode]['ytickposition'] is not None:
            ax.yaxis.set_ticks_position(self._params.axis[mode]['ytickposition'])