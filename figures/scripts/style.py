import os
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if 'Inter' in f: fm.fontManager.addfont(f)
INK='#1f1f1d'; INK2='#5f5e5a'; MUTED='#8a8a85'; GRID='#ebeae6'; SURF='#ffffff'
C={'champ':'#2a78d6','sig':'#eb6834','dn':'#8a8a85','ctl':'#1baf7a','yellow':'#c98500','mag':'#d55181','green':'#008300'}
def setup():
    mpl.rcParams.update({'font.family':'Inter','font.size':10,'axes.edgecolor':'#c9c8c2','axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.7,'axes.axisbelow':True,
        'axes.titleweight':'semibold','axes.titlesize':11.5,'axes.titlecolor':INK,'axes.titlelocation':'left','legend.frameon':False,
        'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'lines.linewidth':2,'savefig.dpi':220})
def save(fig,name):
    out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'); fig.savefig(os.path.join(out,f'{name}.png'),bbox_inches='tight'); fig.savefig(os.path.join(out,f'{name}.svg'),bbox_inches='tight'); plt.close(fig)
