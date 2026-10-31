import sys,matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from style import *; setup()
N=[('v0.1','10/21','first Skill\ncontract, roles,\nCPU check','base'),
   ('v0.2','10/21','agent-written\nchecklist\npass 73%→83%','ok'),
   ('v0.3.x','10/21–22','equivalence test,\njudge trials','ok'),
   ('v0.4','10/22','agent revision\npass 65% vs 77%','no'),
   ('v0.5','10/25','4 models compete,\nFlash wins; parent\nclones 19%→2%','ok'),
   ('v0.6 ×6','10/26','6 candidates,\nnone passes','no'),
   ('v0.5.1','10/29','+6 GPU\nexperience cards,\nno regression','ok'),
   ('v0.7','10/30','5 models, new\nevidence; GPT-6.1:\ndistinct ideas 32→42','ok')]
col={'base':MUTED,'ok':C['champ'],'no':C['sig']}
fig,ax=plt.subplots(figsize=(13,3.6)); ax.set_xlim(-0.6,len(N)-0.4); ax.set_ylim(-1.45,1.45); ax.axis('off')
ax.plot([0,len(N)-1],[0,0],color='#d9d8d2',lw=2,zorder=1)
for i,(v,d,txt,st) in enumerate(N):
    c=col[st]; mk='o' if st!='no' else 'X'
    ax.scatter([i],[0],s=260 if st!='no' else 170,color=c,marker=mk,zorder=3,edgecolor='white',linewidth=2)
    ax.text(i,0.42,v,ha='center',fontsize=12.5,fontweight='bold',color=c)
    ax.text(i,0.85,d,ha='center',fontsize=9,color=MUTED)
    ax.text(i,-0.42,txt,ha='center',va='top',fontsize=9,color=INK2,linespacing=1.35)
ax.text(-0.55,1.18,'Skill lineage: agents propose revisions, a fixed benchmark decides',fontsize=13,fontweight='semibold',color=INK,va='bottom')
ax.scatter([5.6],[1.27],s=60,color=C['champ']); ax.text(5.7,1.27,'promoted',va='center',fontsize=9,color=INK2)
ax.scatter([6.6],[1.27],s=50,color=C['sig'],marker='X'); ax.text(6.7,1.27,'rejected',va='center',fontsize=9,color=INK2)
save(fig,'F8_skill_lineage')
