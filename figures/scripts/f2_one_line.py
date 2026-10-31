import json,numpy as np,matplotlib.pyplot as plt,sys
import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from style import *; setup()
C30=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','results','data','curves_30ep.json')))
def runs(prefix,only=None):
    out=[]
    for j,v in C30.items():
        if j.startswith('tr-'+prefix) and v.get('finished') and len(v['epochs'])>=30 and (only is None or any(o in j for o in only)):
            out.append(np.array(v['epochs'])[:30])
    return out
S={'champ':runs('r02a-e29092'),'sig':runs('r03a-263bee'),'dn':runs('i00-deltanet',['fresh'])}
FIN={'champ':'68.95','sig':'73.54','dn':'71.82'}
lab={'champ':'identity gate (champion)','sig':'sigmoid gate (old)','dn':'DeltaNet (expert)'}
fig,axs=plt.subplots(1,2,figsize=(10.5,4.0),gridspec_kw={'width_ratios':[1.25,1]})
ax=axs[0]
for k in ('dn','sig','champ'):
    A=np.stack(S[k]); ep=A[0,:,0]; v=A[:,:,2]; m=v.mean(0)
    ax.fill_between(ep,v.min(0),v.max(0),color=C[k],alpha=0.15,lw=0)
    ax.plot(ep,m,color=C[k],label=f"{lab[k]}  ·  {FIN[k]}")
ax.set_yscale('log'); ax.set_ylim(64,190); ax.set_yticks([70,80,100,120,160]); ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y,_:f'{y:.0f}'))
ax.set_xlabel('epoch'); ax.set_ylabel('validation perplexity (log)'); ax.set_title('Validation perplexity')
ax.axvline(2,color=MUTED,lw=0.8,ls=':'); ax.text(2.3,175,'search scores\nhere (2 epochs)',color=INK2,fontsize=8.5,va='top')
ax.legend(loc='upper right',fontsize=9)
ax=axs[1]
base=np.stack(S['dn'])[:,:,2].mean(0)
for k in ('sig','champ'):
    A=np.stack(S[k]); ep=A[0,:,0]; d=100*(A[:,:,2]/base-1)
    ax.fill_between(ep,d.min(0),d.max(0),color=C[k],alpha=0.15,lw=0); ax.plot(ep,d.mean(0),color=C[k])
    ax.text(30.4,d.mean(0)[-1],f'{d.mean(0)[-1]:+.1f}%',color=C[k],va='center',fontsize=9,fontweight='semibold')
ax.axhline(0,color=MUTED,lw=1); ax.text(1,0.4,'DeltaNet',color=INK2,fontsize=8.5)
ax.set_xlabel('epoch'); ax.set_ylabel('val PPL relative to DeltaNet (%)'); ax.set_title('Gap to the expert'); ax.set_xlim(0.5,33.5)
fig.text(0.01,-0.04,'WikiText-2, SLTT spiking neurons, seq 128; bands = range over 3 seeds. Old gate: gate = sigmoid(W·spike(h)); champion: gate = W·spike(h). Same parameter count.',fontsize=8,color=MUTED)
fig.suptitle('One line of code: tied when the search scores them (epoch 2), apart after 30 epochs',x=0.01,ha='left',fontsize=13,fontweight='semibold',color=INK,y=1.03)
save(fig,'F2_one_line')
