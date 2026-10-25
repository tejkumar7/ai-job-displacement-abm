# Builds sens_summary.csv and fig6_sensitivity.png from sens.csv (run run_sens.py first)
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
s=pd.read_csv('sens.csv')
t=s.groupby(['param','value'])[['peakU','endU','cumGDP','default','pov60']].mean()
t.to_csv('sens_summary.csv')
rows=[]
for p,g in s.groupby('param',sort=False):
    m=g.groupby('value').endU.mean()*100
    rows.append((p,m.min(),m.max()))
rows.sort(key=lambda r:r[2]-r[1])
f,ax=plt.subplots(figsize=(7.2,3.6))
mid=s[(s.param.str.startswith("recycle"))&(s.value==0.3)].endU.mean()*100
for i,(p,lo,hi) in enumerate(rows):
    ax.barh(i,hi-lo,left=lo,color="#2171b5",alpha=.85); ax.text(hi+.2,i,f"{lo:.1f}–{hi:.1f}",va="center",fontsize=7)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows],fontsize=7.5)
ax.axvline(mid,color="k",lw=.8,ls=":"); ax.set_xlabel("Unemployment rate in year 10 (%), severe scenario (S2)")
ax.spines[['top','right']].set_visible(False); f.tight_layout(); f.savefig("fig6_sensitivity.png",dpi=200)
