import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.grid":True,"grid.alpha":.25,"figure.dpi":200})
df=pd.read_csv("results.csv")
names=list(df.scenario.unique())
short={n:n.split(" ")[0] for n in names}
b=df[df.scenario==names[0]].set_index(["seed","t"])
col={"S0":"#444","S1":"#6baed6","S2":"#2171b5","S3":"#08306b","S4":"#e6550d","S5":"#8c6bb1","S6":"#31a354","S7":"#a1d99b"}
def band(ax,sc,y,rel=False,scale=1,label=None,ls="-"):
    g=df[df.scenario==sc].pivot(index="t",columns="seed",values=y)
    if rel:
        gb=b[y].unstack("seed"); g=g/gb-1
    g=g*scale
    m=g.mean(axis=1); lo=g.quantile(.1,axis=1); hi=g.quantile(.9,axis=1)
    c=col[short[sc]]
    ax.plot(m.index/12,m,color=c,label=label or short[sc],ls=ls,lw=1.6)
    ax.fill_between(m.index/12,lo,hi,color=c,alpha=.15,lw=0)
def fig(path,panels,ncol=2,size=(7.2,5.6)):
    nr=int(np.ceil(len(panels)/ncol)); f,axs=plt.subplots(nr,ncol,figsize=size); axs=np.atleast_1d(axs).ravel()
    for ax,(title,ylab,y,scs,rel,scale) in zip(axs,panels):
        for sc in scs: band(ax,sc,y,rel,scale)
        ax.axvline(1,color="k",lw=.6,ls=":"); ax.set_title(title,fontsize=9,loc="left"); ax.set_ylabel(ylab); ax.set_xlabel("Years")
    axs[0].legend(frameon=False,fontsize=7,ncol=2)
    for ax in axs[len(panels):]: ax.axis("off")
    f.tight_layout(); f.savefig(path); plt.close(f)
sev=names[:4]
fig("fig1_core.png",[("(a) Unemployment rate","% of workforce","unemp",sev,False,100),
 ("(b) Real output (GDP) vs baseline","% difference","gdp",sev,True,100),
 ("(c) Household consumption vs baseline","% difference","cons",sev,True,100),
 ("(d) Wage share of output","% of output","wage_share",sev,False,100)])
fig("fig2_distribution.png",[("(a) Gini of gross income","Gini","gini",sev,False,1),
 ("(b) Poverty headcount","% below subsistence","poverty",sev,False,100),
 ("(c) Financially fragile households","% with savings < 3 months","fragile",sev,False,100),
 ("(d) Mortgage defaults (cumulative)","% of mortgages","default_rate",sev,False,100)])
# savings by class, severe scenario vs baseline
f,axs=plt.subplots(1,2,figsize=(7.2,2.9))
for ax,sc,t in zip(axs,[names[0],names[2]],["Baseline","Severe (S2)"]):
    g=df[df.scenario==sc].groupby("t")
    for k,c in zip(["sav_Low","sav_LowerMid","sav_UpperMid","sav_High","sav_ITorigin"],["#d73027","#fc8d59","#91bfdb","#4575b4","#000"]):
        m=g[k].mean(); ax.plot(m.index/12,m/m.iloc[0],label=k.replace("sav_",""),color=c,lw=1.6 if k!="sav_ITorigin" else 2.2,ls="-" if k!="sav_ITorigin" else "--")
    ax.axvline(1,color="k",lw=.6,ls=":"); ax.set_title(t,loc="left",fontsize=9); ax.set_xlabel("Years"); ax.set_ylabel("Mean savings (index, t=0 → 1)")
axs[0].legend(frameon=False,fontsize=7); f.tight_layout(); f.savefig("fig3_savings.png"); plt.close(f)
fig("fig4_shops_banks.png",[("(a) Retail output vs initial","index","retail_out",sev,False,1),
 ("(b) Shops still trading","share of shops","shops_alive",sev,False,1),
 ("(c) Bank loss / capital","ratio","bank_loss_ratio",sev,False,1),
 ("(d) IT-sector employment","workers","emp_it",sev,False,1)])
cmp_=[names[2],names[4],names[5],names[6],names[7]]
fig("fig5_policy.png",[("(a) Unemployment","% of workforce","unemp",cmp_,False,100),
 ("(b) GDP vs baseline","% difference","gdp",cmp_,True,100),
 ("(c) Poverty headcount","%","poverty",cmp_,False,100),
 ("(d) Mortgage defaults","%","default_rate",cmp_,False,100)])
# tables
def last(g,col,t): return g.xs(t,level="t")[col]
rows=[]
for n in names:
    g=df[df.scenario==n].set_index(["seed","t"])
    bg=b
    cum=(g.gdp.groupby("seed").sum()-bg.gdp.groupby("seed").sum())/bg.gdp.groupby("seed").sum()
    d=lambda col,t:(last(g,col,t)-last(bg,col,t))
    r=dict(Scenario=n,
      peakU=g.unemp.groupby("seed").max().mean()*100,
      U_y5=last(g,"unemp",71).mean()*100, U_y10=last(g,"unemp",119).mean()*100,
      dU10=d("unemp",119).mean()*100, dU10_sd=d("unemp",119).std()*100,
      gdp5=(last(g,"gdp",71)/last(bg,"gdp",71)-1).mean()*100,
      gdp10=(last(g,"gdp",119)/last(bg,"gdp",119)-1).mean()*100,
      cum=cum.mean()*100, cum_sd=cum.std()*100,
      cons10=(last(g,"cons",119)/last(bg,"cons",119)-1).mean()*100,
      pov5=last(g,"poverty",71).mean()*100, pov0=last(bg,"poverty",71).mean()*100,
      frag5=last(g,"fragile",71).mean()*100,
      dflt=last(g,"default_rate",119).mean()*100,
      shops=g.shops_alive.groupby("seed").min().mean()*100,
      ws=last(g,"wage_share",119).mean()*100, gini=last(g,"gini",119).mean(),
      itsav=(last(g,"sav_ITorigin",71)/last(g,"sav_ITorigin",11)).mean(),
      fisc=((g.deficit.groupby("seed").sum()-bg.deficit.groupby("seed").sum())/ (bg.gdp.groupby("seed").sum()/10)).mean()*100,
      itU=last(g,"unemp_it",71).mean()*100)
    rows.append(r)
T=pd.DataFrame(rows); T.to_csv("summary.csv",index=False)
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
print(T.round(2).T)
