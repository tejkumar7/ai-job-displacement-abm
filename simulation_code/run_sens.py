import numpy as np, pandas as pd
from dataclasses import replace
from joblib import Parallel, delayed
from aiecon_abm import *
NS=12
base=Params(it_displace=0.60)
grid={
 "recycle (windfall → domestic demand)":("recycle",[0.0,0.3,0.6,1.0]),
 "owner MPC":("owner_mpc",[0.15,0.30,0.60]),
 "skill transferability ×":("transfer_scale",[0.5,1.0,1.5]),
 "hiring speed":("hire_speed",[0.15,0.25,0.40]),
 "displacement speed (months)":("shock_len",[12,36,60]),
 "domestic share of IT profit":("it_domestic_share",[0.25,0.5,1.0]),
 "credit-crunch strength":("cc_strength",[0.0,0.5,1.0]),
 "mortgage EMI/wage":("emi_to_wage",[0.20,0.30,0.40]),
 "unemployment insurance replacement":("ui_rr",[0.0,0.4,0.8]),
}
def run(label,key,val,seed):
    b=run_scenario(replace(base,it_displace=0.0,**{key:val}),seed)
    d=run_scenario(replace(base,**{key:val}),seed,b.ret_dem.values)
    return dict(param=label,value=val,seed=seed,
        peakU=d.unemp.max(), endU=d.unemp.iloc[-1],
        cumGDP=(d.gdp.sum()-b.gdp.sum())/b.gdp.sum(),
        default=d.default_rate.iloc[-1], pov60=d.poverty.iloc[60])
jobs=[(l,k,v,s) for l,(k,vs) in grid.items() for v in vs for s in range(NS)]
res=Parallel(n_jobs=1)(delayed(run)(*j) for j in jobs)
pd.DataFrame(res).to_csv("sens.csv",index=False)
