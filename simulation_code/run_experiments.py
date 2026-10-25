import numpy as np, pandas as pd, json, os
from dataclasses import replace
from aiecon_abm import *
from joblib import Parallel, delayed

NSEED = 20
base = Params()
policy = dict(ui_rr=0.60, ui_dur=18, reskill=True, it_extra_corp_tax=0.15)
SC = {
 "S0 Baseline (no AI shock)": dict(),
 "S1 Moderate (30% IT labour displaced)": dict(it_displace=0.30),
 "S2 Severe (60%)": dict(it_displace=0.60),
 "S3 Extreme (90%)": dict(it_displace=0.90),
 "S4 Severe + white-collar spillover": dict(it_displace=0.60, prof_spillover=0.40),
 "S5 Severe + export loss (20%)": dict(it_displace=0.60, it_export_loss=0.20),
 "S6 Severe + policy package": dict(it_displace=0.60, **policy),
 "S7 Severe + spillover + policy": dict(it_displace=0.60, prof_spillover=0.40, **policy),
}
def one(seed):
    out = []
    b = None
    for name, kw in SC.items():
        df = run_scenario(replace(base, **kw), seed, None if b is None else b.ret_dem.values)
        if b is None: b = df
        df["scenario"] = name; df["seed"] = seed
        out.append(df)
    return out
if __name__ == "__main__":
    res = [x for s in range(NSEED) for x in one(s)]
    df = pd.concat(res, ignore_index=True)
    df.to_parquet("results.parquet") if False else df.to_csv("results.csv", index=False)
    print(df.shape)
