"""
AI-EconSim: a stylised agent-based model of an economy in which AI displaces
software-industry labour.

Agents / components
  * Worker-households (N): sector, ability, wage, savings, mortgage EMI, MPC class
  * Capital owners: receive dividends, low marginal propensity to consume
  * Six sectors: IT, Professional services, Manufacturing, Retail (shops),
    Informal/low-skill services, Public sector
  * Retail shops modelled as individual firms (entry / exit)
  * Banks: mortgage defaults erode capital -> credit crunch
  * Government: progressive income tax, corporate tax, unemployment insurance,
    UBI, reskilling programme, deficit / debt tracking

Time step: one month.  All money in abstract currency units (CU) per month.
"""
from dataclasses import dataclass, field, replace
import numpy as np
import pandas as pd

SECT = ["IT", "Prof", "Manu", "Retail", "Informal", "Public"]
IT, PROF, MANU, RETAIL, INFORMAL, PUBLIC = range(6)


@dataclass
class Params:
    # --- population ---------------------------------------------------------
    n_workers: int = 10000
    owner_share: float = 0.015
    sector_share: tuple = (0.08, 0.12, 0.15, 0.25, 0.30, 0.10)
    base_unemp: float = 0.05
    base_wage: tuple = (110, 65, 38, 26, 16, 52)
    ability_sigma: float = 0.30
    savings_months_median: tuple = (10, 8, 5, 3, 1.5, 7)
    savings_sigma: float = 0.8
    owner_savings_median: float = 3000.0
    mpc_by_quartile: tuple = (0.95, 0.88, 0.78, 0.62)
    owner_mpc: float = 0.30
    mortgage_prob: tuple = (0.45, 0.35, 0.20, 0.10, 0.03, 0.30)
    emi_to_wage: float = 0.30
    loan_multiple: float = 60.0       # outstanding principal = 60 x EMI
    bank_capital_ratio: float = 0.12
    subsistence: float = 12.0

    # --- production / demand -------------------------------------------------
    ls_target: tuple = (0.55, 0.65, 0.60, 0.70, 0.90, 1.00)   # target labour share of sector revenue
    cons_share: tuple = (0.02, 0.10, 0.22, 0.36, 0.14, 0.0)    # household spending pattern (rest = imports)
    payout: float = 0.60
    import_share_invest: float = 0.30
    recycle: float = 0.30              # share of windfall (above-baseline) retained profit recycled into domestic investment demand
    transfer_scale: float = 1.0
    corp_tax: float = 0.20
    it_domestic_share: float = 0.50    # share of IT profit accruing to domestic owners / tax base (rest: foreign owners, transfer pricing)
    it_extra_corp_tax: float = 0.0     # windfall tax on IT profits (policy)

    # --- labour market -------------------------------------------------------
    churn: float = 0.015
    vacancy_slack: float = 0.0       # steady-state open-vacancy share in baseline
    fire_speed: float = 0.30
    hire_speed: float = 0.25
    search_prob: float = 0.70
    duration_decay: float = 0.02
    switch_penalty: float = 0.90
    wage_flex: float = 1.0
    transfer: tuple = (          # rows: from sector, cols: to sector (IT..Informal)
        (1.00, 0.50, 0.15, 0.10, 0.05),
        (0.30, 1.00, 0.30, 0.30, 0.10),
        (0.05, 0.20, 1.00, 0.50, 0.40),
        (0.02, 0.10, 0.30, 1.00, 0.70),
        (0.01, 0.03, 0.20, 0.60, 1.00),
        (0.02, 0.20, 0.20, 0.30, 0.30),
    )

    # --- households' behaviour -----------------------------------------------
    draw_rate: float = 0.12
    habit_adapt: float = 0.04
    default_after: int = 3
    default_cons_cut: float = 0.90

    # --- policy ---------------------------------------------------------------
    ui_rr: float = 0.40
    ui_dur: int = 6
    social_floor: float = 4.0
    ubi: float = 0.0
    reskill: bool = False
    reskill_cost: float = 15.0
    reskill_boost: float = 2.5

    # --- AI shock --------------------------------------------------------------
    shock_start: int = 12
    shock_len: int = 36
    it_displace: float = 0.0           # fraction of IT labour-per-output removed
    prof_spillover: float = 0.0        # prof-services displacement = spill x IT
    it_export_loss: float = 0.0        # fall in IT export revenue

    # --- shops -------------------------------------------------------------------
    n_shops: int = 800
    shop_theta: tuple = (0.55, 0.90)
    shop_cap_slack: float = 1.10
    shop_reopen: float = 0.02

    # --- banking -------------------------------------------------------------------
    cc_strength: float = 0.5
    bank_recap: float = 0.015          # monthly recapitalisation of bank losses

    horizon: int = 120


def income_tax(w):
    return (0.05 * np.clip(np.minimum(w, 50) - 20, 0, None)
            + 0.15 * np.clip(np.minimum(w, 100) - 50, 0, None)
            + 0.30 * np.clip(w - 100, 0, None))


def gini(x):
    x = np.sort(np.clip(x, 0, None))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


class Economy:
    def __init__(self, p: Params, seed: int = 0, ref_retail=None):
        self.p = p
        self.ref_retail = ref_retail
        self.rng = np.random.default_rng(seed)
        self._init_agents()
        self._calibrate()

    # ------------------------------------------------------------------ setup
    def _init_agents(self):
        p, rng = self.p, self.rng
        n = p.n_workers
        self.n = n
        self.n_own = max(1, int(round(n * p.owner_share)))
        self.sector = rng.choice(6, size=n, p=np.array(p.sector_share))
        self.ability = np.exp(rng.normal(0, p.ability_sigma, n))
        base = np.array(p.base_wage)
        self.wage = self.ability * base[self.sector]
        self.employed = np.ones(n, bool)
        unemp = (rng.random(n) < p.base_unemp) & (self.sector != PUBLIC)
        self.employed[unemp] = False
        self.last_sector = self.sector.copy()
        self.last_wage = self.wage.copy()
        self.wage = np.where(self.employed, self.wage, 0.0)
        self.months_unemp = np.where(self.employed, 0, rng.integers(1, 6, n))
        self.ui_left = np.where(self.employed, 0, rng.integers(0, p.ui_dur + 1, n))
        # savings
        med = np.array(p.savings_months_median)[self.sector] * self.last_wage
        self.savings = med * np.exp(rng.normal(0, p.savings_sigma, n))
        self.savings = np.where(self.employed, self.savings, self.savings * 0.5)
        self.owner_savings = p.owner_savings_median * np.exp(rng.normal(0, 0.6, self.n_own))
        # income classes by initial potential wage quartile
        q = np.quantile(self.last_wage, [0.25, 0.5, 0.75])
        self.quart = np.digitize(self.last_wage, q)           # 0..3
        self.mpc = np.array(p.mpc_by_quartile)[self.quart]
        # mortgages
        has = rng.random(n) < np.array(p.mortgage_prob)[self.sector]
        self.emi = np.where(has, p.emi_to_wage * self.last_wage, 0.0)
        self.has_loan = has.copy()
        self.missed = np.zeros(n, int)
        self.defaulted = np.zeros(n, bool)
        self.total_loans = float((self.emi * p.loan_multiple).sum())
        self.bank_capital = p.bank_capital_ratio * self.total_loans
        self.bank_loss = 0.0
        # shops
        w = np.exp(rng.normal(0, 0.8, p.n_shops))
        self.shop_w = w / w.sum()
        self.shop_alive = np.ones(p.n_shops, bool)
        self.shop_theta = rng.uniform(*p.shop_theta, p.n_shops)
        self.shop_bad = np.zeros(p.n_shops, int)
        # government
        self.debt = 0.0

    def _tax_income(self, income):
        return income_tax(income)

    def _calibrate(self):
        """Calibrate so that month 0 is an equilibrium of the circular flow:
        sector revenue = household demand + government + investment + exogenous
        (export) demand; exogenous demand is the residual that delivers the
        target labour share in IT, Prof and Manu."""
        p = self.p
        ls = np.array(p.ls_target)
        cs = np.array(p.cons_share)
        L0 = np.array([np.sum(self.employed & (self.sector == s)) for s in range(6)], float)
        wb0 = np.array([self.wage[self.employed & (self.sector == s)].sum() for s in range(6)])
        self.L0 = L0
        # households
        ben0 = self._benefit(initial=True)
        tax_w = self._tax_income(self.wage + ben0)
        disp = self.wage + ben0 - tax_w
        emi = self.emi
        net = disp - emi
        c = self.mpc * np.clip(net, 0, None)
        c = np.where(self.employed, c, np.minimum(c + p.draw_rate * self.savings, self._habit0(c)))
        c = np.minimum(c, self.savings + disp - emi)
        c = np.clip(c, 0, None)
        self.ref_c = c.copy()
        self.ref_net = net.copy()
        C0 = c.sum()
        Y0 = wb0 / np.where(ls > 0, ls, 1)
        for _ in range(80):
            profit = Y0 - wb0
            profit[PUBLIC] = 0.0
            profit_pos = np.clip(profit, 0, None)
            profit_pos[IT] *= p.it_domestic_share
            corp = p.corp_tax * profit_pos
            after = profit_pos - corp
            div_i = p.payout * after.sum() / self.n_own
            own_net = div_i - income_tax(div_i)
            C0 = c.sum() + p.owner_mpc * own_net * self.n_own
            rev0 = tax_w.sum() + income_tax(div_i) * self.n_own + corp.sum()
            G_other0 = max(rev0 - wb0[PUBLIC] - ben0.sum(), 0.0)
            G = np.zeros(6)
            G[MANU] = 0.40 * G_other0
            G[PROF] = 0.30 * G_other0
            G[RETAIL] = 0.20 * G_other0
            G[INFORMAL] = 0.10 * G_other0
            G[PUBLIC] = wb0[PUBLIC]
            retained = (1 - p.payout) * after.sum()
            inv = np.zeros(6)
            inv[MANU] = 0.6 * retained * (1 - p.import_share_invest)
            inv[PROF] = 0.4 * retained * (1 - p.import_share_invest)
            dem = cs * C0 + G + inv
            X = np.zeros(6)
            for s in range(5):
                X[s] = max(0.0, wb0[s] / ls[s] - dem[s])
            Y0 = dem + X
            Y0[PUBLIC] = wb0[PUBLIC]
        self.C0, self.G_other0, self.G, self.inv0, self.X = C0, G_other0, G, inv, X
        self.Y0 = Y0
        self.prod0 = Y0 / (L0 * (1 + np.where(np.arange(6) == PUBLIC, 0.0, p.vacancy_slack)))
        self.cs = cs
        self.import_leak = 1 - cs.sum()
        self.own_ref_c = p.owner_mpc * own_net
        self.gdp0 = Y0.sum()
        self.c_prev = C0
        self.cc = 0.0
        self.t = 0
        self.realised_ls = wb0 / Y0

    def _habit0(self, c):
        return np.maximum(c, self.p.subsistence)

    # -------------------------------------------------------------- utilities
    def _benefit(self, initial=False):
        p = self.p
        if initial:
            elig = (self.ui_left > 0)
            b = np.where(elig, p.ui_rr * self.last_wage, p.social_floor)
        else:
            elig = (self.ui_left > 0)
            b = np.where(elig, p.ui_rr * self.last_wage, p.social_floor)
        return np.where(self.employed, 0.0, b)

    def _d_it(self):
        p = self.p
        t = self.t - p.shock_start
        if t <= 0:
            return 0.0
        ramp = min(1.0, t / p.shock_len)
        return p.it_displace * ramp

    # ------------------------------------------------------------------ step
    def step(self):
        p, rng = self.p, self.rng
        t = self.t
        n = self.n
        sector, emp = self.sector, self.employed
        ramp = min(1.0, max(0.0, (t - p.shock_start) / p.shock_len)) if t > p.shock_start else 0.0

        # ---------- 1. sector demand and output (uses last month's spending)
        cs = self.cs
        D = cs * self.c_prev + self.G + self.inv_now if hasattr(self, "inv_now") else cs * self.c_prev + self.G + self.inv0
        X = self.X.copy()
        X[IT] *= (1 - p.it_export_loss * ramp)
        D = D + X
        # shops: capacity constraint
        alive_frac = float(np.sum(self.shop_w[self.shop_alive]))
        ref = self.Y0[RETAIL] if self.ref_retail is None else self.ref_retail[min(t, len(self.ref_retail) - 1)]
        ratio = D[RETAIL] / (ref * max(alive_frac, 1e-6))
        cap = p.shop_cap_slack * alive_frac * self.Y0[RETAIL]
        Y = D.copy()
        Y[RETAIL] = min(D[RETAIL], cap)
        # shop entry/exit
        bad = self.shop_alive & (ratio < self.shop_theta)
        self.shop_bad = np.where(bad, self.shop_bad + 1, 0)
        close = self.shop_alive & (self.shop_bad >= 3)
        self.shop_alive[close] = False
        if ratio > 0.95:
            dead = ~self.shop_alive
            reopen = dead & (rng.random(p.n_shops) < p.shop_reopen)
            self.shop_alive[reopen] = True
            self.shop_bad[reopen] = 0
        self.Y = Y

        # ---------- 2. labour demand
        d_it = self._d_it()
        d_pr = p.prof_spillover * d_it
        prod = self.prod0.copy()
        prod[IT] = self.prod0[IT] / (1 - d_it)
        prod[PROF] = self.prod0[PROF] / (1 - d_pr)
        Ltarget = Y / prod
        Ltarget[PUBLIC] = self.L0[PUBLIC]
        L = np.array([np.sum(emp & (sector == s)) for s in range(6)], float)

        # ---------- 3. separations: churn + layoffs
        idx_emp = np.where(emp & (sector != PUBLIC))[0]
        churn_mask = rng.random(len(idx_emp)) < p.churn
        sep = list(idx_emp[churn_mask])
        for s in range(5):
            gap = Ltarget[s] - (L[s] - np.sum(churn_mask & (sector[idx_emp] == s)))
            if gap < 0:
                cand = np.where(emp & (sector == s))[0]
                cand = np.setdiff1d(cand, np.array(sep, int))
                k = int(min(len(cand), np.ceil(p.fire_speed * (-gap))))
                if k > 0:
                    sep.extend(rng.choice(cand, k, replace=False))
        if sep:
            sep = np.array(sep, int)
            self.last_sector[sep] = sector[sep]
            self.last_wage[sep] = self.wage[sep]
            self.employed[sep] = False
            self.wage[sep] = 0.0
            self.months_unemp[sep] = 0
            self.ui_left[sep] = p.ui_dur
            self.n_sep = len(sep)
        emp = self.employed

        # ---------- 4. matching
        L = np.array([np.sum(emp & (sector == s)) for s in range(6)], float)
        T = np.array(p.transfer, float)
        for i in range(6):
            for j in range(5):
                if i != j:
                    T[i, j] = min(1.0, T[i, j] * p.transfer_scale)
        U = np.where(~emp)[0]
        searching = U[rng.random(len(U)) < p.search_prob]
        order = rng.permutation(5)
        urate_now = len(U) / n
        wflex = 1 - p.wage_flex * max(0.0, urate_now - p.base_unemp)
        wflex = max(wflex, 0.5)
        base = np.array(p.base_wage)
        taken = np.zeros(n, bool)
        for s in order:
            gap = Ltarget[s] - L[s]
            if gap <= 0:
                continue
            V = int(np.ceil(p.hire_speed * gap * (1 - self.cc)))
            cand = searching[~taken[searching]]
            if len(cand) == 0 or V <= 0:
                continue
            w = T[self.last_sector[cand], s] * np.exp(-p.duration_decay * self.months_unemp[cand])
            if p.reskill and s != IT:
                boost = (self.last_sector[cand] == IT) & (self.months_unemp[cand] <= 12) & (s in (PROF, MANU))
                w = np.where(boost, w * p.reskill_boost, w)
            apply_ = rng.random(len(cand)) < np.clip(w, 0, 1)
            cand = cand[apply_]
            if len(cand) == 0:
                continue
            k = min(V, len(cand))
            pick = rng.choice(cand, k, replace=False)
            taken[pick] = True
            same = self.last_sector[pick] == s
            pen = np.where(same, 1.0, p.switch_penalty)
            if p.reskill:
                pen = np.where((self.last_sector[pick] == IT) & (s in (PROF, MANU)), 1.0, pen)
            self.sector[pick] = s
            self.wage[pick] = self.ability[pick] * base[s] * pen * wflex
            self.employed[pick] = True
            self.months_unemp[pick] = 0
            self.ui_left[pick] = 0
            L[s] += k
        sector, emp = self.sector, self.employed
        self.months_unemp = np.where(emp, 0, self.months_unemp + 1)
        self.ui_left = np.where(emp, 0, np.clip(self.ui_left - 1, 0, None))

        # ---------- 5. profits, dividends, taxes, investment
        wb = np.array([self.wage[emp & (sector == s)].sum() for s in range(6)])
        profit = Y - wb
        profit[PUBLIC] = 0.0
        profit_pos = np.clip(profit, 0, None)
        profit_pos[IT] *= p.it_domestic_share
        ctax_rate = np.full(6, p.corp_tax)
        ctax_rate[IT] += p.it_extra_corp_tax
        corp = ctax_rate * profit_pos
        after = profit_pos - corp
        div_i = p.payout * after.sum() / self.n_own
        retained = (1 - p.payout) * after.sum()
        inv = np.zeros(6)
        inv[MANU] = 0.6 * retained * (1 - p.import_share_invest)
        inv[PROF] = 0.4 * retained * (1 - p.import_share_invest)
        self.inv_now = self.inv0 + p.recycle * (inv - self.inv0)

        # ---------- 6. household income, EMI, consumption
        ben = self._benefit()
        training = np.zeros(n, bool)
        if p.reskill:
            training = (~emp) & (self.last_sector == IT) & (self.months_unemp <= 6)
        ubi = p.ubi
        income = self.wage + ben + ubi
        tax = income_tax(self.wage + ben)
        disp = income - tax
        # EMI
        due = np.where(self.defaulted, 0.0, self.emi)
        avail = self.savings + disp
        can_pay = avail >= due
        pay = np.where(can_pay, due, 0.0)
        newmiss = (~can_pay) & (due > 0)
        self.missed = np.where(newmiss, self.missed + 1, 0)
        newdef = (self.missed >= p.default_after) & (~self.defaulted)
        if newdef.any():
            self.bank_loss += float((self.emi[newdef] * p.loan_multiple).sum())
            self.defaulted |= newdef
            self.ref_c[newdef] *= p.default_cons_cut
        net = disp - pay
        c_inc = self.mpc * np.clip(net, 0, None)
        c_draw = np.minimum(self.ref_c, c_inc + p.draw_rate * self.savings)
        c = np.where(net >= self.ref_net, c_inc, c_draw)
        c = np.minimum(c, np.clip(self.savings + disp - pay, 0, None))
        c = np.clip(c, 0, None)
        self.savings = self.savings + disp - pay - c
        self.ref_c = self.ref_c + p.habit_adapt * (c - self.ref_c)
        self.ref_net = self.ref_net + p.habit_adapt * (net - self.ref_net) * (net < self.ref_net)

        # owners
        own_net = div_i - income_tax(div_i)
        own_c = p.owner_mpc * own_net
        self.owner_savings += own_net - own_c
        C = c.sum() + own_c * self.n_own
        self.c_prev = C

        # ---------- 7. government & banks
        wage_tax = income_tax(self.wage + 0).sum()
        ben_tax = income_tax(self.wage + ben).sum() - wage_tax
        rev = tax.sum() + income_tax(div_i) * self.n_own + corp.sum()
        spend = (wb[PUBLIC] + self.G_other0 + ben.sum() + ubi * n
                 + (p.reskill_cost * training.sum() if p.reskill else 0.0))
        self.deficit = spend - rev
        self.debt += self.deficit
        self.bank_loss *= (1 - p.bank_recap)
        loss_ratio = self.bank_loss / max(self.bank_capital, 1e-9)
        self.cc = float(np.clip(p.cc_strength * loss_ratio, 0, 0.5))

        # ---------- 8. record
        workers = np.ones(n, bool)
        gross_inc = np.concatenate([self.wage + ben + ubi, np.full(self.n_own, div_i)])
        rec = dict(
            t=t,
            gdp=Y.sum(),
            cons=C,
            unemp=float(np.mean(~emp)),
            unemp_it=float(np.mean(~emp[self.last_sector == IT])) if np.any(self.last_sector == IT) else 0.0,
            emp_it=float(np.sum(emp & (sector == IT))),
            it_wage_bill=wb[IT],
            wage_share=float(wb.sum() / Y.sum()),
            profit_it=profit[IT],
            poverty=float(np.mean(c < p.subsistence - 1e-9)),
            fragile=float(np.mean(self.savings < 3 * self.ref_c)),
            default_rate=float(self.defaulted.sum() / max(self.has_loan.sum(), 1)),
            bank_loss_ratio=loss_ratio,
            shops_alive=float(self.shop_alive.mean()),
            retail_out=Y[RETAIL] / self.Y0[RETAIL],
            ret_dem=D[RETAIL],
            deficit=self.deficit,
            debt=self.debt,
            revenue=rev,
            gini=gini(gross_inc),
            sav_owner=float(self.owner_savings.mean()),
        )
        for qi, nm in enumerate(["Low", "LowerMid", "UpperMid", "High"]):
            rec[f"sav_{nm}"] = float(self.savings[self.quart == qi].mean())
        for s in range(6):
            rec[f"u_{SECT[s]}"] = float(np.mean(~emp[self.last_sector == s])) if np.any(self.last_sector == s) else 0.0
            rec[f"L_{SECT[s]}"] = float(np.sum(emp & (sector == s)))
        it_origin = self.last_sector == IT
        # use original IT workers: those whose *initial* sector was IT
        rec["sav_ITorigin"] = float(self.savings[self.init_it].mean()) if hasattr(self, "init_it") else np.nan
        self.t += 1
        return rec

    def run(self):
        self.init_it = (self.sector == IT).copy()
        # re-compute initial sector flag from first state (before dynamics)
        out = []
        for _ in range(self.p.horizon):
            out.append(self.step())
        df = pd.DataFrame(out)
        df["gdp_idx"] = df.gdp / self.gdp0
        return df


def run_scenario(params: Params, seed: int, ref_retail=None):
    return Economy(params, seed, ref_retail).run()
