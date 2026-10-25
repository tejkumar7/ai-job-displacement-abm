AI-EconSim: agent-based model of AI-driven software job displacement
Requirements: python 3, numpy, pandas, matplotlib, joblib
Run order:
  python run_experiments.py   # 8 scenarios x 20 seeds -> results.csv (a few minutes)
  python run_sens.py          # one-at-a-time sensitivity -> sens.csv (10-15 minutes)
  python analyze.py           # main figures + summary.csv
  python analyze_sens.py      # sensitivity figure + table
Model: aiecon_abm.py (all parameters in the Params dataclass)
