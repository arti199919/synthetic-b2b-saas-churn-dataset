 Synthetic B2B SaaS Churn & Revenue Dataset

## Start here (free sample)
- HuggingFace dataset (sample): https://huggingface.co/datasets/arti199919/synthetic-saas-churn-sample

## Proof / baseline
- Kaggle notebook (baseline): https://www.kaggle.com/code/arti199919/synthetic-b2b-saas-churn-revenue-sample
- Kaggle dataset (sample): https://www.kaggle.com/datasets/arti199919/synthetic-saas-churn-sample

## Paid full dataset (100k users + baseline)
- Gumroad: https://gorchakov.gumroad.com/l/zhawn?utm_source=github&utm_medium=readme&utm_campaign=landing

## Quick baseline run (local)
python3 -m pip install -r requirements-ml.txt
python3 churn_xgboost_baseline.py --data-dir . --format parquet --split user --test-size 0.2## Legal / privacy
100% synthetic. No real customer data used. GDPR risk: none (see `meta.json` in dataset packages).
