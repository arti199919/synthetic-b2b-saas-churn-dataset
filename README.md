# synthetic-b2b-saas-churn-dataset
Links

- **Free sample on HuggingFace**: <PAShttps://huggingface.co/datasets/arti199919/synthetic-saas-churn-sampleTE_HF_URL>
- **Kaggle dataset (sample)**: <https://www.kaggle.com/datasets/arti199919/synthetic-saas-churn-sample>
- **Kaggle notebook (baseline)**: <https://www.kaggle.com/code/arti199919/synthetic-b2b-saas-churn-revenue-sample>
- **Paid full dataset on Gumroad (100k users + baseline)**: <https://gorchakov.gumroad.com/l/zhawn?utm_source=github&utm_medium=readme&utm_campaign=sample>

Proof (baseline)

The paid package includes `BASELINE.txt` with an example result:
- XGBoost next-month churn baseline: **AUC ≈ 0.81

FAQ

**Is this real customer data?**  
No. It’s **100% synthetic**. See `meta.json` fields: `data_origin`, `real_data_used=false`, `gdpr_risk=none`.

**What tasks does it support?**  
Churn prediction, LTV / revenue modeling, segmentation, cohort analysis.

**What’s the recommended ML label?**  
Use `user_monthly` features at month *t* and label `churned_next_month`.



