## GitHub README blocks (copy/paste)

### Add near the top of your GitHub `README.md`

```markdown
## Links

- **Free sample on HuggingFace**: <PASTE_HF_URL>
- **Kaggle dataset (sample)**: <PASTE_KAGGLE_DATASET_URL>
- **Kaggle notebook (baseline)**: <PASTE_KAGGLE_NOTEBOOK_URL>
- **Paid full dataset on Gumroad (100k users + baseline)**: https://gorchakov.gumroad.com/l/zhawn?utm_source=github&utm_medium=readme&utm_campaign=sample

## Proof (baseline)

The paid package includes `BASELINE.txt` with an example result:
- XGBoost next-month churn baseline: **AUC ≈ 0.81**
```

### Optional FAQ (good for SEO)

```markdown
## FAQ

**Is this real customer data?**  
No. It’s **100% synthetic**. See `meta.json` fields: `data_origin`, `real_data_used=false`, `gdpr_risk=none`.

**What tasks does it support?**  
Churn prediction, LTV / revenue modeling, segmentation, cohort analysis.

**What’s the recommended ML label?**  
Use `user_monthly` features at month *t* and label `churned_next_month`.
```


