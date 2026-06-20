# 🪿 GEESE

An interactive demo for behavioral phenotyping of rodent models from 3D pose
recordings. Given a recording, it predicts behavior over time, cohort, and
genotype, and lets you query the results through a chat assistant.

**🪿📣 GEESE-HONK (interactive live demo):** https://huggingface.co/spaces/EvilRagdollCat/GEESE-HONK

## Usage

Open the live demo, then either upload a `.mat` recording or pick a built-in
example, and click **Analyze**. The app returns:

- a behavior distribution and a behavior timeline,
- cohort and genotype predictions (with probabilities),
- a downloadable Excel summary,
- a chat box for asking questions about the results.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```



## Data

Example recordings come from the socialDANNCE dataset
([Harvard Dataverse](https://dataverse.harvard.edu/dataverse/socialDANNCE_data), CC0).
