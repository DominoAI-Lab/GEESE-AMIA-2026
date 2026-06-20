<h1 align="center">🪿 GEESE</h1>

<p align="center">
  <b>Genotype-aware End-to-End Spatio-temporal Embedding for Behavioral Phenotyping</b>
</p>

<p align="center">
  Behavioral phenotyping of rodent models from 3D pose recordings that
  predicts behavior over time, cohort, and genotype, with a chat assistant
  for querying the results.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.24370">
    <img src="https://img.shields.io/badge/📄%20arXiv-2605.24370-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"/>
  </a>
  &nbsp;
  <a href="https://huggingface.co/spaces/EvilRagdollCat/GEESE-HONK">
    <img src="https://img.shields.io/badge/🪿📣%20GEESE--HONK-Live%20Demo-FFD21E?style=for-the-badge" alt="Live Demo"/>
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Presented%20at-AMIA%202026-2E7D32?style=for-the-badge" alt="AMIA 2026"/>
</p>

<p align="center">
  <b>Yiran Ding</b><sup>1</sup> &nbsp;&nbsp;
  <b>Yuen Gao</b><sup>2</sup> &nbsp;&nbsp;
  <b>Chunqi Qian</b><sup>2</sup> &nbsp;&nbsp;
  <b>Zijun Cui</b><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Department of Computer Science and Engineering, Michigan State University &nbsp;&nbsp;
  <sup>2</sup>Department of Radiology, Michigan State University
</p>

---

## Usage

Open the [live demo](https://huggingface.co/spaces/EvilRagdollCat/GEESE-HONK),
then either upload a `.mat` recording or pick a built-in example, and click
**Analyze**. The app returns:

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

## Acknowledgements

This work builds on [MOMENT](https://github.com/moment-timeseries-foundation-model/moment),
a time-series foundation model.

## Citation

```bibtex
@misc{ding2026geesegenotypeawareendtoendspatiotemporal,
      title={GEESE: Genotype-aware End-to-End Spatio-temporal Embedding for Behavioral Phenotyping}, 
      author={Yiran Ding and Yuen Gao and Chunqi Qian and Zijun Cui},
      year={2026},
      eprint={2605.24370},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.24370}, 
}
```
