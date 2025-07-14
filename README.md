# FasText-ChemT5
This project tackles the Molecule Captioning task, using a T5 model as a backbone.
Though the original T5 model (<a href="https://arxiv.org/abs/1910.10683"> <img src="https://img.shields.io/badge/arxiv-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="Repo"> </a>) was initially designed to handle transfer learning and multiple downstream tasks, we use Text+ChemT5 (<a href="https://arxiv.org/abs/2301.12586"> <img src="https://img.shields.io/badge/arxiv-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="Repo"> </a>) as our backbone, and only focus on ONE task: Molecule Captioning (or "mol2text").

<p align="center"><img src="./assets/tasks.png" width=80%></p>
<p align="center"><em>Figure 1.</em> Original Text+ChemT5 downstream tasks.</p>

---
Our main objective is to make molecule captioning **efficient and accurate**. We proceed in two steps described below:
<p align="center"><img src="./assets/goals.png" width=80%></p>
<p align="center"><em>Figure 2.</em> Goals of this project.</p>

For the first step, we use the L+M-24 dataset (<a href="https://arxiv.org/abs/2403.00791"> <img src="https://img.shields.io/badge/arxiv-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="Repo"> </a>) to fine-tune our models.

For the second step, "Speculative Decoding" is a whole topic by itself... Conceptually, the idea is to spend more resources on decoding "harder" tokens.
In this project, we adapt the implementation of Romain Storaï (<a href="https://github.com/romsto/Speculative-Decoding"> <img src="https://img.shields.io/badge/Repo-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="Repo"> </a>)!

---

You can finetune your own models (no need to explain how, it's really just downloading the original dataset and fine-tuning the T5 model from the transformers library) but we provide checkpoints just for evaluation reproducibility.

You can then have fun playing with the Speculative Decoding (by installing the requirements and running `Speculative\ Decoding/infer.py` in the CLI).

# Results

The first objective, being improving overall captioning quality, is fullfilled:
<p align="center"><img src="./assets/results_performance.png" width=80%></p>
<p align="center"><em>Figure 3.</em> Results for performance metrics after fine-tuning the Small and Base models.</p>

For the second objective, inference speed, it's a bit hard to cover it all so here are three slides to explain what Speculative Decoding is, examples in our project and key results:
<p align="center"><img src="./assets/results_speed1.png" width=80%></p>
<p align="center"><img src="./assets/results_speed2.png" width=80%></p>
<p align="center"><img src="./assets/results_speed3.png" width=80%></p>
(For reference, the figure examplifying Speculative Decoding comes from the original manuscript:<a href="https://arxiv.org/abs/2211.17192"> <img src="https://img.shields.io/badge/arxiv-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="arxiv"> </a> ).