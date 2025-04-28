<div id="top"></div>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/spritz-group/DUMBer">
    <img src="assets/logo.png" alt="Logo" width="150" height="150">
  </a>

  <h1 align="center">DUMB and DUMBer</h1>

  <p align="center">Is Adversarial Training Worth It in the Real World?
    <br />
    <a href="https://github.com/spritz-group/DUMBer"><strong>Paper in progress »</strong></a>
    <br />
    <br />
    <a href="https://www.math.unipd.it/~fmarchio/">Francesco Marchiori</a>
    ·
    <a href="https://github.com/MarcoAlecci">Marco Alecci</a>
    ·
    <a href="https://sites.google.com/view/lucapajola/home">Luca Pajola</a>
    ·
    <a href="https://www.math.unipd.it/~conti/">Mauro Conti</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary><strong>Table of Contents</strong></summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
    </li>
  </ol>
</details>


<div id="abstract"></div>

## 🧩 Abstract

>Adversarial examples are small and often imperceptible perturbations crafted to fool machine learning models. These attacks seriously threaten the reliability of deep neural networks, especially in security-sensitive domains. Evasion attacks, a form of adversarial attack where input is modified at test time to cause misclassification, are particularly insidious due to their transferability: adversarial examples crafted against one model often fool other models as well. This property, known as adversarial transferability, complicates defense strategies since it enables black-box attacks to succeed without direct access to the victim model. While adversarial training is one of the most widely adopted defense mechanisms, its effectiveness is typically evaluated on a narrow and homogeneous population of models. This limitation hinders the generalizability of empirical findings and restricts practical adoption. In this work, we introduce **DUMBer**, an attack framework built on the foundation of the DUMB (Dataset soUrces, Model architecture, and Balance) methodology, to systematically evaluate the resilience of adversarially trained models. Our testbed spans multiple adversarial training techniques evaluated across three diverse computer vision tasks, using a heterogeneous population of uniquely trained models to reflect real-world deployment variability. Our experimental pipeline comprises over 130k evaluations spanning 13 state-of-the-art attack algorithms, allowing us to capture nuanced behaviors of adversarial training under varying threat models and dataset conditions. Our findings offer practical, actionable insights for AI practitioners, identifying which defenses are most effective based on the model, dataset, and attacker setup.

<p align="right"><a href="#top">(back to top)</a></p>
<div id="usage"></div>

## ⚙️ Usage

First, start by cloning the repository.

```bash
git clone https://github.com/spritz-group/DUMBer.git
cd DUMBer
```

Then, install the required Python packages by running:

```bash
pip install -r requirements.txt
```

You now need to add the datasets in the repository. You can do this by downloading the zip file [here](https://forms.gle/fZtKiysf4FVdU3ui6) and extracting it in this repository.

To replicate the results in our paper, you need to execute the scripts in a specific order (`modelTrainer.py`, `attackGenerationVal.py`, `adversarialTraining.py`, `attackGenerationTest.py` and `evaluation.py`), or you can execute them one after another by running the dedicated shell script.

```bash
chmod +x ./run.sh && ./run.sh
```

If instead you want to run each script one by one, you will need to specify the task through an environment variable.

* **TASK=0**: `[bike, motorbike]`
* **TASK=1**: `[cat, dog]`
* **TASK=2**: `[man, woman]`

```bash
export TASK=0 && python3 modelTrainer.py
```

<p align="right"><a href="#top">(back to top)</a></p>
