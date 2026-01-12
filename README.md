<div align="center">

# <img src="images/AutoVLA-Logo.png" width="35" height="35" style="vertical-align: middle; margin-right: 10px;"> AutoVLA 
<!-- <br> <span style="font-size: 0.5em; font-weight: normal;">A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning</span> -->

[![website](https://img.shields.io/badge/Website-Explore%20Now-blueviolet?style=flat&logo=google-chrome)](https://autovla.github.io/)
[![paper](https://img.shields.io/badge/arXiv-2506.13757-B31B1B.svg?style=flat&logo=arxiv)](https://arxiv.org/abs/2506.13757)
[![dataset](https://img.shields.io/badge/Dataset-Coming_Soon-gre.svg?style=flat&logo=huggingface)]()
[![License](https://img.shields.io/badge/License-Academic_Software_License-blue)]()

</div>

[NeurIPS 2025] This is the official implementation of the paper:

**AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning**

[Zewei Zhou](https://zewei-zhou.github.io/)\*</sup>, [Tianhui Cai](https://www.tianhui-vicky.com/)\*</sup>, [Seth Z. Zhao](https://sethzhao506.github.io/), [Yun Zhang](https://handsomeyun.github.io/), [Zhiyu Huang](https://mczhi.github.io/)<sup>†</sup>, [Bolei Zhou](https://boleizhou.github.io/), [Jiaqi Ma](https://mobility-lab.seas.ucla.edu/about/)

University of California, Los Angeles | <sup>*</sup> Equal contribution, <sup>†</sup> Project leader

![teaser](images/AutoVLA_framework.png)


- 🚗 AutoVLA integrates **chain-of-thought (CoT) reasoning** and **physical action tokenization** to directly generate planning trajectories through a unified autoregressive process, dynamically switching dual-thinking modes.
- ⚙️ **Supervised fine-tuning (SFT)** is employed to equip the model with dual thinking modes: fast thinking (trajectory-only) and slow thinking (enhanced with chain-of-thought reasoning). 
- 🪜 **Reinforcement fine-tuning (RFT)** based on Group Relative Policy Optimization (GRPO) is adopted to further enhance planning performance and efficiency, reducing unnecessary reasoning in straightforward scenarios.
- 🔥 Extensive experiments across real-world and simulated datasets and benchmarks, including **nuPlan**, **nuScenes**, **Waymo**, and **CARLA**, demonstrate its competitive performance in both open-loop and closed-loop settings. 

## News
- **`2026/01`**: AutoVLA codebase is now released.
- **`2025/09`**: [AutoVLA](https://arxiv.org/abs/2506.13757) is accepted by [NeurIPS 2025](https://neurips.cc/) 👏👏.
- **`2025/06`**: [AutoVLA](https://arxiv.org/abs/2506.13757) paper release.
- **`2025/05`**: In the [Waymo Vision-based End-to-end Driving Challenge](https://waymo.com/open/challenges/2025/e2e-driving/), AutoVLA ranks highly in both RFS Overall and achieves the top RFS Spotlight score, which focuses on the most challenging scenarios.

<!-- ## Overview
- [Release Plan](#release-plan)
- [Dataset](#dataset)
- [Citation](#citation) -->

## Release Plan
- **`2025/06`**: ✅ AutoVLA paper.
- **`2025/12`**: ✅ Reasoning annotation code.
- **`2026/01`**: ✅ AutoVLA SFT code.
- **`2026/01`**: ✅ AutoVLA RFT code.
- **`2026/02`**: AutoVLA checkpoints.
- **`TBD`** : Reasoning data (Pending approval from the data provider).

## Devkit Setup
### 1. Dataset Downloading
#### nuPlan Dataset
You can refer to [here](https://github.com/autonomousvision/navsim/blob/main/docs/install.md) to prepare the nuPlan dataset. Be careful with the dataset structure.
```bash
bash navsim/download/download_maps.sh
bash navsim/download/download_trainval.sh
bash navsim/download/download_test.sh
```
#### Waymo E2E Dataset
The waymo end-to-end driving dataset can be downloaded at [here](https://waymo.com/open/download/). 

#### nuScenes Dataset
<span style="color:red">[TBD]</span>

### 2. Conda Environment Setup
You can perform the following command to create a conda environment and install the required dependencies.
```bash
conda env create -f environment.yml
conda activate autovla
pip install -e . --no-warn-conflicts
bash install.sh
```

### 3. Navsim Setup
We have included the navsim code in this repo, and you can go to the `navsim` folder to install it. You can also refer to [here](https://github.com/autonomousvision/navsim/blob/v2.0/docs/install.md) to set up the navsim devkit, but please ensure version compatibility for the dependencies.
```bash
cd navsim
pip install -e . --no-warn-conflicts
```
Remember to set the navsim required environment variables:
```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
```
### 4. Pretrained Model Downloading
We use the `Qwen2.5-VL` model series as the pretrained LLM in the vision-language-action model and chain-of-thought (CoT) annotation model. You can run the command to download the pretrained model.
```bash
bash scripts/download_qwen.sh
```
Specifically, we use the 72B model in CoT annotation, and you can choose `Qwen2.5-VL-72B-Instruct` or `Qwen2.5-VL-72B-Instruct-AWQ` based on your device. We use the `Qwen2.5-VL-3B-Instruct` in the AutoVLA model.

<!-- Retrieve `Qwen2.5-VL-3B-Instruct` from Hugging Face. -->


## Getting Started
### 1. Dataset Preprocessing
#### nuPlan Dataset
You can perform the command to preprocess the nuPlan dataset. Please first revise your path and data split (refer to [here](https://github.com/autonomousvision/navsim/blob/v2.0/docs/splits.md)) in the config. The `INCLUDE_COT` setting in the bash determines whether to launch the CoT reasoning annotation.
```bash
bash scripts/run_nuplan_preprocessing.sh
```
#### Waymo E2E Dataset
To organize the image data and support random access, we first cache the image data in the same format as the other dataset we used.
```bash
bash scripts/run_waymo_e2e_image_extraction.sh
```
You can perform the following command to preprocess the Waymo E2E dataset. Please also first revise your path and data split in the config and set the `INCLUDE_COT`.
```bash
bash scripts/run_waymo_e2e_preprocessing.sh
```
You can use `waymo_e2e_traj_project_visualization.py` and `waymo_e2e_visualization.py` in the `tools/visualization` folder to visualize the waymo data after preprocessing.
#### nuScenes Dataset
<span style="color:red">[TBD]</span>

### 2. Action Codebook Creation
<span style="color:red">[TBD]</span>

### 3. Supervised Fine-tuning (SFT)
First revise the dataset path and SFT parameters in the config file in `config/training`. Then, launch the sft training with the command.
```bash
bash scripts/run_sft.sh
```

### 4. Reinforcement Fine-tuning (RFT)
We introduce a reinforcement fine-tuning method based on Group Relative Policy Optimization (GRPO), reducing unnecessary reasoning in straightforward scenarios. 

You can revise your dataset path and GRPO parameters in the config file in `config/training`. Then, perform the following command to run the reinforcement finetuning.
```bash
bash scripts/run_rft.sh
```

### 5. Navsim Testing
We leverage navsim and its Predictive Driver Model Score (PDMS) to test and evaluate our model. You need to set up the dataset path and split in the evaluation bash, and run the command to launch the testing.
```bash
bash navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh
```


## Citation
If you find this repository useful for your research, please consider giving us a star 🌟 and citing our paper.
 ```bibtex
@article{zhou2025autovla,
  title={AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning},
  author={Zhou, Zewei and Cai, Tianhui and Zhao, Seth Z.and Zhang, Yun and Huang, Zhiyu and Zhou, Bolei and Ma, Jiaqi},
  journal={arXiv preprint arXiv:2506.13757},
  year={2025}
}