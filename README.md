# <img src="images/AutoVLA-Logo.png" width="35" height="35" style="vertical-align: bottom; margin-right: 10px;"> AutoVLA

[![website](https://img.shields.io/badge/Website-Explore%20Now-blueviolet?style=flat&logo=google-chrome)](https://autovla.github.io/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2506.13757)
[![dataset](https://img.shields.io/badge/Dataset-Download-F9D371.svg)]()
<!-- [![License](https://img.shields.io/badge/License-CC_BY_NC_ND_4.0-F9D371)](https://creativecommons.org/licenses/by-nc-nd/4.0/) -->

<!-- This is the official implementation for the following paper: -->

[NeurIPS 2025] This is the official implementation of the paper:

**AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning**

[Zewei Zhou](https://zewei-zhou.github.io/)\*</sup>, [Tianhui Cai](https://www.tianhui-vicky.com/)\*</sup>, [Seth Z. Zhao](https://sethzhao506.github.io/), [Yun Zhang](https://handsomeyun.github.io/), [Zhiyu Huang](https://mczhi.github.io/)<sup>†</sup>, [Bolei Zhou](https://boleizhou.github.io/), [Jiaqi Ma](https://mobility-lab.seas.ucla.edu/about/)

University of California, Los Angeles - <sup>*</sup> Equal contribution, <sup>†</sup> Project leader

![teaser](/images/AutoVLA_framework.png)


- 🚗 AutoVLA integrates **chain-of-thought (CoT) reasoning** and **physical action tokenization** to directly generate planning trajectories through a unified autoregressive process, dynamically switching dual-thinking modes.
- ⚙️ **Supervised fine-tuning (SFT)** is employed to equip the model with dual thinking modes: fast thinking (trajectory-only) and slow thinking (enhanced with chain-of-thought reasoning). 
- 🪜 **Reinforcement fine-tuning (RFT)** based on Group Relative Policy Optimization (GRPO) is adopted to further enhance planning performance and efficiency, reducing unnecessary reasoning in straightforward scenarios.
- 🔥 Extensive experiments across real-world and simulated datasets and benchmarks, including **nuPlan**, **nuScenes**, **Waymo**, and **CARLA**, demonstrate its competitive performance in both open-loop and closed-loop settings. 

<!-- Supported by the [UCLA Mobility Lab](https://mobility-lab.seas.ucla.edu/) -->

<!-- ## Overview
- [Codebase Features](#codebase-features)
- [Release Plan](#release-plan)
- [Dataset](#dataset)
- [Acknowledgment](#acknowledgement)
- [Citation](#citation) -->

## News
- **`2025/09`**: [AutoVLA](https://arxiv.org/abs/2506.13757) is accepted by [NeurIPS 2025](https://neurips.cc/) 👏👏.
- **`2025/06`**: [AutoVLA](https://arxiv.org/abs/2506.13757) paper release.
- **`2025/05`**: In the [Waymo Vision-based End-to-end Driving Challenge](https://waymo.com/open/challenges/2025/e2e-driving/), AutoVLA ranks highly in both RFS Overall and achieves the top RFS Spotlight score, which focuses on the most challenging scenarios.

## Release Plan
- **`2025/06`**: ✅ AutoVLA paper.
- **`2025/11`**: Reasoning data.
- **`2025/11`**: Reasoning annotation code.
- **`2025/11`**: AutoVLA code.
- **`2025/11`**: AutoVLA checkpoints.

## Citation
If you find this repository useful for your research, please consider giving us a star 🌟 and citing our paper.
 ```bibtex
@article{zhou2025autovla,
  title={AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning},
  author={Zhou, Zewei and Cai, Tianhui and Zhao, Seth Z.and Zhang, Yun and Huang, Zhiyu and Zhou, Bolei and Ma, Jiaqi},
  journal={arXiv preprint arXiv:2506.13757},
  year={2025}
}