<h1 align="center"> Beyond Green: Annotation-Free Robustness Metrics for Non-Green Ornamental Plant Detection </h1>

<div align="center">

**Domain-Aware Self-Validated Instance Counting for *Loropetalum* and Non-Green Ornamental Species**

[XXXX-1](XXXX-6)<sup>1</sup>,
[XXXX-2](https://scholar.google.com/citations?user=YOUR_ID)<sup>1†</sup>,
[XXXX-3](https://scholar.google.com/citations?user=BQJE_UIAAAAJ)<sup>2</sup>

<sup>1</sup> XXXX-4, <sup>2</sup> XXXX-5         
(†) Corresponding author.

</div>

<div align="center">

<a href="https://openreview.net/forum?id=SD6FZaEJAH"><img src="https://img.shields.io/badge/OpenReview-CVPR_2026-b31b1b" alt='openreview'></a>
<a href="XXXX-7"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Demo-Coming_Soon-F0CD4B?labelColor=666EEE" alt='HuggingFace Space'></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt='license'></a>
<a href="XXXX-6"><img src="https://img.shields.io/badge/GitHub-Code-blue" alt='code'></a>
</div>

<div align="center">

<!-- Animated GIF showing detection pipeline -->
<img src="assets/demo_detection.gif" width="800" alt="Plant Detection Pipeline Demo">

<br>

<!-- Video player with fallback -->
<details>
<summary>🎬 <b>Watch High-Quality Video Version (MP4)</b></summary>

<br>

https://anonymous.4open.science/r/Beyond-Green-Loropetalum-and-Non-Green-Ornamental-Species-Annotation-Free-Robustness-Metrics-B2D2/assets/demo_detection.mp4

<br>

<em>Full resolution detection pipeline with better quality</em>

</details>

</div>

## Demo Package

A high-resolution storyboard demo has been generated in:

- `results/cvpr_demo/cvpr_demo.mp4`
- `results/cvpr_demo/index.html`
- `results/cvpr_demo/storyboard.md`

This version follows the same presentation standard used in the `Quantum-Feature-Selection` CVPR demo workflow: `2560x1440`, `30 fps`, white-background storyboard slides, and publication-style captions.


## 📣 News

- **[Nov/2025]** 🎉 Paper submitted to CVPR 2026!
- **[Dec/2025]** 🚀 Code and evaluation framework released
- **[Dec/2025]** 🌿 Cross-dataset evaluation on 3,024 images (MangoNet, Leafy Spurge, PlantNet Purple)

## Abstract

Evaluating computer vision models for agricultural deployment remains challenging when ground-truth annotations are expensive or unavailable. Traditional accuracy-centric metrics often fail to reveal a model's reliability when confronted with dense occlusion, non-standard colouration, and variable lighting conditions. To address this, we introduce a novel **annotation-free evaluation framework** that measures model robustness through four complementary consistency metrics, eliminating the need for manual labels while predicting real-world deployment success.

Our framework addresses critical gaps in UAV-based monitoring of non-green ornamental species, where traditional green-vegetation indices fail and dense canopy occlusion creates counting ambiguity. We propose four self-validated metrics:

- **Radial Counting Stability (RCS)**: Measures spatial prediction consistency under perturbations
- **Cross-Scale Consistency (CSC)**: Evaluates robustness across different UAV flight altitudes  
- **Semantic-Visual Stability (SVS)**: Quantifies segmentation coherence without ground truth
- **Adaptive Repeatability Index (ARI)**: Assesses stochastic consistency under augmentation

Comprehensive evaluation on **469 ultra-high-resolution UAV images** of *Loropetalum chinense* demonstrates strong correlation with deployment success (Spearman ρ = 0.87, p < 0.001), achieving **26.8× faster evaluation** than manual verification and identifying model failures invisible to traditional accuracy metrics.

<p align="center">
  <img src="assets/Abstract_Image.png" alt="Framework" width="80%" style="background-color: white; padding: 20px;">
</p>

<p align="center">
  <img src="assets/metrics.png" alt="Metrics Radar Plot" width="70%">
</p>

<p align="center">

| Dataset | N | RCS | CSC | SVS | ARI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Loropetalum | 469 | 0.71 ± 0.04 | 0.92 ± 0.05 | 0.62 ± 0.05 | 0.41 ± 0.09 |
| MangoNet | 855 | 0.69 ± 0.05 | 0.88 ± 0.06 | 0.64 ± 0.03 | 0.45 ± 0.11 |
| Leafy Spurge | 900 | 0.45 ± 0.28 | 0.52 ± 0.31 | 0.53 ± 0.25 | 0.50 ± 0.45 |
| PlantNet Purple | 800 | 0.57 ± 0.22 | 0.69 ± 0.33 | 0.52 ± 0.16 | 0.29 ± 0.23 |

</p>

<p align="center">
  <img src="assets/DJI_20250408145313_0091_D_analysis.png" alt="Analysis Steps Visualization" width="100%">
</p>


## Our Framework

<p align="center">
  <img src="assets/CVPR_Architecture.png" alt="Framework" width="100%" style="background-color: white; padding: 20px;">
</p>

</div>

## Online Demo

* Visit our demo
  <a href="XXXX-8"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Space-F0CD4B?labelColor=666EEE" alt='HuggingFace Space'></a>
  and test our annotation-free metrics on your own images!
* Upload UAV images of ornamental plants and get instant robustness scores

## 📊 Full Loropetalum Dataset (469 Images)

* The Loropetalum dataset consists of **469 ultra-high-resolution UAV images** (4000×3000 px) of *Loropetalum chinense* canopies captured under diverse field conditions
* Dataset includes:
  - 🌿 Dense canopy occlusion (60-80%)
  - 🎨 Non-green foliage (purple/burgundy colouration)
  - ☀️ Variable lighting (morning, midday, evening)
  - 🚁 Multiple flight altitudes (15-25 meters)
* **Dataset will be released upon paper acceptance**
* Download instructions:
