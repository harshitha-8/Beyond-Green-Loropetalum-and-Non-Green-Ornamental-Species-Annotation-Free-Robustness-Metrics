# CVPR Demo: Beyond Green Ornamental Plant Robustness Metrics

This storyboard reformats the ornamental-plant detection project into the same high-resolution CVPR-style demo standard used in the Quantum-Feature-Selection repository, while following the step-by-step stage ordering shown in the original Demonstration_video.mp4 after the first three slides.

## Slide 1: Beyond Green: Annotation-Free Robustness Metrics for Non-Green Ornamental Plant Detection

CVPR-style demonstration storyboard

## Slide 2: Detection Pipeline Demo Frame

![Detection Pipeline Demo Frame](../../assets/demo_first_frame.png)

Opening frame from the original project demo video, showing the UAV perspective over burgundy Loropetalum rows before the downstream detection and robustness analysis stages are presented.

## Slide 3: Annotation-Free Robustness Framework

![Annotation-Free Robustness Framework](../../assets/CVPR_Architecture.png)

System overview covering UAV image acquisition, adaptive color segmentation, shadow and soil removal, morphological cleanup, watershed-based instance grouping, and the annotation-free evaluation metrics RCS, CSC, SVS, and ARI.

## Slide 4: Color Segmentation

![Color Segmentation](../../assets/demo_stage_02_color_segmentation.png)

Stage 2 from the original demonstration video. The pipeline highlights the initial color-based segmentation response used to separate ornamental canopy regions from the background before downstream filtering.

## Slide 5: Shadow Removal

![Shadow Removal](../../assets/demo_stage_03_shadow_removal.png)

Stage 3 from the original demonstration video. Shadow-suppressed processing removes darker nuisance structure while preserving the plant-related response required for stable counting.

## Slide 6: Morphological Cleaning

![Morphological Cleaning](../../assets/demo_stage_04_morphological_cleaning.png)

Stage 4 from the original demonstration video. Morphological cleanup removes fragmented noise and keeps the more meaningful candidate regions aligned with the plant structure.

## Slide 7: Distance Transform

![Distance Transform](../../assets/demo_stage_05_distance_transform.png)

Stage 5 from the original demonstration video. The distance-transform response emphasizes compact local maxima that can seed instance-aware grouping across dense ornamental rows.

## Slide 8: Final Labels

![Final Labels](../../assets/demo_stage_06_final_labels.png)

Stage 6 from the original demonstration video. Final connected labels reveal the grouped plant regions after the successive filtering and instance-generation steps.

## Slide 9: Final Detection

![Final Detection](../../assets/demo_stage_07_final_detection.png)

Stage 7 from the original demonstration video. The final detection overlay shows the bounding-box result after the full pipeline has finished the ornamental plant counting process.
