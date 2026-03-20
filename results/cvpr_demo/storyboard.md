# Beyond Green Ornamental Plant Robustness Metrics Demo

This storyboard keeps the first three presentation slides and then follows the five high-resolution processing stages used for the ornamental plant detection demo from frame 4 onward.

## Slide 1: Beyond Green: Annotation-Free Robustness Metrics for Non-Green Ornamental Plant Detection



## Slide 2: Detection Pipeline Demo Frame

![Detection Pipeline Demo Frame](../../assets/demo_first_frame.png)

Opening frame from the original project demo video, showing the UAV perspective over burgundy Loropetalum rows before the downstream detection and robustness analysis stages are presented.

## Slide 3: Annotation-Free Robustness Framework

![Annotation-Free Robustness Framework](../../assets/CVPR_Architecture.png)

System overview covering UAV image acquisition, adaptive color segmentation, shadow and soil removal, morphological cleanup, watershed-based instance grouping, and the annotation-free evaluation metrics RCS, CSC, SVS, and ARI.

## Slide 4: Color Segmentation

![Color Segmentation](../../assets/demo_stage_02_color_segmentation.png)

High-resolution binary canopy segmentation showing the first dense extraction of burgundy ornamental plant regions from the UAV scene.

## Slide 5: Morphological Cleaning

![Morphological Cleaning](../../assets/demo_stage_04_morphological_cleaning.png)

High-resolution cleaned mask after the structural filtering stage removes small fragments and preserves the plant instances used in later counting.

## Slide 6: Distance Transform

![Distance Transform](../../assets/demo_stage_05_distance_transform.png)

High-resolution distance-transform response emphasizing the instance centers inside the cleaned ornamental canopy regions.

## Slide 7: Final Labels

![Final Labels](../../assets/demo_stage_06_final_labels.png)

High-resolution connected-label map showing the instance-separated plant regions retained for the final counting result.

## Slide 8: Final Detection

![Final Detection](../../assets/demo_stage_07_final_detection.png)

High-resolution final detection overlay using the provided Total Plants: 1444 frame, showing the complete box-based count over the ornamental rows.
