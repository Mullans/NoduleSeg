# NoduleSeg - Synthetic Dataset Generation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7689509.svg)](https://doi.org/10.5281/zenodo.7689509)

This is the code to generate the synthetic nodule dataset described in **Kernel-Weighted Contribution: A Novel Method of Visual Attribution for 3D Deep Learning Segmentation in Medical Imaging**. A pre-generated sample dataset of 100 training and 100 testing samples can be found on Zenodo either using the badge above or [using this link](https://doi.org/10.5281/zenodo.7689509). The companion code that provides the methods to generate and evaluate visual explanations for segmentation models can be found at [github.com/Mullans/AttributionQuality](https://github.com/Mullans/AttributionQuality)

This algorithm was developed to enable the comprehensive evaluation of visual explanation methods applied to deep learning segmentation predictions. While many datasets exist with ground-truth for the segmentation targets, there are too many confounding factors in natural images to be able to provide a complete evaluation as to whether or not a method of visual explanation (such as GradCAM, ScoreCAM, or Kernel-Weighted Contribution) is identifying the correct contributing factors for the model prediction. The samples generated with this code provide ground-truth labels for the foreground segmentation targets (spiculated nodules), background false-positive targets (non-spiculated nodules), and contributing background objects (spiculations).

## How To Use

From the main directory, calling `scripts/generate_nodule_dataset.py` will generate a dataset of 100 training and 100 testing samples. By default, the samples will be saved in the [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) raw data folder with the task name "Task902_NoduleSeg". You can change the number of samples, task name, or base directory by editing lines 55-58 in the script file.

## Required Packages
* [gouda](https://github.com/Mullans/GOUDA)
* [GoudaMI](https://github.com/Mullans/GoudaMI)
* itk
* [nnunet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1)
* numpy
* [opensimplex](https://github.com/lmas/opensimplex)
* SimpleITK
* tqdm (for progress bars)
* vtk (for surface smoothing)
