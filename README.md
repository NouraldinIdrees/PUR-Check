# AI-Supported Image Processing Tool for Automated Cell Recognition and Measurement in polyurethane rigid foam (PUR) Foam Samples

This repository contains code and resources for an AI-based image processing tool designed for the automated detection and measurement of cells in polyurethane rigid foam (PUR) samples. The tool is developed as part of a master's thesis at HafenCity University Hamburg, with the goal of enhancing the quality inspection process for insulation materials in district heating systems.

## Overview

Polyurethane rigid foam (PUR) is widely used in district heating systems due to its excellent thermal insulation properties. The morphology of the foam cells, especially cell size and cell wall thickness, significantly impacts thermal conductivity and material aging. Manual inspection, based on DIN EN 253 standards, is often time-consuming and error-prone. This tool applies image processing techniques to automate the cell recognition and measurement process, improving both efficiency and accuracy.

## Key Features

- **Classical Image Processing Approach**: This tool primarily utilizes classical image processing methods, such as thresholding, watershed transformation, and contour detection, for cell segmentation. The classical approach has proven to be highly effective and forms the core of the solution in both:
  - `01a_pur-cell-analysis-DE.ipynb`
  - `01b_pur-cell-analysis-DE (multi-image).ipynb`
- **Automated Cell Measurement**: Accurately calculates cell diameters and wall thicknesses for quality evaluation against standards like DIN EN 253.
- **Segmentation and Watershed Transformation**: Includes steps for extracting cell structures from microscopy images through adaptive thresholding and watershed segmentation.
- **Model Development**: A deep learning model using the Detectron2 framework has been integrated for research purposes, but it is still under development and currently does not perform as accurately as the classical methods.

  ![PUR-Check_Idrees](https://github.com/user-attachments/assets/c03be6f7-fecd-4510-9fbf-70bd5debf00e)


## Installation

To run the tool locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/NouraldinIdrees/PUR-Check.git
   cd repository
   ```

2. (Optional) Set up a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate  # For Windows: env\Scripts\activate
   ```

3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter notebook:
   ```bash
   jupyter notebook
   ```
   
Dataset with images used for this tool can be accessed [here](https://drive.google.com/drive/folders/1HdWGsBG0VT-5mqdjPOTnnfafEywor_vK?usp=sharing).

## Usage

1. Upload your microscopy images of PUR foam samples to the specified directory.
2. The tool will apply classical image processing methods for segmentation and measurement. Adjustments to thresholding values can be made depending on image quality.
3. For batch processing, use the multi-image processing notebook `01b_pur-cell-analysis-DE (multi-image).ipynb`.

## Detectron2 Model

The repository also includes a Detectron2-based deep learning model for cell segmentation. However, this model is still in development and requires further refinement to match the accuracy of the classical image processing methods. It is included for research purposes and future development.

## GIF Visualization

A GIF showcasing the image processing steps—such as segmentation and watershed transformation—will be added to demonstrate the workflow. This visual aid will provide a clearer understanding of the stages involved in the cell recognition process.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more information.
