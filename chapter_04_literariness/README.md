# Chapter 4 - Literariness

This repository contains the pipeline to process and analyze the CONLIT dataset for Chapter 4.

## Data Requirements (CONLIT Dataset)
The CONLIT dataset (Contemporary Literature) is a hand-curated corpus compiled by the txtLAB at McGill University. Due to size constraints, the raw dataset files are not included in this repository. 

To run the pipeline, you must acquire the CONLIT dataset (available via txtLAB/Figshare) and place the following CSV files in the root directory of this chapter:
- `CONLIT_META.csv`
- `CONLIT_SUPERSENSE.csv`
- `CONLIT_UNIGRAM.csv`
*(Note: `CONLIT_LIWC.csv` is also used in some analyses but may require a separate LIWC license).*

## Requirements
To run these scripts, install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Pipeline
You can run the full pipeline in order by executing:
```bash
python run_all.py
```

The pipeline consists of the following steps:
1. `01_prepare.py`: Prepares the master analytic dataset.
2. `02_axes.py`: Fits three classifiers and tests orthogonality.
3. `03_stats.py`: Computes univariate discrimination, PCA, and genre profiles.
4. `04_figures.py`: Generates all publication figures.
5. `05_workbook.py`: Assembles result tables into a formatted Excel workbook.

All outputs will be saved to the `results/` folder.
