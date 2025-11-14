# Naturalness Predictor Using ChemBERTa

This project aims to predict the naturalness of chemical compounds using a fine-tuned ChemBERTa foundation model. The work serves as a preliminary proof-of-concept study with promising results.

## Dataset

The dataset was compiled from two major sources:

1. **Natural Products Atlas:** Contains information on natural products.
2. **ChEMBL:** A comprehensive database of small molecules.

The dataset consists of the following:

- Each entry in the two data sources was matched using the InChI Key.
- Out of approximately 1.5 million molecules in ChEMBL, 5,298 were marked as natural, with the remaining marked as unnatural. Note that this dataset is not 100% accurate but follows a common strategy where negative data is expected to dominate.

## Methodology

1. **Data Preparation:**
   - All natural products in the annotated ChEMBL dataset were used for further analysis.
   - Unnatural molecules were randomly sampled to make the negative data 1.5x the size of the positive data.

2. **Model Training:**
   - 80% of the dataset was used to fine-tune the ChemBERTa foundation model.
   - The remaining 20% of the dataset was reserved for performance evaluation.

3. **Results:**
   - The prediction achieved an accuracy of 93.88%. This was accomplished using a simple fine-tuning strategy, which required only a few minutes on a personal computer.

## Future Work

This project is just a starting point and will benefit from further refinement and testing for a more robust evaluation of the strategy. Future work will focus on enhancing data accuracy and exploring additional features that can improve prediction capability.
