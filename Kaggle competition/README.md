## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Place training and test data in the `data/` folder:

- `train.npz` (training features and labels)
- `test.npz` (test features)

## Usage

Run `final.ipynb` to execute the complete pipeline:

1. Feature selection:
2. Model training
3. Submissions

## Output Files

Results are saved to CSV and PNG files including feature selection results, grid search configurations, visualizations, and submission files for each model.
