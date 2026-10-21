# RGB: Refined Gradient Boosting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

**RGB (Refined Gradient Boosting)** is a sophisticated feature selection method designed for high-dimensional biological data. It combines LightGBM with SHAP values to provide robust feature importance estimation through a three-stage selection process.

## Key Features

- **Three-Stage Feature Selection**: Pooling → Distillation → Ranking for robust feature identification
- **MSA Preprocessing**: Convert Multiple Sequence Alignment data to ML-ready features
- **Model Training & Evaluation**: Built-in cross-validation and test evaluation with selected features
- **SHAP-Based Importance**: Reliable feature importance using SHAP values
- **Parallel Processing**: Efficient processing with multi-core support
- **Flexible Configuration**: JSON-based configuration system
- **Command-Line Tools**: Ready-to-use CLI for feature selection, MSA preprocessing, and prediction
- **Multiple Tasks**: Support for both regression and classification

## Installation

```bash
git clone https://github.com/QiuyiLi/RGB.git
cd RGB
pip install -e .
```

### Requirements

- Python >= 3.7
- pandas >= 1.3.0
- numpy >= 1.21.0
- lightgbm >= 3.3.0
- shap >= 0.40.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- tqdm >= 4.60.0

## Quick Start

### 1. MSA Preprocessing

```python
from rgb import MSAPreprocessor

# Initialize preprocessor
preprocessor = MSAPreprocessor(
    min_variant_freq=0.05,
    gene_correlation_threshold=0.8,
    n_jobs=-1  # Use 80% of CPUs
)

# Run full preprocessing pipeline
processed_data = preprocessor.full_preprocessing_pipeline(
    msa_file_path='alignment.csv',
    output_prefix='msa_processed',
    save_results=True
)

print(f"Processed {processed_data.shape[1]} features")
```

### 2. Feature Selection

```python
from rgb import RefinedGradientBoosting, Config

# Load configuration
config = Config('config.json')  # or use Config() for defaults

# Initialize feature selector
selector = RefinedGradientBoosting(config)

# Load data
selector.load_data(
    X_path='features.parquet',  # or .csv
    y_path='target.csv'
)

# Run three-stage feature selection
features_pool = selector.feature_pooling('./results')
features_distilled = selector.feature_distillation(features_pool, './results')
final_features = selector.feature_ranking(features_distilled, n_features=30, output_dir='./results')

print(f"Selected {len(final_features)} features")
```

Target files use the `residuals` column when present, or the only data column
when there is exactly one. For other multi-column target files, specify the
column explicitly with `selector.load_data(..., target_column='trait')` or
`rgb-feature-select --target_column trait ...`. Ambiguous targets and missing
explicitly requested columns raise an error.

### 3. Model Training and Evaluation

```python
from rgb import RGBPredictor

# Initialize predictor
predictor = RGBPredictor(
    task='regression',  # or 'classification'
    n_folds=10,
    test_size=0.2,
    random_state=42
)

# Load data with selected features
predictor.load_data(
    X='features.parquet',
    y='target.csv',
    selected_features='results/feature_ranking.csv'
)

# Run 10-fold cross-validation
cv_results = predictor.cross_validate()
print(f"CV R²: {cv_results['val_r2_mean']:.4f} ± {cv_results['val_r2_std']:.4f}")

# Train and evaluate on test set
test_results = predictor.train_test_evaluate()
print(f"Test R²: {test_results['val_r2']:.4f}")

# Save results and feature importance
predictor.save_results('./prediction_results')
```

## Command-Line Usage

### Feature Selection

```bash
# Basic usage
rgb-feature-select \
    --X_path features.parquet \
    --y_path target.csv \
    --output_dir ./results \
    --n_features 30

# With custom configuration
rgb-feature-select \
    --config my_config.json \
    --X_path features.parquet \
    --y_path target.csv \
    --output_dir ./results \
    --n_features 30 \
    --stage all  # or: pooling, distillation, ranking

# With taxonomy filter
rgb-feature-select \
    --X_path features.parquet \
    --y_path target.csv \
    --taxonomy_path taxonomy.csv \
    --taxonomy_filter "order:Passeriformes" \
    --output_dir ./results
```

### MSA Preprocessing

```bash
# Basic usage
rgb-msa-preprocess \
    --msa_file alignment.csv \
    --output_prefix msa_processed

# With custom parameters
rgb-msa-preprocess \
    --msa_file alignment.csv \
    --min_variant_freq 0.10 \
    --gene_correlation_threshold 0.85 \
    --n_jobs 8 \
    --output_prefix msa_processed

# With taxonomy filter
rgb-msa-preprocess \
    --msa_file alignment.csv \
    --taxonomy_path taxonomy.csv \
    --taxonomy_filter "order:Passeriformes" \
    --output_prefix msa_passeriformes
```

### Model Prediction

```bash
# Basic usage - both CV and test evaluation
rgb-predict \
    --X_path features.parquet \
    --y_path target.csv \
    --features results/feature_ranking.csv \
    --task regression \
    --output_dir ./prediction_results

# Cross-validation only
rgb-predict \
    --X_path features.parquet \
    --y_path target.csv \
    --features results/feature_ranking.csv \
    --task regression \
    --mode cv \
    --n_folds 10

# Classification task
rgb-predict \
    --X_path features.csv \
    --y_path labels.csv \
    --features selected_features.txt \
    --task classification \
    --target_column label \
    --mode both
```

## Method Overview

### Three-Stage Feature Selection

#### Stage 1: Feature Pooling
Accumulates features through iterative training and importance calculation. Features are selected based on SHAP values across multiple model iterations with random data splits.

#### Stage 2: Feature Distillation
Refines the feature pool by removing less important features. This stage focuses on finding features that interact well with the most important features from the pooling stage.

#### Stage 3: Feature Ranking
Final ranking and selection of features using recursive elimination. Features are eliminated in batches, and the final ranking indicates the importance of each feature.

### MSA Preprocessing Pipeline

1. **One-Hot Encoding**: Convert sequences to binary features (GENENAME_POSITION_VARIANT)
2. **Frequency Filtering**: Remove rare and overly common variants
3. **KS Test Filtering**: Keep variants uniformly distributed across samples
4. **Gene-wise Merge**: Remove correlated variants within each gene

## Configuration

Configuration can be provided via JSON file or programmatically:

```json
{
  "model_params": {
    "learning_rate": 0.05,
    "n_estimators": 512,
    "num_leaves": 32,
    "random_state": 42,
    "importance_type": "shap"
  },
  "feature_pooling": {
    "step": 50,
    "n_iter": 20,
    "max_features": 10000
  },
  "feature_distillation": {
    "step": 100,
    "top_n": 50,
    "n_iter": 100
  },
  "feature_ranking": {
    "n_iter": 200
  },
  "data_settings": {
    "test_size": 0.1,
    "validation_size": 0.1
  }
}
```

See [configs/config.json](configs/config.json) for the complete default configuration.

`model_params.importance_type` applies to pooling, distillation, refinement,
and ranking. It defaults to `shap`; `gain` uses total split gain and `split`
uses split counts from LightGBM. Unknown values raise an error. These methods
can produce different feature rankings.

Parquet loading uses `pyarrow>=10.0.1` ([Parquet reader API](https://arrow.apache.org/docs/10.0/python/generated/pyarrow.parquet.ParquetFile.html)). If reading fails specifically because
Thrift metadata exceeds its size limit, RGB retries with string and container
limits of 1000 MiB. Other read errors propagate without a retry.

## Examples

The [examples/](examples/) directory contains detailed usage examples:

- [example_msa_processing.py](examples/example_msa_processing.py): MSA preprocessing examples
- [example_feature_selection.py](examples/example_feature_selection.py): Feature selection examples
- [example_prediction.py](examples/example_prediction.py): Model training and evaluation examples

## API Documentation

### MSAPreprocessor

Class for MSA data preprocessing.

**Methods:**
- `load_data(msa_file_path)`: Load MSA data
- `one_hot_encode(data=None)`: Convert sequences to one-hot features
- `filter_by_variant_frequency(data=None, min_freq=None, max_freq=None)`: Filter by frequency
- `filter_by_ks_test(data=None)`: Filter by KS test for uniform distribution
- `gene_wise_merge(data=None)`: Merge correlated variants within genes
- `full_preprocessing_pipeline(...)`: Run complete pipeline

### RefinedGradientBoosting

Main class for feature selection.

**Methods:**
- `load_data(X_path, y_path, training_samples_path=None, taxonomy_filter=None)`: Load and prepare data
- `feature_pooling(output_dir)`: Stage 1 - Feature pooling
- `feature_distillation(initial_features, output_dir)`: Stage 2 - Feature distillation
- `feature_ranking(features, n_features, output_dir)`: Stage 3 - Feature ranking

### RGBPredictor

Class for model training and evaluation with selected features.

**Methods:**
- `load_data(X, y, selected_features=None, target_column='target')`: Load data
- `cross_validate(X=None, y=None, early_stopping_rounds=50)`: Perform k-fold cross-validation
- `train_test_evaluate(X=None, y=None, early_stopping_rounds=50)`: Train on train set and evaluate on test set
- `save_results(output_dir, prefix='prediction')`: Save prediction results
- `get_cv_results()`: Get cross-validation results as DataFrame
- `get_feature_importance()`: Get feature importance from final model

### Utility Functions

See [rgb/utils.py](src/rgb/utils.py) for data loading, filtering, and statistical utilities.

## Performance Tips

1. **Parallel Processing**: Set `n_jobs=-1` to use 80% of CPU cores
2. **Memory**: For very large datasets (>100K features), use Parquet format for X_path
3. **Speed vs Accuracy**: Reduce `n_iter` in configuration for faster results (may reduce stability)
4. **SHAP Computation**: Most time-consuming step; consider using `importance_type: "gain"` for faster results

## Citation

If you use RGB in your research, please cite:

```bibtex
@software{rgb2024,
  author = {Li, Qiuyi},
  title = {RGB: Refined Gradient Boosting for Feature Selection},
  year = {2024},
  url = {https://github.com/QiuyiLi/RGB}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Qiuyi Li**

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built with [LightGBM](https://github.com/microsoft/LightGBM) for gradient boosting
- Uses [SHAP](https://github.com/slundberg/shap) for feature importance interpretation
