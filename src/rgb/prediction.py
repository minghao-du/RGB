"""
Prediction Module - Train and evaluate models using selected features

This module provides functionality to train prediction models using features
selected by RGB feature selection, with cross-validation and performance evaluation.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import json
from pathlib import Path
import warnings
from rgb.utils import load_feature_data

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class RGBPredictor:
    """
    Predictor class for training and evaluating models with selected features

    Supports both regression and classification tasks with k-fold cross-validation.
    """

    def __init__(
        self,
        task: str = 'regression',
        n_folds: int = 10,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        model_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize RGB Predictor

        Args:
            task: 'regression' or 'classification'
            n_folds: Number of folds for cross-validation (default: 10)
            test_size: Test set size for final evaluation (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
            model_params: LightGBM model parameters (optional)
        """
        if task not in ['regression', 'classification']:
            raise ValueError("task must be 'regression' or 'classification'")

        self.task = task
        self.n_folds = n_folds
        self.test_size = test_size
        self.random_state = random_state

        # Default model parameters
        self.model_params = self._get_default_params()
        if model_params:
            self.model_params.update(model_params)

        # Results storage
        self.cv_results = []
        self.test_results = {}
        self.trained_models = []
        self.feature_importance = None

        logger.info(f"Initialized RGBPredictor for {task}")
        logger.info(f"Cross-validation: {n_folds}-fold, Test size: {test_size}")

    def _get_default_params(self) -> Dict[str, Any]:
        """Get default model parameters based on task"""
        base_params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_child_samples': 20,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1
        }

        if self.task == 'regression':
            base_params['objective'] = 'regression'
            base_params['metric'] = 'l2'
        else:
            base_params['objective'] = 'binary'
            base_params['metric'] = 'binary_logloss'

        return base_params

    def load_data(
        self,
        X: Union[pd.DataFrame, str, Path],
        y: Union[pd.Series, pd.DataFrame, str, Path],
        selected_features: Optional[Union[List[str], str, Path]] = None,
        target_column: str = 'target'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare data for training

        Args:
            X: Feature DataFrame or path to feature file (CSV/Parquet)
            y: Target Series/DataFrame or path to target file
            selected_features: List of feature names or path to feature file
            target_column: Name of target column in y (if DataFrame)

        Returns:
            Tuple of (X_selected, y)
        """
        # Load X
        if isinstance(X, (str, Path)):
            X_path = Path(X)
            X = load_feature_data(X_path)
            logger.info(f"Loaded X from {X_path}: {X.shape}")

        # Load y
        if isinstance(y, (str, Path)):
            y_path = Path(y)
            y = pd.read_csv(y_path, index_col=0)
            if isinstance(y, pd.DataFrame):
                if target_column in y.columns:
                    y = y[target_column]
                else:
                    y = y.iloc[:, 0]
            logger.info(f"Loaded y from {y_path}: {len(y)} samples")

        # Align indices
        if not all(X.index == y.index):
            logger.warning("X and y indices don't match, aligning...")
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            logger.info(f"Aligned to {len(common_idx)} common samples")

        # Load selected features
        if selected_features is not None:
            if isinstance(selected_features, (str, Path)):
                feat_path = Path(selected_features)
                if feat_path.suffix == '.csv':
                    feat_df = pd.read_csv(feat_path)
                    if 'feature' in feat_df.columns:
                        selected_features = feat_df['feature'].tolist()
                    else:
                        selected_features = feat_df.iloc[:, 0].tolist()
                elif feat_path.suffix == '.txt':
                    with open(feat_path, 'r') as f:
                        selected_features = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(selected_features)} features from {feat_path}")

            # Filter to selected features
            missing_features = set(selected_features) - set(X.columns)
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features in X, using available ones")
                selected_features = [f for f in selected_features if f in X.columns]

            X = X[selected_features]
            logger.info(f"Selected {len(selected_features)} features")

        self.X = X
        self.y = y

        return X, y

    def cross_validate(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        early_stopping_rounds: int = 50
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation

        Args:
            X: Feature DataFrame (uses self.X if None)
            y: Target Series (uses self.y if None)
            early_stopping_rounds: Early stopping rounds for training

        Returns:
            Dictionary with cross-validation results
        """
        if X is None:
            if not hasattr(self, 'X'):
                raise ValueError("No data loaded. Call load_data() first or provide X.")
            X = self.X

        if y is None:
            if not hasattr(self, 'y'):
                raise ValueError("No data loaded. Call load_data() first or provide y.")
            y = self.y

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting {self.n_folds}-Fold Cross-Validation")
        logger.info(f"{'='*60}")

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        self.cv_results = []
        self.trained_models = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            logger.info(f"\nFold {fold_idx}/{self.n_folds}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train model
            if self.task == 'regression':
                model = lgb.LGBMRegressor(**self.model_params)
            else:
                model = lgb.LGBMClassifier(**self.model_params)

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
            )

            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)

            # Evaluate
            if self.task == 'regression':
                results = self._evaluate_regression(y_train, y_pred_train, y_val, y_pred_val)
            else:
                # For classification, get probabilities
                y_pred_train_proba = model.predict_proba(X_train)[:, 1]
                y_pred_val_proba = model.predict_proba(X_val)[:, 1]
                results = self._evaluate_classification(
                    y_train, y_pred_train, y_pred_train_proba,
                    y_val, y_pred_val, y_pred_val_proba
                )

            results['fold'] = fold_idx
            results['n_train'] = len(train_idx)
            results['n_val'] = len(val_idx)
            results['n_estimators_used'] = model.best_iteration_ if hasattr(model, 'best_iteration_') else self.model_params['n_estimators']

            self.cv_results.append(results)
            self.trained_models.append(model)

            # Log fold results
            self._log_fold_results(fold_idx, results)

        # Compute average results
        cv_summary = self._summarize_cv_results()

        logger.info(f"\n{'='*60}")
        logger.info("Cross-Validation Summary")
        logger.info(f"{'='*60}")
        self._log_summary(cv_summary)

        return cv_summary

    def train_test_evaluate(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        early_stopping_rounds: int = 50
    ) -> Dict[str, Any]:
        """
        Train on train set and evaluate on held-out test set

        Args:
            X: Feature DataFrame (uses self.X if None)
            y: Target Series (uses self.y if None)
            early_stopping_rounds: Early stopping rounds for training

        Returns:
            Dictionary with test results
        """
        if X is None:
            X = self.X
        if y is None:
            y = self.y

        logger.info(f"\n{'='*60}")
        logger.info("Train-Test Split Evaluation")
        logger.info(f"{'='*60}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        # Train model
        if self.task == 'regression':
            model = lgb.LGBMRegressor(**self.model_params)
        else:
            model = lgb.LGBMClassifier(**self.model_params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
        )

        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Evaluate
        if self.task == 'regression':
            results = self._evaluate_regression(y_train, y_pred_train, y_test, y_pred_test)
        else:
            y_pred_train_proba = model.predict_proba(X_train)[:, 1]
            y_pred_test_proba = model.predict_proba(X_test)[:, 1]
            results = self._evaluate_classification(
                y_train, y_pred_train, y_pred_train_proba,
                y_test, y_pred_test, y_pred_test_proba
            )

        results['n_train'] = len(X_train)
        results['n_test'] = len(X_test)
        results['n_features'] = X.shape[1]
        results['n_estimators_used'] = model.best_iteration_ if hasattr(model, 'best_iteration_') else self.model_params['n_estimators']

        self.test_results = results
        self.final_model = model

        # Compute feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("\nTest Set Results:")
        self._log_summary(results)

        return results

    def _evaluate_regression(
        self,
        y_train: pd.Series,
        y_pred_train: np.ndarray,
        y_val: pd.Series,
        y_pred_val: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate regression metrics"""
        # Ensure numpy arrays
        y_train_arr = np.array(y_train).flatten()
        y_val_arr = np.array(y_val).flatten()

        return {
            'train_r2': r2_score(y_train_arr, y_pred_train),
            'train_mse': mean_squared_error(y_train_arr, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train_arr, y_pred_train)),
            'train_mae': mean_absolute_error(y_train_arr, y_pred_train),
            'train_corr': np.corrcoef(y_train_arr, y_pred_train)[0, 1],
            'val_r2': r2_score(y_val_arr, y_pred_val),
            'val_mse': mean_squared_error(y_val_arr, y_pred_val),
            'val_rmse': np.sqrt(mean_squared_error(y_val_arr, y_pred_val)),
            'val_mae': mean_absolute_error(y_val_arr, y_pred_val),
            'val_corr': np.corrcoef(y_val_arr, y_pred_val)[0, 1]
        }

    def _evaluate_classification(
        self,
        y_train: pd.Series,
        y_pred_train: np.ndarray,
        y_pred_train_proba: np.ndarray,
        y_val: pd.Series,
        y_pred_val: np.ndarray,
        y_pred_val_proba: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate classification metrics"""
        return {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'train_precision': precision_score(y_train, y_pred_train, average='binary'),
            'train_recall': recall_score(y_train, y_pred_train, average='binary'),
            'train_f1': f1_score(y_train, y_pred_train, average='binary'),
            'train_auc': roc_auc_score(y_train, y_pred_train_proba),
            'val_accuracy': accuracy_score(y_val, y_pred_val),
            'val_precision': precision_score(y_val, y_pred_val, average='binary'),
            'val_recall': recall_score(y_val, y_pred_val, average='binary'),
            'val_f1': f1_score(y_val, y_pred_val, average='binary'),
            'val_auc': roc_auc_score(y_val, y_pred_val_proba)
        }

    def _summarize_cv_results(self) -> Dict[str, Any]:
        """Compute average and std of cross-validation results"""
        if not self.cv_results:
            return {}

        summary = {}

        # Get all metric keys (excluding non-numeric fields)
        metric_keys = [k for k in self.cv_results[0].keys()
                      if k not in ['fold', 'n_train', 'n_val', 'n_estimators_used']]

        for key in metric_keys:
            values = [r[key] for r in self.cv_results]
            summary[f'{key}_mean'] = np.mean(values)
            summary[f'{key}_std'] = np.std(values)

        return summary

    def _log_fold_results(self, fold: int, results: Dict[str, float]) -> None:
        """Log results for a single fold"""
        if self.task == 'regression':
            logger.info(f"  Train - R²: {results['train_r2']:.4f}, RMSE: {results['train_rmse']:.4f}, Corr: {results['train_corr']:.4f}")
            logger.info(f"  Val   - R²: {results['val_r2']:.4f}, RMSE: {results['val_rmse']:.4f}, Corr: {results['val_corr']:.4f}")
        else:
            logger.info(f"  Train - Acc: {results['train_accuracy']:.4f}, F1: {results['train_f1']:.4f}, AUC: {results['train_auc']:.4f}")
            logger.info(f"  Val   - Acc: {results['val_accuracy']:.4f}, F1: {results['val_f1']:.4f}, AUC: {results['val_auc']:.4f}")

    def _log_summary(self, summary: Dict[str, float]) -> None:
        """Log summary results"""
        if self.task == 'regression':
            if 'val_r2_mean' in summary:
                logger.info(f"  R² (Val):   {summary['val_r2_mean']:.4f} ± {summary['val_r2_std']:.4f}")
                logger.info(f"  RMSE (Val): {summary['val_rmse_mean']:.4f} ± {summary['val_rmse_std']:.4f}")
                logger.info(f"  Corr (Val): {summary['val_corr_mean']:.4f} ± {summary['val_corr_std']:.4f}")
            else:
                logger.info(f"  R² (Test):   {summary['val_r2']:.4f}")
                logger.info(f"  RMSE (Test): {summary['val_rmse']:.4f}")
                logger.info(f"  Corr (Test): {summary['val_corr']:.4f}")
        else:
            if 'val_accuracy_mean' in summary:
                logger.info(f"  Accuracy (Val): {summary['val_accuracy_mean']:.4f} ± {summary['val_accuracy_std']:.4f}")
                logger.info(f"  F1 (Val):       {summary['val_f1_mean']:.4f} ± {summary['val_f1_std']:.4f}")
                logger.info(f"  AUC (Val):      {summary['val_auc_mean']:.4f} ± {summary['val_auc_std']:.4f}")
            else:
                logger.info(f"  Accuracy (Test): {summary['val_accuracy']:.4f}")
                logger.info(f"  F1 (Test):       {summary['val_f1']:.4f}")
                logger.info(f"  AUC (Test):      {summary['val_auc']:.4f}")

    def save_results(self, output_dir: Union[str, Path], prefix: str = 'prediction') -> None:
        """
        Save prediction results to files

        Args:
            output_dir: Output directory
            prefix: Prefix for output files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CV results
        if self.cv_results:
            cv_df = pd.DataFrame(self.cv_results)
            cv_path = output_dir / f'{prefix}_cv_results.csv'
            cv_df.to_csv(cv_path, index=False)
            logger.info(f"Saved CV results to {cv_path}")

            # Save CV summary
            cv_summary = self._summarize_cv_results()
            summary_path = output_dir / f'{prefix}_cv_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(cv_summary, f, indent=2)
            logger.info(f"Saved CV summary to {summary_path}")

        # Save test results
        if self.test_results:
            test_path = output_dir / f'{prefix}_test_results.json'
            with open(test_path, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            logger.info(f"Saved test results to {test_path}")

        # Save feature importance
        if self.feature_importance is not None:
            imp_path = output_dir / f'{prefix}_feature_importance.csv'
            self.feature_importance.to_csv(imp_path, index=False)
            logger.info(f"Saved feature importance to {imp_path}")

    def get_cv_results(self) -> pd.DataFrame:
        """Get cross-validation results as DataFrame"""
        return pd.DataFrame(self.cv_results)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from final model"""
        return self.feature_importance


def main():
    """Command-line interface for prediction"""
    import argparse

    parser = argparse.ArgumentParser(
        description='RGB Predictor - Train and evaluate models with selected features'
    )

    # Data arguments
    parser.add_argument('--X_path', type=str, required=True,
                       help='Path to feature data (CSV or Parquet)')
    parser.add_argument('--y_path', type=str, required=True,
                       help='Path to target data (CSV)')
    parser.add_argument('--features', type=str,
                       help='Path to selected features file (CSV or TXT)')
    parser.add_argument('--target_column', type=str, default='target',
                       help='Name of target column (default: target)')

    # Model arguments
    parser.add_argument('--task', type=str, default='regression',
                       choices=['regression', 'classification'],
                       help='Prediction task type')
    parser.add_argument('--n_folds', type=int, default=10,
                       help='Number of CV folds (default: 10)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed (default: 42)')

    # Evaluation mode
    parser.add_argument('--mode', type=str, default='both',
                       choices=['cv', 'test', 'both'],
                       help='Evaluation mode: cv, test, or both (default: both)')

    # Output
    parser.add_argument('--output_dir', type=str, default='./prediction_results',
                       help='Output directory (default: ./prediction_results)')
    parser.add_argument('--prefix', type=str, default='prediction',
                       help='Output file prefix (default: prediction)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    # Initialize predictor
    predictor = RGBPredictor(
        task=args.task,
        n_folds=args.n_folds,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # Load data
    predictor.load_data(
        X=args.X_path,
        y=args.y_path,
        selected_features=args.features,
        target_column=args.target_column
    )

    # Run evaluation
    if args.mode in ['cv', 'both']:
        predictor.cross_validate()

    if args.mode in ['test', 'both']:
        predictor.train_test_evaluate()

    # Save results
    predictor.save_results(args.output_dir, args.prefix)

    logger.info("\n✓ Prediction completed successfully!")


if __name__ == '__main__':
    main()
