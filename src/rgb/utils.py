"""
Utility functions for RGB package

This module provides data loading, filtering, and statistical utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
from scipy import stats
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_feature_data(
    file_path: Union[str, Path],
    index_col: Optional[int] = 0
) -> pd.DataFrame:
    """
    Load feature data from CSV or Parquet file

    Args:
        file_path: Path to the feature data file
        index_col: Column to use as index (default: 0)

    Returns:
        DataFrame with feature data

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading feature data from {file_path}")

    if file_path.suffix in ['.parquet', '.parq']:
        import pyarrow as pa
        import pyarrow.parquet as pq

        try:
            df = pd.read_parquet(file_path, engine='pyarrow')
        except (pa.ArrowInvalid, OSError) as error:
            message = str(error).lower()
            if 'thrift' not in message or 'exceeded size limit' not in message:
                raise
            logger.warning("Parquet Thrift metadata limit exceeded; retrying with 1000 MiB limits")
            parquet_file = pq.ParquetFile(
                file_path,
                thrift_string_size_limit=1000 * 1024 * 1024,
                thrift_container_size_limit=1000 * 1024 * 1024,
            )
            try:
                df = parquet_file.read().to_pandas()
            finally:
                parquet_file.close()
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path, index_col=index_col)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    logger.info(f"Loaded data with shape {df.shape}")
    return df


def load_target_data(
    file_path: Union[str, Path],
    target_column: Optional[str] = None,
    index_col: Optional[int] = 0
) -> pd.Series:
    """
    Load target variable data

    Args:
        file_path: Path to the target data file
        target_column: Explicit target column, or auto-select residuals/the only column
        index_col: Column to use as index (default: 0)

    Returns:
        Series with target data

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If the target column is missing or ambiguous
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading target data from {file_path}")

    df = pd.read_csv(file_path, index_col=index_col)

    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError(f"Requested target_column {target_column!r} not found in {list(df.columns)}")
        data = df[target_column]
    elif 'residuals' in df.columns:
        data = df['residuals']
    elif len(df.columns) == 1:
        data = df.iloc[:, 0]
    else:
        raise ValueError(
            f"Expected one target column or a 'residuals' column; found {list(df.columns)}. "
            "Specify target_column explicitly."
        )

    logger.info(f"Loaded target with {len(data)} samples")
    return data


def load_msa_data(
    file_path: Union[str, Path],
    index_col: Optional[int] = 0
) -> pd.DataFrame:
    """
    Load Multiple Sequence Alignment (MSA) data

    Args:
        file_path: Path to MSA CSV file
        index_col: Column to use as index (default: 0)

    Returns:
        DataFrame with MSA sequences

    Raises:
        FileNotFoundError: If file does not exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading MSA data from {file_path}")

    df = pd.read_csv(file_path, index_col=index_col)

    logger.info(f"Loaded MSA data: {df.shape[0]} samples, {df.shape[1]} genes")
    return df


def align_data_indices(
    X: pd.DataFrame,
    y: pd.Series,
    strict: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align indices between features and target

    Args:
        X: Feature DataFrame
        y: Target Series
        strict: If True, raise error if indices don't match perfectly

    Returns:
        Tuple of (aligned_X, aligned_y)

    Raises:
        ValueError: If strict=True and indices don't match
    """
    if list(X.index) == list(y.index):
        logger.info("Indices already aligned")
        return X, y

    if strict:
        raise ValueError("Indices don't match and strict mode is enabled")

    logger.warning("Aligning indices - some samples may be dropped")

    common_idx = X.index.intersection(y.index)

    if len(common_idx) == 0:
        raise ValueError("No common indices found between X and y")

    X_aligned = X.loc[common_idx]
    y_aligned = y.loc[common_idx]

    logger.info(f"Aligned data: {len(common_idx)} common samples "
                f"(dropped {len(X) - len(common_idx)} from X, "
                f"{len(y) - len(common_idx)} from y)")

    return X_aligned, y_aligned


def apply_taxonomy_filter(
    data: pd.DataFrame,
    taxonomy_path: Union[str, Path],
    filter_dict: Dict[str, Any],
    index_col: str = 'species_name'
) -> pd.DataFrame:
    """
    Filter data based on taxonomy criteria

    Args:
        data: Data to filter
        taxonomy_path: Path to taxonomy CSV file
        filter_dict: Dictionary of column:value pairs to filter on
        index_col: Column in taxonomy data to use for matching indices

    Returns:
        Filtered DataFrame

    Example:
        >>> filtered = apply_taxonomy_filter(
        ...     data, 'taxonomy.csv', {'order': 'Passeriformes'}
        ... )
    """
    taxonomy_df = pd.read_csv(taxonomy_path)

    logger.info(f"Applying taxonomy filter: {filter_dict}")

    # Apply filters
    mask = pd.Series([True] * len(taxonomy_df))
    for col, value in filter_dict.items():
        if col not in taxonomy_df.columns:
            raise ValueError(f"Column '{col}' not found in taxonomy data")
        mask &= (taxonomy_df[col] == value)

    filtered_taxonomy = taxonomy_df.loc[mask]

    if index_col not in filtered_taxonomy.columns:
        raise ValueError(f"Index column '{index_col}' not found in taxonomy data")

    # Filter data
    valid_indices = filtered_taxonomy[index_col].values
    filtered_data = data.loc[data.index.isin(valid_indices)]

    logger.info(f"Filtered from {len(data)} to {len(filtered_data)} samples")

    return filtered_data


def save_results(
    data: Union[pd.DataFrame, pd.Series],
    output_path: Union[str, Path],
    index: bool = True
) -> None:
    """
    Save results to file (CSV or Parquet based on extension)

    Args:
        data: Data to save
        output_path: Output file path
        index: Whether to include index in output
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix in ['.parquet', '.parq']:
        data.to_parquet(output_path, index=index)
    else:
        data.to_csv(output_path, index=index)

    logger.info(f"Saved results to {output_path}")


# ============================================================================
# Statistical Filtering Functions
# ============================================================================

def _process_frequency_chunk(args):
    """
    Process a chunk of columns for frequency filtering (module-level for pickling)

    Args:
        args: Tuple of (cols, data_subset, n_samples)

    Returns:
        List of (col, frequency) tuples
    """
    cols, data_subset, n_samples = args

    # Vectorized computation for the chunk
    frequencies = (data_subset != 0).sum(axis=0) / n_samples

    return list(zip(cols, frequencies))


def filter_by_frequency(
    data: pd.DataFrame,
    min_freq: float = 0.05,
    max_freq: float = 0.95,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Filter features by frequency of non-zero values

    Args:
        data: Feature DataFrame
        min_freq: Minimum frequency (default: 0.05)
        max_freq: Maximum frequency (default: 0.95)
        n_jobs: Number of parallel jobs (default: -1, uses 80% of CPUs)

    Returns:
        Filtered DataFrame
    """
    n_samples = len(data)

    if n_jobs != 1 and len(data.columns) > 100:
        # Use 80% of CPUs if n_jobs == -1
        if n_jobs == -1:
            actual_jobs = max(1, int(cpu_count() * 0.8))
        else:
            actual_jobs = max(1, n_jobs)

        # Split columns into chunks and pass data subsets
        col_chunks = np.array_split(data.columns, actual_jobs)

        # Prepare arguments: pass only the subset of data for each chunk
        args_list = [
            (chunk.tolist(), data[chunk].values, n_samples)
            for chunk in col_chunks
        ]

        # Parallel processing with progress bar
        logger.info(f"Filtering by frequency with {actual_jobs} processes...")

        with Pool(processes=actual_jobs) as pool:
            results = list(tqdm(
                pool.imap(_process_frequency_chunk, args_list),
                total=len(args_list),
                desc=f"Frequency filter ({len(data.columns)} features)"
            ))

        # Flatten results and filter
        frequencies = {}
        for chunk_result in results:
            for col, freq in chunk_result:
                frequencies[col] = freq

        filtered_cols = [col for col, freq in frequencies.items()
                        if min_freq <= freq <= max_freq]
    else:
        # Sequential processing - vectorized
        frequencies = (data != 0).sum(axis=0) / n_samples
        mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        filtered_cols = data.columns[mask].tolist()

    logger.info(f"Frequency filter: {len(filtered_cols)}/{len(data.columns)} features retained")

    return data[filtered_cols]


def _process_conserved_chunk(args):
    """
    Process a chunk of columns for conservation filtering (module-level for pickling)

    Args:
        args: Tuple of (cols, data, n_samples)

    Returns:
        List of (col, max_freq) tuples
    """
    cols, data, n_samples = args
    max_freqs = []
    for col in cols:
        value_counts = data[col].value_counts()
        if len(value_counts) > 0:
            max_freq = value_counts.iloc[0] / n_samples
            max_freqs.append((col, max_freq))
        else:
            max_freqs.append((col, 1.0))
    return max_freqs


def filter_conserved_positions(
    data: pd.DataFrame,
    conservation_threshold: float = 0.95,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Remove highly conserved positions (features with same value across most samples)

    Args:
        data: Feature DataFrame
        conservation_threshold: Threshold for conservation (default: 0.95)
        n_jobs: Number of parallel jobs (default: -1, uses 80% of CPUs; set to 1 for sequential)

    Returns:
        DataFrame with conserved positions removed
    """
    n_samples = len(data)

    if n_jobs != 1 and len(data.columns) > 100:
        # Use 80% of CPUs if n_jobs == -1
        if n_jobs == -1:
            actual_jobs = max(1, int(cpu_count() * 0.8))
        else:
            actual_jobs = max(1, n_jobs)

        # Split columns into chunks
        col_chunks = np.array_split(data.columns, actual_jobs)

        # Prepare arguments
        args_list = [(chunk.tolist(), data, n_samples) for chunk in col_chunks]

        # Parallel processing with progress bar
        logger.info(f"Filtering conserved positions with {actual_jobs} processes...")

        with Pool(processes=actual_jobs) as pool:
            results = list(tqdm(
                pool.imap(_process_conserved_chunk, args_list),
                total=len(args_list),
                desc="Conservation filter"
            ))

        # Flatten results
        max_frequencies = {}
        for chunk_result in results:
            for col, freq in chunk_result:
                max_frequencies[col] = freq

        # Filter columns
        filtered_cols = [col for col, freq in max_frequencies.items()
                        if freq < conservation_threshold]
    else:
        # Sequential processing with progress bar
        max_frequencies = []
        for col in tqdm(data.columns, desc="Conservation filter"):
            value_counts = data[col].value_counts()
            if len(value_counts) > 0:
                max_freq = value_counts.iloc[0] / n_samples
                max_frequencies.append(max_freq)
            else:
                max_frequencies.append(1.0)

        # Keep columns where max frequency is below threshold
        mask = np.array(max_frequencies) < conservation_threshold
        filtered_cols = data.columns[mask]

    logger.info(f"Conservation filter: {len(filtered_cols)}/{len(data.columns)} features retained")

    return data[filtered_cols]


def remove_correlated_features(
    data: pd.DataFrame,
    correlation_threshold: float = 0.95,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Remove highly correlated features

    Args:
        data: Feature DataFrame
        correlation_threshold: Correlation threshold (default: 0.95)
        method: Correlation method: 'pearson', 'kendall', 'spearman' (default: 'pearson')

    Returns:
        DataFrame with correlated features removed
    """
    if len(data.columns) == 0:
        return data

    # Calculate correlation matrix
    corr_matrix = data.corr(method=method)

    # Find correlated features
    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                # Remove the feature with higher index
                correlated_features.add(corr_matrix.columns[i])
                break

    # Keep non-correlated features
    filtered_cols = [col for col in data.columns if col not in correlated_features]

    logger.info(f"Correlation filter: removed {len(correlated_features)} correlated features, "
                f"{len(filtered_cols)} retained")

    return data[filtered_cols]


def _process_ks_test_chunk(args):
    """
    Process a chunk of columns for KS test filtering (module-level for pickling)

    Args:
        args: Tuple of (cols, data_subset, n_samples, p_threshold, min_freq, max_freq)

    Returns:
        List of (col, p_value, passes_test) tuples
    """
    cols, data_subset, n_samples, p_threshold, min_freq, max_freq = args
    from scipy import stats

    results = []
    for i, col in enumerate(cols):
        col_data = data_subset[:, i]

        # Calculate frequency of 1s
        sum_ones = col_data.sum()
        freq = sum_ones / n_samples

        # Filter by frequency range first (must be between min_freq and max_freq)
        if freq < min_freq or freq > max_freq:
            results.append((col, 0.0, False))
            continue

        # For positions where the variant is present (value == 1),
        # test if their indices are uniformly distributed
        indices_with_ones = np.where(col_data == 1)[0]

        if len(indices_with_ones) == 0:
            results.append((col, 0.0, False))
            continue

        # Normalize indices to [0, 1]
        normalized_positions = indices_with_ones / n_samples

        # KS test against uniform distribution
        # Use 'uniform' string to match the original implementation
        ks_stat, p_value = stats.kstest(normalized_positions, 'uniform', method='exact')

        passes_test = p_value > p_threshold
        results.append((col, p_value, passes_test))

    return results


def filter_by_ks_test(
    data: pd.DataFrame,
    p_threshold: float = 0.05,
    min_freq: float = 0.2,
    max_freq: float = 0.8,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Filter binary features based on Kolmogorov-Smirnov test for uniform distribution

    For each binary (0/1) feature:
    1. Check if frequency of 1s is between min_freq and max_freq
    2. For positions where value == 1, test if their indices are uniformly distributed
    3. Keep features where KS test p-value > p_threshold

    This ensures that variants are:
    - Not too rare or too common (frequency filter)
    - Uniformly distributed across samples (KS test)

    Args:
        data: Feature DataFrame (expects binary 0/1 encoded features)
        p_threshold: P-value threshold (default: 0.05, keeps features with p > 0.05)
        min_freq: Minimum frequency of 1s (default: 0.2)
        max_freq: Maximum frequency of 1s (default: 0.8)
        n_jobs: Number of parallel jobs (default: -1, uses 80% of CPUs; set to 1 for sequential)

    Returns:
        DataFrame with only features that pass the KS test
    """
    n_samples = len(data)
    logger.info(f"Filtering by KS test (freq: [{min_freq}, {max_freq}], p > {p_threshold})...")

    if n_jobs != 1 and len(data.columns) > 100:
        # Use 80% of CPUs if n_jobs == -1
        if n_jobs == -1:
            actual_jobs = max(1, int(cpu_count() * 0.8))
        else:
            actual_jobs = max(1, n_jobs)

        # Split columns into chunks
        col_chunks = np.array_split(data.columns, actual_jobs)

        # Prepare arguments: pass only the subset of data for each chunk
        args_list = [
            (chunk.tolist(), data[chunk].values, n_samples, p_threshold, min_freq, max_freq)
            for chunk in col_chunks
        ]

        # Parallel processing with progress bar
        logger.info(f"Running KS tests with {actual_jobs} processes...")

        with Pool(processes=actual_jobs) as pool:
            results = list(tqdm(
                pool.imap(_process_ks_test_chunk, args_list),
                total=len(args_list),
                desc=f"KS test ({len(data.columns)} features)"
            ))

        # Flatten results and filter
        filtered_cols = []
        for chunk_result in results:
            for col, p_value, passes_test in chunk_result:
                if passes_test:
                    filtered_cols.append(col)
    else:
        # Sequential processing with progress bar
        from scipy import stats

        filtered_cols = []
        for col in tqdm(data.columns, desc="KS test"):
            col_data = data[col].values

            # Calculate frequency of 1s
            sum_ones = col_data.sum()
            freq = sum_ones / n_samples

            # Filter by frequency range first
            if freq < min_freq or freq > max_freq:
                continue

            # For positions where the variant is present (value == 1),
            # test if their indices are uniformly distributed
            indices_with_ones = np.where(col_data == 1)[0]

            if len(indices_with_ones) == 0:
                continue

            # Normalize indices to [0, 1]
            normalized_positions = indices_with_ones / n_samples

            # KS test against uniform distribution
            # Use 'uniform' string to match the original implementation
            ks_stat, p_value = stats.kstest(normalized_positions, 'uniform', method='exact')

            if p_value > p_threshold:
                filtered_cols.append(col)

    logger.info(f"KS test filter: {len(filtered_cols)}/{len(data.columns)} features retained "
               f"(freq: [{min_freq}, {max_freq}], p > {p_threshold})")

    return data[filtered_cols]


def _process_gene_group_for_merge(args):
    """
    Process a single gene group for merging (module-level function for pickling)

    Args:
        args: Tuple of (gene_id, features, gene_data_values, gene_data_index, correlation_threshold, method)

    Returns:
        Tuple of (gene_kept_data, gene_merge_dict, gene_removed)
    """
    gene_id, features, gene_data_values, gene_data_index, correlation_threshold, method = args

    if len(features) == 1:
        # Single feature, no merging needed
        return {features[0]: gene_data_values[:, 0]}, {features[0]: [features[0]]}, 0

    # Reconstruct gene_data from values
    import pandas as pd
    gene_data = pd.DataFrame(gene_data_values, index=gene_data_index, columns=features)

    # Calculate correlation matrix
    corr_matrix = gene_data.corr()

    # Find correlated groups - keep only the first feature from each group
    kept_features = {}
    merge_dict = {}
    removed_count = 0
    processed = set()

    for i in range(len(corr_matrix.columns)):
        if corr_matrix.columns[i] in processed:
            continue

        # This is the first feature in a group - we keep it
        first_feature = corr_matrix.columns[i]
        group = [first_feature]

        # Find all features highly correlated with this one
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.columns[j] not in processed:
                if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                    group.append(corr_matrix.columns[j])
                    processed.add(corr_matrix.columns[j])

        processed.add(first_feature)

        # Keep only the first feature from this group
        kept_features[first_feature] = gene_data[first_feature].values
        merge_dict[first_feature] = group

        # Count removed features
        if len(group) > 1:
            removed_count += len(group) - 1

    return kept_features, merge_dict, removed_count


def gene_wise_merge_variants(
    data: pd.DataFrame,
    correlation_threshold: float = 0.8,
    gene_delimiter: str = '_',
    gene_position: int = 1,
    method: str = 'max',
    n_jobs: int = -1
) -> Tuple[pd.DataFrame, dict]:
    """
    Remove correlated variants within each gene separately (keep only the first)

    For each gene, finds groups of highly correlated features and keeps only
    the first feature from each group, removing the rest.

    Assumes feature names follow pattern: prefix_GENEID_position_variant

    Args:
        data: Feature DataFrame
        correlation_threshold: Correlation threshold (default: 0.8)
        gene_delimiter: Delimiter in feature names (default: '_')
        gene_position: Position of gene ID in split name (default: 1)
        method: Deprecated - kept for backward compatibility, not used
        n_jobs: Number of parallel jobs (default: -1, uses 80% of CPUs; set to 1 for sequential)

    Returns:
        Tuple of (filtered DataFrame, merge dictionary)

        The merge dictionary maps kept features to the list of features they represent.
        Example: {'gene1_0_A': ['gene1_0_A', 'gene1_1_A', 'gene1_2_A']}
        means gene1_0_A was kept and gene1_1_A, gene1_2_A were removed due to high correlation.
    """
    # Group features by gene
    gene_groups = {}
    for col in data.columns:
        parts = col.split(gene_delimiter)
        if len(parts) > gene_position:
            gene_id = parts[gene_position]
            if gene_id not in gene_groups:
                gene_groups[gene_id] = []
            gene_groups[gene_id].append(col)
        else:
            # If can't parse gene ID, keep the feature
            if 'unknown' not in gene_groups:
                gene_groups['unknown'] = []
            gene_groups['unknown'].append(col)

    logger.info(f"Found {len(gene_groups)} gene groups")

    # Prepare arguments: pass only the data for each gene (not the whole DataFrame)
    args_list = [
        (gene_id, features, data[features].values, data.index, correlation_threshold, method)
        for gene_id, features in gene_groups.items()
    ]

    # Process gene groups (parallel or sequential)
    if n_jobs != 1 and len(gene_groups) > 1:
        # Use 80% of CPUs if n_jobs == -1
        if n_jobs == -1:
            actual_jobs = max(1, int(cpu_count() * 0.8))
        else:
            actual_jobs = max(1, n_jobs)

        # Parallel processing with progress bar
        logger.info(f"Merging gene groups with {actual_jobs} processes...")

        with Pool(processes=actual_jobs) as pool:
            results = list(tqdm(
                pool.imap(_process_gene_group_for_merge, args_list),
                total=len(args_list),
                desc=f"Gene-wise merge ({len(gene_groups)} genes)"
            ))
    else:
        # Sequential processing with progress bar
        results = [
            _process_gene_group_for_merge(args)
            for args in tqdm(args_list, desc=f"Gene-wise merge ({len(gene_groups)} genes)")
        ]

    # Combine results - optimized to avoid repeated DataFrame copies
    logger.info("Combining merged results...")

    all_merged_data = {}
    total_merge_dict = {}
    total_removed = 0

    for gene_merged_data, gene_merge_dict, gene_removed in results:
        all_merged_data.update(gene_merged_data)
        total_merge_dict.update(gene_merge_dict)
        total_removed += gene_removed

    # Create DataFrame once from dict (much faster than adding columns iteratively)
    merged_data = pd.DataFrame(all_merged_data, index=data.index)

    logger.info(f"Gene-wise correlation filter: {len(data.columns)} features -> {len(merged_data.columns)} features")
    logger.info(f"  Removed {total_removed} correlated features (kept first from each group)")

    return merged_data, total_merge_dict

