from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rgb import Config, RefinedGradientBoosting, RGBPredictor
from rgb import feature_selection, utils


@pytest.fixture
def selector():
    config = Config()
    config._config['model_params'].update(n_estimators=5, n_jobs=1, num_threads=1)
    selector = RefinedGradientBoosting(config)
    rng = np.random.RandomState(42)
    selector.X_train = pd.DataFrame(rng.normal(size=(60, 3)), columns=['a', 'b', 'c'])
    selector.y_train = 5 * selector.X_train['a'] + selector.X_train['b']
    selector.X_test = selector.X_train.iloc[:10].copy()
    selector.y_test = selector.y_train.iloc[:10].copy()
    return selector


@pytest.mark.parametrize('kind', ['gain', 'split'])
def test_builtin_importance_matches_booster(selector, kind):
    model = selector._create_model()
    model.fit(selector.X_train, selector.y_train)
    actual = selector._compute_feature_importance(model, selector.X_train, 0.5, kind)
    expected = pd.Series(
        0.5 * model.booster_.feature_importance(importance_type=kind),
        index=selector.X_train.columns,
    )
    np.testing.assert_allclose(actual.set_index('feature')['importance'].reindex(expected.index), expected)


def test_unknown_importance_is_rejected(selector):
    with pytest.raises(ValueError, match='importance_type'):
        selector._compute_feature_importance(None, selector.X_train, 1., 'gian')


@pytest.mark.parametrize('configured', ['gain', None])
def test_refinement_and_ranking_honor_config(selector, monkeypatch, tmp_path, configured):
    params = selector.config._config['model_params']
    if configured is None:
        params.pop('importance_type')
    else:
        params['importance_type'] = configured
    model = selector._create_model()
    model.fit(selector.X_train, selector.y_train)
    metrics = dict(cor_train=1., cor_val=1., cor_test=1., mse_val=0., mae_val=0.)
    monkeypatch.setattr(selector, '_train_and_evaluate', lambda *a, **k: (model, metrics))
    shap_calls = []

    def explainer(model):
        def explain(X):
            shap_calls.append(True)
            return SimpleNamespace(values=np.tile([1., 2., 3.], (len(X), 1)))
        return explain

    monkeypatch.setattr(feature_selection.shap, 'Explainer', explainer)
    selector.feature_pool = selector.X_train.columns.tolist()
    selector.X_train_distill = selector.X_train
    selector.X_test_distill = selector.X_test
    accuracy = str(tmp_path / 'accuracy.csv')
    selector._refine_feature_pool(accuracy, 0, {'refinement_iter': 1})
    selector._refine_distillation_pool(selector.X_train.columns.tolist(), accuracy, 0, {'refinement_iter': 1})
    ranking, _ = selector._run_ranking_iteration(selector.X_train.columns.tolist(), accuracy, 0, {'n_iter': 1})
    if configured == 'gain':
        assert not shap_calls
        expected = model.booster_.feature_importance(importance_type='gain')
        np.testing.assert_allclose(ranking.set_index('feature')['importance'].reindex(selector.X_train.columns), expected)
    else:
        assert len(shap_calls) == 3
        assert ranking['feature'].tolist() == ['c', 'b', 'a']


@pytest.mark.parametrize('n_iter', [0, 1])
def test_empty_distillation_returns_empty_result(selector, monkeypatch, n_iter):
    selector.X_train_distill = selector.X_train.iloc[:, :0]
    selector.X_test_distill = selector.X_test.iloc[:, :0]
    monkeypatch.setattr(selector, '_train_and_evaluate', lambda *a, **k: pytest.fail('trained empty data'))
    summary, metrics = selector._run_distillation_iteration([], {'n_iter': n_iter}, 'shap', 0)
    assert summary.empty
    assert metrics == []


def test_distillation_after_dropping_all_features(selector, monkeypatch):
    selector.X_train_distill = selector.X_train
    selector.X_test_distill = selector.X_test
    monkeypatch.setattr(selector, '_train_and_evaluate', lambda *a, **k: pytest.fail('trained empty data'))
    summary, metrics = selector._run_distillation_iteration(['a', 'b', 'c'], {'n_iter': 1}, 'shap', 0)
    assert summary.empty and metrics == []


def test_empty_public_distillation(selector, tmp_path):
    assert selector.feature_distillation([], str(tmp_path)) == []


@pytest.mark.parametrize('columns,target,expected', [
    ({'trait': [1., 2., 3.]}, None, 'trait'),
    ({'metadata': [4., 5., 6.], 'residuals': [1., 2., 3.]}, None, 'residuals'),
    ({'metadata': [4., 5., 6.], 'trait': [1., 2., 3.]}, 'trait', 'trait'),
])
def test_selector_loads_unambiguous_target(tmp_path, columns, target, expected):
    X = tmp_path / 'X.csv'
    y = tmp_path / 'y.csv'
    pd.DataFrame({'a': [1., 2., 3.]}).to_csv(X)
    pd.DataFrame(columns).to_csv(y)
    selector = RefinedGradientBoosting(Config())
    selector.load_data(str(X), str(y), target_column=target)
    assert isinstance(selector.y_train, pd.Series)
    assert selector.y_all.name == expected


@pytest.mark.parametrize('columns,target', [
    ({'metadata': [1., 2.], 'trait': [3., 4.]}, None),
    ({'trait': [1., 2.]}, 'missing'),
    ({}, None),
])
def test_invalid_target_fails_clearly(tmp_path, columns, target):
    y = tmp_path / 'y.csv'
    pd.DataFrame(columns, index=[0, 1]).to_csv(y)
    with pytest.raises(ValueError, match='target_column|target column'):
        utils.load_target_data(y, target_column=target)


@pytest.mark.parametrize('consumer', ['selector', 'predictor', 'utility'])
@pytest.mark.parametrize('limited', [False, True])
def test_parquet_preserves_index_and_values(tmp_path, monkeypatch, consumer, limited):
    expected = pd.DataFrame({'a': [1., 2., 3.]}, index=pd.Index(['x', 'y', 'z'], name='sample'))
    path = tmp_path / 'X.parquet'
    y = tmp_path / 'y.csv'
    expected.to_parquet(path)
    pd.DataFrame({'residuals': [1., 2., 3.]}, index=expected.index).to_csv(y)
    if limited:
        read_parquet = pd.read_parquet
        def restricted(path, **kwargs):
            return read_parquet(path, thrift_string_size_limit=1, **kwargs)
        monkeypatch.setattr(pd, 'read_parquet', restricted)
    if consumer == 'selector':
        selector = RefinedGradientBoosting(Config())
        selector.load_data(str(path), str(y))
        actual = selector.X_all
    elif consumer == 'predictor':
        actual, _ = RGBPredictor().load_data(path, y)
    else:
        actual = utils.load_feature_data(path)
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize('error', [
    FileNotFoundError('missing'), PermissionError('denied'),
    pa.ArrowInvalid('Parquet magic bytes not found in footer'),
])
def test_unrelated_parquet_errors_are_not_retried(monkeypatch, tmp_path, error):
    path = tmp_path / 'X.parquet'
    path.touch()
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(pd, 'read_parquet', fail)
    monkeypatch.setattr(pq, 'ParquetFile', lambda *a, **k: pytest.fail('unexpected fallback'))
    with pytest.raises(type(error)) as caught:
        utils.load_feature_data(path)
    assert caught.value is error


def test_parquet_fallback_preserves_original_error(monkeypatch, tmp_path):
    path = tmp_path / 'X.parquet'
    path.touch()
    original = pa.ArrowInvalid("Couldn't deserialize thrift: Exceeded size limit")
    def fail(*args, **kwargs):
        raise original
    def retry(*args, **kwargs):
        raise OSError('retry failed')
    monkeypatch.setattr(pd, 'read_parquet', fail)
    monkeypatch.setattr(pq, 'ParquetFile', retry)
    with pytest.raises(OSError) as caught:
        utils.load_feature_data(path)
    assert caught.value.__context__ is original


@pytest.mark.parametrize('kind', ['shap', 'gain'])
def test_three_stage_pipeline_with_real_training(tmp_path, kind):
    config = Config()
    config._config['model_params'].update(
        importance_type=kind, n_estimators=8, num_leaves=4,
        n_jobs=1, num_threads=1, early_stopping_round=3,
    )
    config._config['feature_pooling'].update(step=1, n_iter=1, refinement_iter=1, max_features=100)
    config._config['feature_distillation'].update(
        step=1, top_n=0, n_iter=1, refinement_iter=1, max_features=3,
    )
    config._config['feature_ranking']['n_iter'] = 1
    signal = np.linspace(-2., 2., 120)
    X = pd.DataFrame({'a': signal, 'b': 2 * signal, 'c': -signal})
    y = pd.DataFrame({'trait': 3 * signal})
    X_path, y_path = tmp_path / 'X.parquet', tmp_path / 'y.csv'
    X.to_parquet(X_path)
    y.to_csv(y_path)
    selector = RefinedGradientBoosting(config)
    selector.load_data(X_path, y_path, target_column='trait')
    pool = selector.feature_pooling(str(tmp_path / 'pooling'))
    distilled = selector.feature_distillation(pool, str(tmp_path / 'distillation'))
    ranked = selector.feature_ranking(distilled, 3, str(tmp_path / 'ranking'))
    assert set(pool) == set(distilled) == set(ranked['feature']) == set(X.columns)
    assert (tmp_path / 'ranking' / 'feature_ranking.csv').is_file()
