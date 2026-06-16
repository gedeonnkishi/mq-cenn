import numpy as np
import pytest

from mq_cenn.core.reliability import NoveltyDetector, ReliabilityCalibrator


def test_novelty_detector_fit_score_shape():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 4))

    detector = NoveltyDetector().fit(X)
    scores = detector.score(X[:5])

    assert scores.shape == (5,)
    assert np.isfinite(scores).all()


def test_novelty_detector_score_before_fit_raises_error():
    X = np.random.randn(5, 3)

    detector = NoveltyDetector()

    with pytest.raises(RuntimeError):
        detector.score(X)


def test_reliability_calibrator_scores_between_zero_and_one():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 4))
    pool_preds = rng.normal(size=(40, 3))

    calibrator = ReliabilityCalibrator().fit(X, pool_preds)
    scores = calibrator.score(X[:10], pool_preds[:10])

    assert scores.shape == (10,)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_reliability_calibrator_rejects_length_mismatch():
    X = np.random.randn(20, 4)
    pool_preds = np.random.randn(19, 3)

    calibrator = ReliabilityCalibrator()

    with pytest.raises(ValueError):
        calibrator.fit(X, pool_preds)


def test_reliability_score_before_fit_raises_error():
    X = np.random.randn(10, 4)
    pool_preds = np.random.randn(10, 3)

    calibrator = ReliabilityCalibrator()

    with pytest.raises(RuntimeError):
        calibrator.score(X, pool_preds)
