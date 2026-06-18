def test_new_estimators_public_api_imports():
    from mq_cenn import (
        MQCeNNMultiStepRegressor,
        MQCeNNAnomalyDetector,
        make_multistep_windows,
    )

    assert MQCeNNMultiStepRegressor is not None
    assert MQCeNNAnomalyDetector is not None
    assert make_multistep_windows is not None


def test_direct_estimator_imports():
    from mq_cenn.estimators.multistep import MQCeNNMultiStepRegressor
    from mq_cenn.estimators.anomaly import MQCeNNAnomalyDetector

    assert MQCeNNMultiStepRegressor is not None
    assert MQCeNNAnomalyDetector is not None
