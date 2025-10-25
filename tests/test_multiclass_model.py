import pytest
from tests.helpers import _run_unseen_evaluation

@pytest.mark.integration
def test_unseen_multiclass(multiclass_model_under_test):
    _run_unseen_evaluation(multiclass_model_under_test, multiclass_model_under_test.multi_predict)