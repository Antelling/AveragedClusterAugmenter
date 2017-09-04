from ACA import ACATransformer, ACAWrapper
from sklearn.utils.estimator_checks import check_estimator

check_estimator(ACAWrapper)

print("compatibility checks passed")