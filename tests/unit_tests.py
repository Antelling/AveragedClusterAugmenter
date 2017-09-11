from ACA import ACATransformer, ACAWrapper
from sklearn.datasets import load_boston
from sklearn.cluster import KMeans, Birch
from sklearn.linear_model import LinearRegression
import numpy as np

np.random.seed(1)

X, y = load_boston(return_X_y=True)


def test_clusterer_param():
    transformer = ACATransformer(clusterer=KMeans(n_clusters=3), return_old=False)
    new_X_1 = transformer.fit_transform(X)
    new_X_1_target = [
        [0.02730999999999995, 0.0, 7.070000000000018, 0.0, 0.46899999999999664, 6.421000000000008, 78.90000000000047,
         4.9671000000000225, 2.0, 242.0, 17.800000000000114, 396.8999999999973, 9.139999999999977],
        [0.02730999999999996, 0.0, 7.070000000000011, 0.0, 0.4690000000000007, 6.4210000000000065, 78.89999999999985,
         4.967100000000005, 2.0, 242.0, 17.79999999999997, 396.9000000000007, 9.139999999999988],
        [0.027309999999999973, 0.0, 7.069999999999996, 0.0, 0.46899999999999986, 6.4209999999999985, 78.90000000000005,
         4.967099999999997, 2.0, 242.0, 17.799999999999994, 396.8999999999998, 9.139999999999992]]

    new_X_2, new_y_2 = transformer.fit_transform(X, y)  # the inclusion of y changes what clusters are formed
    new_X_2_target = [[1.09105113e+01, 0.00000000e+00, 1.85725490e+01,
                       7.84313725e-02, 6.71225490e-01, 5.98226471e+00,
                       8.99137255e+01, 2.07716373e+00, 2.30196078e+01,
                       6.68205882e+02, 2.01950980e+01, 3.71803039e+02,
                       1.78740196e+01],
                      [3.74992678e-01, 1.57103825e+01, 8.35953552e+00,
                       7.10382514e-02, 5.09862568e-01, 6.39165301e+00,
                       6.04133880e+01, 4.46074481e+00, 4.45081967e+00,
                       3.11232240e+02, 1.78177596e+01, 3.83489809e+02,
                       1.03886612e+01],
                      [1.49558803e+01, 0.00000000e+00, 1.79268421e+01,
                       2.63157895e-02, 6.73710526e-01, 6.06550000e+00,
                       8.99052632e+01, 1.99442895e+00, 2.25000000e+01,
                       6.44736842e+02, 1.99289474e+01, 5.77863158e+01,
                       2.04486842e+01]]
    new_y_2_target = [17.42941176, 24.93169399, 13.12631579]

    assert np.allclose(new_X_1, new_X_1_target)
    assert np.allclose(new_X_2, new_X_2_target)
    assert np.allclose(new_y_2, new_y_2_target)

    # now we change the clusterer and redo all the tests
    transformer.clusterer = Birch(n_clusters=3)

    new_X_1 = transformer.fit_transform(X)
    new_X_1_target = [
        [0.02730999999999995, 0.0, 7.07000000000002, 0.0, 0.46899999999999664, 6.4210000000000065, 78.90000000000049,
         4.9671000000000225, 2.0, 242.0, 17.800000000000114, 396.89999999999725, 9.139999999999977],
        [0.02730999999999996, 0.0, 7.070000000000012, 0.0, 0.4690000000000007, 6.421000000000007, 78.89999999999985,
         4.967100000000006, 2.0, 242.0, 17.79999999999997, 396.9000000000008, 9.139999999999988],
        [0.02730999999999998, 0.0, 7.069999999999996, 0.0, 0.4689999999999998, 6.420999999999999, 78.90000000000005,
         4.9670999999999985, 2.0, 242.0, 17.8, 396.8999999999998, 9.139999999999993]]

    new_X_2, new_y_2 = transformer.fit_transform(X, y)  # the inclusion of y changes what clusters are formed
    new_X_2_target = [
        [0.3887744444444444, 15.582655826558266, 8.420894308943073, 0.07317073170731707, 0.5118474254742559,
         6.388005420054201, 60.632249322493216, 4.441271544715446, 4.455284552845528, 311.9268292682927,
         17.8092140921409, 381.04257452574467, 10.41745257452574],
        [11.115060096153847, 0.0, 18.56346153846151, 0.07692307692307693, 0.6697980769230766, 5.982673076923073,
         90.10769230769226, 2.0665923076923076, 23.03846153846154, 668.1634615384615, 20.195192307692334,
         368.406153846154, 17.91355769230769],
        [15.727845454545452, 0.0, 18.10000000000001, 0.0, 0.671060606060606, 6.0803939393939395, 89.52727272727273,
         2.0162666666666667, 24.0, 666.0, 20.200000000000003, 47.21545454545455, 21.072727272727274]]
    new_y_2_target = [24.857181571815715, 17.515384615384615, 12.354545454545454]

    assert np.allclose(new_X_1, new_X_1_target)
    assert np.allclose(new_X_2, new_X_2_target)
    assert np.allclose(new_y_2, new_y_2_target)


def test_return_old():
    t_old_true = ACATransformer(return_old=True, clusterer=KMeans(n_clusters=2))
    t_old_false = ACATransformer(return_old=False, clusterer=KMeans(n_clusters=2))

    augmented = t_old_true.fit_transform(X)
    replaced = t_old_false.fit_transform(X)

    assert len(augmented) == len(X) + 2
    assert len(replaced) == 2


def test_percentage():
    transformer = ACATransformer(return_old=False)

    for percentage in [.1, .4, 1]:
        transformer.percentage = percentage
        new_X = transformer.fit_transform(X)
        assert len(new_X) == int(percentage * len(X))


# ACAWrapper uses the same code paths as ACATransformer for the clusterer, percentage, and return_old params.

def test_wrapper_estimator():
    wrapper = ACAWrapper()  # defaults to SVR

    train_X = X[0:50]
    train_y = y[0:50]

    test_X = X[50:53]
    target_y = [19.68491156, 19.68491156, 19.68491156]

    wrapper.fit(train_X, train_y)
    assert np.allclose(wrapper.predict(test_X), target_y)

    wrapper = ACAWrapper(estimator=LinearRegression())

    target_y = [18.08602901, 17.63156814, 23.30413024]

    wrapper.fit(train_X, train_y)
    assert np.allclose(wrapper.predict(test_X), target_y)

def test_wrapper_disabled():
    working = ACAWrapper(clusterer=LinearRegression()) # broken clusterer
    disabled = ACAWrapper(disabled=True, clusterer=LinearRegression()) #so this is just a normal SVR

    errored = False
    try:
        working.fit(X, y)
    except TypeError:
        errored = True
    assert errored #check that the non disabled throws and error

    disabled.fit(X, y) #and make sure that disabled doesn't
    disabled.predict(X[2:3])


test_clusterer_param()
test_return_old()
test_percentage()
test_wrapper_estimator()
test_wrapper_disabled()
