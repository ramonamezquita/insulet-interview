import numpy as np
import skimage
from skimage.feature import hog
from skimage.io import imread
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from tqdm.auto import tqdm
import pandas as pd


class ImageResizer(BaseEstimator, TransformerMixin):
    """Resizes images to given output shape.
    """

    def __init__(self, output_shape, anti_aliasing=False):
        self.output_shape = output_shape
        self.anti_aliasing = anti_aliasing

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return skimage.transform.resize(X, self.output_shape,
                                        anti_aliasing=self.anti_aliasing)


class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """Converts an array of RGB images to grayscale
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if X.ndim == 2:
            return X
        return skimage.color.rgb2gray(X)


class HogTransformer(BaseEstimator, TransformerMixin):
    """Calculates hog features.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return hog(X, **self.kwargs)


class AlphaChannelRemover(BaseEstimator, TransformerMixin):
    """Removes alpha channel.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.has_alpha(X):
            return X[:, :, :3]
        return X

    def has_alpha(self, X):
        return X.shape[-1] == 4


class FromPathImageTransformer(BaseEstimator, TransformerMixin):
    """Applies transformation to a collection of image paths.
    """

    def __init__(self, transformer, root_path='', stack_output=False):
        self.transformer = transformer
        self.root_path = root_path
        self.stack_output = stack_output

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        imgs = []
        for path in tqdm(X, desc="Image transforms",
                         bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            im = self.read_image(path)
            im = self.transform_image(im)
            imgs.append(im)

        if self.stack_output:
            return np.stack(imgs)
        return imgs

    def transform_image(self, img):
        return self.transformer.transform(img)

    def read_image(self, path):
        full_path = self.root_path + path
        return imread(full_path)


class UnitCircleProjector(BaseEstimator, TransformerMixin):
    """Unit circle projector.

    Projects X into a unit circle by computing its sine and cosine
    transformations.
    """

    def __init__(self, period):
        self.period = period

        self._sin_transformer = FunctionTransformer(
            lambda x: np.sin(x / period * 2 * np.pi))
        self._cos_transformer = FunctionTransformer(
            lambda x: np.cos(x / period * 2 * np.pi))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sin = self._sin_transformer.fit_transform(X)
        cos = self._cos_transformer.fit_transform(X)
        return np.column_stack((sin, cos))


class CyclicalDatesEncoding(BaseEstimator, TransformerMixin):
    """Trigonometric encoding of datetime features.
    """

    def __init__(self, datetime_attrs=('day', 'month', 'dayofweek')):
        self.datetime_attrs = datetime_attrs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.to_datetime(X)
        projections = []
        for attr in self.datetime_attrs:
            x = getattr(X.dt, attr)
            x = UnitCircleProjector(period=x.max()).fit_transform(x)
            projections.append(x)

        return np.hstack(projections)
