"""Training python script.
"""

import argparse

import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.compose import make_column_transformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import transformers

parser = argparse.ArgumentParser(
    description='Trains gradient boosting regressor.'
)
parser.add_argument(
    '-r', '--run',
    default=None,
    type=str,
    help='Mlflow run name'
)
parser.add_argument(
    '-n', '--n_estimators',
    default=100,
    type=int,
    help='Number of estimators in gradient boosting'
)
args = parser.parse_args()


def read_csv(filepath):
    """Auxiliary for reading csv files from project root.
    """
    csv = pd.read_csv(filepath)
    csv['date'] = pd.to_datetime(csv['date'])
    csv = csv.sort_values('date').reset_index(drop=True)
    return csv


def log_model(sk_model, X):
    """Auxiliary for logging sklearn model to mlflow.
    """
    signature = infer_signature(
        model_input=X,
        model_output=sk_model.predict(X.head())
    )

    mlflow.sklearn.log_model(
        sk_model=sk_model,
        artifact_path='estimator',
        signature=signature
    )


# Create image transformer.
# The resulting image transformer (:class:`FromPathImageTransformer`) takes
# filepaths as input and processes the images using whatever transformer is
# given through its ``transformer`` parameter. For this case, the
# ``transformer`` is itself a sklearn :class:`Pipeline` containing 3 image
# transformations: RGB2GrayTransformer, ImageResizer, HogTransformer.
image_pipeline = make_pipeline(
    *[
        transformers.AlphaChannelRemover(),
        transformers.RGB2GrayTransformer(),
        transformers.ImageResizer(output_shape=(200, 200), anti_aliasing=True),
        transformers.HogTransformer()
    ]
)
image_transformer = transformers.FromPathImageTransformer(
    transformer=image_pipeline, stack_output=True)

# Create preprocessor, a sklearn :class:`Pipeline` containing all data
# transformations. In addition to the image transformer previously mentioned,
# "dates" will be cyclically encoded and "image classes" will be one hot
# encoded.
transformers = [
    (image_transformer, 'image'),
    (transformers.CyclicalDatesEncoding(), 'date'),
    (OneHotEncoder(), ['image class'])
]
column_trans = make_column_transformer(*transformers, remainder='passthrough')
preprocessor = make_pipeline(column_trans, MinMaxScaler())

# Now, we can bring everything together in a new :class:``Pipeline`` called
# ``estimator`` containing the following:
# - preprocessor: Transforms data prior to fit.
# - selector: Selects relevant features.
# - regressor: Fits data.
estimator = Pipeline(
    [
        ('preprocessor', preprocessor),
        ('selector', SelectFromModel(ExtraTreesRegressor(n_estimators=10))),
        ('regressor', GradientBoostingRegressor(n_estimators=args.n_estimators))
    ]
)

# Fit estimator.
with mlflow.start_run(run_name=args.run):
    X = read_csv('training.csv')
    y = X.pop('target')
    estimator.fit(X, y)
    log_model(estimator, X)
