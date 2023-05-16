# Files description

In addition to the original files, please verify the followings
also exist:

1. README.md (this file)
2. train.py
3. transformers.py
4. notebooks (folder)
5. mlruns (folder)
6. mlartifacts (folder)

### 2. train.py

Contains the training script. To execute it, simply do

````
python train.py -r <mlflow_run_name> -n <n_estimators>
````

### 3. transformers.py

Contains a collection of custom sklearn transformers used inside the
training script. For example,

```python
from sklearn.base import BaseEstimator, TransformerMixin


class CustomTransformer(BaseEstimator, TransformerMixin):
    """Custom transformation.
    """

    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def fit(self, X, y=None):
        ...

    def transform(self, X):
        ...
```

### 4. notebooks (folder)

Contains experimental notebooks that show the thought process for arriving to
the final solution.

### 5. mlruns

Contains data for mlflow runs. Initially, it only contains data for a single
run with ``run_id=620014e9c50548a9bd060d8e7309dfc7``, which corresponds to
the final model from which the final predictions where obtained.
To deploy it locally, start a bash session 
(``docker exec -it <container_name> bash``) and do

````
cd DataScienceMLTest/
mlflow models serve -m runs:/620014e9c50548a9bd060d8e7309dfc7/estimator -p <port>
````
**_NOTE:_** 📝  Every ``mlflow models serve ...`` command must be executed from 
inside the **DataScienceMLTest** directory.

**_NOTE:_** 📝  Choose a port number different from *5000* since it is already 
in use by the Mlflow UI.


### 6. mlartifacts

Contains mlflow artifacts.

# Real time inference
The `inference.py` python module contains the `MlflowEndpoint` class used to
obtain predictions from a locally deployed mlflow model.

```python
import pandas as pd
from inference import MlflowEndpoint

# Create instance of :class:`MlflowEndpoint'
port = 5001
endpoint_url = f'http://localhost:{port}/invocations'
endpoint = MlflowEndpoint(endpoint_url)

# Read input data.
X = pd.read_csv('training.csv')

# Request predictions to locally deployed mlflow model.
http = endpoint.post(X.head())
http.json()
```














