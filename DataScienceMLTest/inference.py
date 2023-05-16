from requests import Session


class MlflowEndpoint:
    """Real-time inference in locally deployed Mlflow model.

    Parameters
    ----------
    endpoint_url : str
        Deployment endpoint

    http_session : requests.Session
        Http session.
    """

    def __init__(self, endpoint_url, http_session=Session()):
        self.endpoint_url = endpoint_url
        self.http_session = http_session

        self._headers = {'accept': 'application/json',
                         'Content-Type': 'application/json'}

    def post(self, X):
        split_data = X.to_dict(orient='split')
        json = {"dataframe_split": split_data}
        return self._post(json)

    def _post(self, json):
        return self.http_session.post(self.endpoint_url, json=json,
                                      headers=self._headers)
