import os
import pickle
from typing import Any

import pandas as pd


class PopularModel:
    pop_recs = []
    model_loaded = False

    def __init__(self, path_to_recs="service/models/pop_recs.pkl") -> None:
        if os.path.exists(path_to_recs):
            with open(path_to_recs, "rb") as file:
                pop_recs = pickle.load(file)
                self.model_loaded = True
            self.pop_recs_list = list(pop_recs["item_id"].unique())

    def __call__(self, _="", k_recs=10) -> Any:
        if self.model_loaded:
            return self.pop_recs_list[:k_recs]
        return list(range(k_recs))[:k_recs]


class BaseModel:
    popular_recs = PopularModel()()

    def fill_recs_popular(self, recs, k_recs) -> list:
        if k_recs - len(recs) > 0:
            new_recs = filter(lambda x: x not in recs, self.popular_recs)
            recs.extend(new_recs)
        return recs[:k_recs]

    def mok_predict(self, k_recs) -> list:
        return list(range(k_recs))


class KNNOfflineModel(BaseModel):
    model_loaded = False

    def __init__(self, path_to_recs) -> None:
        if os.path.exists(path_to_recs):
            with open(path_to_recs, "rb") as file:
                self.userknn_predect_result = pickle.load(file)
                self.model_loaded = True

    def __call__(self, user_id, k_recs=10, fill_empty_recs=True) -> list:
        if self.model_loaded:
            if user_id in self.userknn_predect_result:
                recs = self.userknn_predect_result[user_id][:k_recs]
            else:
                recs = []

            if fill_empty_recs:
                recs = self.fill_recs_popular(recs, k_recs)

            return recs
        return self.mok_predict(k_recs)


class KNNOnlineModel(BaseModel):
    model_loaded = False

    def __init__(self, path_to_model) -> None:
        if os.path.exists(path_to_model):
            with open(path_to_model, "rb") as file:
                self.userknn_model = pickle.load(file)
                self.model_loaded = True

    def __call__(self, user_id, k_recs=10, fill_empty_recs=True) -> list:
        if self.model_loaded:
            if user_id in (self.userknn_model.users_mapping):
                recs = self.userknn_model.predict(pd.DataFrame([user_id], columns=["user_id"]))["item_id"].to_list()[
                    :k_recs
                ]
            else:
                recs = []

            if fill_empty_recs:
                recs = self.fill_recs_popular(recs, k_recs)

            return recs
        return self.mok_predict(k_recs)


if __name__ == "__main__":
    user_id_t = 21
    k_recs_t = 10
    validate_input = [1, 2, 3, 4, 5]

    popular_model = PopularModel()
    knn_offline_model = KNNOfflineModel("service/models/userknn_predect_offline.pkl")
    knn_online_model = KNNOnlineModel("service/models/userknn_model.pkl")
    base_model = BaseModel()
    base_result = base_model.fill_recs_popular(validate_input, k_recs_t)

    print(f"[KNN online] k_recs:{k_recs_t}, user_id:{user_id_t}. Output: {knn_online_model(user_id_t, k_recs_t)}")
    print(f"[KNN offline] k_recs:{k_recs_t}, user_id:{user_id_t}. Output: {knn_offline_model(user_id_t, k_recs_t)}")
    print(f"[Popular] k_recs:{k_recs_t}, user_id:{user_id_t}. Output: {popular_model(k_recs_t)}")
    print(f"[Validate] input {validate_input}, k_recs:{k_recs_t}. Output:{base_result}")
