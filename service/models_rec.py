import json
import os
import pickle
from typing import List, Optional

import nmslib
import pandas as pd
from lightfm import LightFM
from rectools.tools.ann import UserToItemAnnRecommender

from userknn import UserKnn


class BaseModel:
    def __init__(self, path_to_model: Optional[str] = None) -> None:
        self.model_loaded: bool = False
        self.model = None

        if path_to_model and os.path.exists(path_to_model):
            try:
                if ".pkl" in path_to_model:
                    with open(path_to_model, "rb") as file:
                        self.model = pickle.load(file)
                        self.model_loaded = True
                elif ".csv" in path_to_model:
                    self.model = pd.read_csv(path_to_model)
                    self.model_loaded = True
            except IOError:
                print(f"I/O error occurred trying to open: {path_to_model}")

    def fill_recs_popular(self, recs: List[int], k_recs: int, popular_recs: List[int]) -> List[int]:
        recs = recs.copy()
        needed = k_recs - len(recs)
        if needed > 0:
            new_recs = [rec for rec in popular_recs if rec not in recs][:needed]
            recs.extend(new_recs)
        return recs

    def mock_predict(self, k_recs: int) -> List[int]:
        return list(range(k_recs))


class RangeModel(BaseModel):
    def recommend(self, *args, k_recs=10, **kwargs):
        return self.mock_predict(k_recs)


class PopularModel(BaseModel):
    def __init__(self, path_to_recs: str = "service/models/pop_recs.pkl") -> None:
        super().__init__(path_to_recs)
        self.pop_recs_list: List[int] = []

        if self.model_loaded:
            self.pop_recs_list = list(self.model["item_id"].unique())

    def recommend(self, *args, k_recs: int = 10, **kwargs) -> List[int]:
        return self.pop_recs_list[:k_recs] if self.model_loaded else self.mock_predict(k_recs)


class KNNOfflineModel(BaseModel):
    def recommend(
        self, user_id: int, k_recs: int = 10, fill_empty_recs: bool = True, popular_model: PopularModel = None
    ) -> List[int]:
        if self.model_loaded and isinstance(self.model, dict):
            recs = self.model.get(user_id, [])[:k_recs]
            if fill_empty_recs and popular_model:
                recs = self.fill_recs_popular(recs, k_recs, popular_model.recommend())
            return recs
        return self.mock_predict(k_recs)


class KNNOnlineModel(BaseModel):
    def recommend(
        self, user_id: int, k_recs: int = 10, fill_empty_recs: bool = True, popular_model: PopularModel = None
    ) -> List[int]:
        if self.model_loaded and isinstance(self.model, UserKnn):
            if user_id in self.model.users_mapping:
                recs = self.model.predict(pd.DataFrame([user_id], columns=["user_id"]))["item_id"].to_list()[:k_recs]
            else:
                recs = []

            if fill_empty_recs and popular_model:
                recs = self.fill_recs_popular(recs, k_recs, popular_model.recommend())
            return recs
        return self.mock_predict(k_recs)


class LightFMOnlineModel(BaseModel):
    def __init__(self, path_to_recs: str) -> None:
        super().__init__(path_to_recs)
        self.model = self.model.model

    def recommend(
        self, user_id: int, k_recs: int = 10, fill_empty_recs: bool = True, popular_model: PopularModel = None
    ) -> List[int]:
        if self.model_loaded and isinstance(self.model, LightFM):
            if user_id in self.model.users_mapping:
                recs = self.model.predict(pd.DataFrame([user_id], columns=["user_id"]))["item_id"].to_list()[:k_recs]
            else:
                recs = []

            if fill_empty_recs and popular_model:
                recs = self.fill_recs_popular(recs, k_recs, popular_model.recommend())
            return recs
        return self.mock_predict(k_recs)


class ALSANNOnlineModel(BaseModel):
    def __init__(self, path_to_recs: str) -> None:
        super().__init__(path_to_recs)
        self.model_loaded = False

        path_to_user_id_map = "service/models/als_user_id_map.pkl"
        path_to_item_id_map = "service/models/als_item_id_map.pkl"
        if os.path.exists(path_to_user_id_map) and os.path.exists(path_to_item_id_map):
            self.user_vectors, self.item_vectors = self.model.get_vectors()
            with open(path_to_user_id_map, "rb") as file:
                self.user_id_map = pickle.load(file)
            with open(path_to_item_id_map, "rb") as file:
                self.item_id_map = pickle.load(file)

            index_init_params = {"method": "hnsw", "space": "negdotprod", "data_type": nmslib.DataType.DENSE_VECTOR}
            self.ann = UserToItemAnnRecommender(
                user_vectors=self.user_vectors,
                item_vectors=self.item_vectors,
                user_id_map=self.user_id_map,
                item_id_map=self.item_id_map,
                index_init_params=index_init_params,
            )

            self.ann.index.loadIndex("service/models/als_ann_index")
            self.model_loaded = True

    def recommend(
        self, user_id: int, k_recs: int = 10, fill_empty_recs: bool = True, popular_model: PopularModel = None
    ) -> List[int]:
        if self.model_loaded:
            if user_id in self.user_id_map.external_ids:
                recs = self.ann.get_item_list_for_user(user_id, k_recs).tolist()
            else:
                recs = []

            if fill_empty_recs and popular_model:
                recs = self.fill_recs_popular(recs, k_recs, popular_model.recommend())
            return recs
        return self.mock_predict(k_recs)


class DSSMMOfflineModel(BaseModel):
    def __init__(self, path_to_recs: str) -> None:
        super().__init__(path_to_recs)
        if self.model_loaded:
            self.model["item_id"] = self.model["item_id"].apply(json.loads)
            self.model = self.model.set_index("user_id").to_dict()["item_id"]

    def recommend(
        self, user_id: int, k_recs: int = 10, fill_empty_recs: bool = True, popular_model: PopularModel = None
    ) -> List[int]:
        if self.model_loaded:
            if user_id in self.model:
                recs = self.model.get(user_id, [])[:k_recs]
            else:
                recs = []

            if fill_empty_recs and popular_model:
                recs = self.fill_recs_popular(recs, k_recs, popular_model.recommend())
            return recs
        return self.mock_predict(k_recs)


class AutoencoderOfflineModel(BaseModel):
    def __init__(self, path_to_recs: str) -> None:
        super().__init__(path_to_recs)
        if self.model_loaded:
            self.model["item_id"] = self.model["item_id"].apply(json.loads)
            self.model = self.model.set_index("user_id").to_dict()["item_id"]

    def recommend(
        self, user_id: int, k_recs: int = 10, fill_empty_recs: bool = True, popular_model: PopularModel = None
    ) -> List[int]:
        if self.model_loaded:
            if user_id in self.model:
                recs = self.model.get(user_id, [])[:k_recs]
            else:
                recs = []

            if fill_empty_recs and popular_model:
                recs = self.fill_recs_popular(recs, k_recs, popular_model.recommend())
            return recs
        return self.mock_predict(k_recs)
