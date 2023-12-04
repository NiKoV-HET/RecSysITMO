import os
import pickle
from typing import List, Optional

import pandas as pd


class BaseModel:
    def __init__(self, path_to_model: Optional[str] = None) -> None:
        self.model_loaded: bool = False
        self.model = None

        if path_to_model and os.path.exists(path_to_model):
            try:
                with open(path_to_model, "rb") as file:
                    self.model = pickle.load(file)
                    self.model_loaded = True
            except Exception as e:
                print(f"Error loading model: {e}")

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
    def recommend(self, k_recs=10, *args, **kwargs):
        return self.mock_predict(k_recs)


class PopularModel(BaseModel):
    def __init__(self, path_to_recs: str = "service/models/pop_recs.pkl") -> None:
        super().__init__(path_to_recs)
        self.pop_recs_list: List[int] = []

        if self.model_loaded:
            self.pop_recs_list = list(self.model["item_id"].unique())

    def recommend(self, k_recs: int = 10, *args, **kwargs) -> List[int]:
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
        if self.model_loaded and hasattr(self.model, "users_mapping") and hasattr(self.model, "predict"):
            if user_id in self.model.users_mapping:
                recs = self.model.predict(pd.DataFrame([user_id], columns=["user_id"]))["item_id"].to_list()[:k_recs]
            else:
                recs = []

            if fill_empty_recs and popular_model:
                recs = self.fill_recs_popular(recs, k_recs, popular_model.recommend())
            return recs
        return self.mock_predict(k_recs)
