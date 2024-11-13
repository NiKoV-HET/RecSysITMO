from typing import Any, Dict, List

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from service.api.exceptions import ModelNotFoundError, ModelNotImplementedError, UserNotFoundError
from service.log import app_logger
from service.models_rec import (
    ALSANNOnlineModel,
    AutoencoderOfflineModel,
    DSSMMOfflineModel,
    KNNOfflineModel,
    KNNOnlineModel,
    PopularModel,
    RangeModel,
)


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


models: Dict[str, Any] = {
    "random": RangeModel(),
    "model_1": None,
    "popular": PopularModel(),
    "knn_online": KNNOnlineModel("service/models/userknn_model.pkl"),
    "knn_offline": KNNOfflineModel("service/models/userknn_predect_offline.pkl"),
    "als_ann_online": ALSANNOnlineModel("service/models/ALS_Online.pkl"),
    "dssm_offline": DSSMMOfflineModel("service/models/reco_dssm.csv"),
    "autoencoder_offline": AutoencoderOfflineModel("service/models/autoencoder_offline.csv"),
}


router = APIRouter()
auth_scheme = HTTPBearer(auto_error=False)


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        200: {"description": "Return recommendations for users."},
        401: {"description": "You are not authenticated"},
        404: {"description": "The Model or User was not found"},
        501: {"description": "The Model not implemented"},
    },
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
) -> RecoResponse:
    if not token or token.credentials != request.app.state.api_key:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    if model_name not in models:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    k_recs = request.app.state.k_recs

    model = models[model_name]
    if model:
        reco = model.recommend(user_id=user_id, k_recs=k_recs, popular_model=models["popular"])
    else:
        raise ModelNotImplementedError(error_message=f"Model {model_name} not implemented")

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
