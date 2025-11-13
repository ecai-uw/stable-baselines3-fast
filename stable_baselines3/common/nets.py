from typing import Any, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import Schedule

class ContinuousValueNet(BaseModel):
    """
    Fill this in.
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            lr_schedule: Schedule,
            net_arch: list[int],
            features_extractor_class: type[BaseFeaturesExtractor],
            features_extractor_kwargs: Optional[dict[str, Any]] = None,
            activation_fn: type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[dict[str, Any]] = None,
            post_linear_modules: Optional[list[type[nn.Module]]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        # Create feature extractor.
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim

        # Create the value network.
        self.value_net = nn.Sequential(
            *create_mlp(self.features_dim, 1, net_arch, activation_fn, post_linear_modules)
        )
        self.add_module("value_net", self.value_net)

        # Create optimizer.
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
    
    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        value = self.value_net(features)
        return value