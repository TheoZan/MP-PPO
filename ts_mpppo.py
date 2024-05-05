from typing import (Any, ClassVar, Dict, Optional, Tuple, Type, TypeVar, Union,
                    Callable, List, NamedTuple, Generator, Iterable)

import sys
import time
import datetime
from collections import deque
import os
import warnings
import pathlib
import io
from functools import partial, lru_cache

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px

import statistics
import random
import math
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from evaluate import load
from accelerate import Accelerator


from transformers import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
    PretrainedConfig)
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import (
    get_lags_for_frequency, 
    time_features_from_frequency_str,
    TimeFeature,
    get_seasonality)
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)
from gluonts.transform.sampler import InstanceSampler
from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches

import gym
from gym.spaces import Box
import gymnasium
from gymnasium import spaces


from citylearn.data import Pricing
from citylearn.citylearn import CityLearnEnv
import citylearn
from citylearn.energy_model import HeatPump
from citylearn.utilities import read_json
from reward_function import RampingCost
from citylearn.reward_function import RewardFunction

from stable_baselines3.common.vec_env.patch_gym import _convert_space
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common import utils
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import (BaseCallback, CallbackList,
                                            ConvertCallback, ProgressBarCallback)
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (explained_variance, get_schedule_fn, 
                                            obs_as_tensor, safe_mean, constant_fn,
                                            check_for_correct_spaces)
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.vec_env.patch_gym import _convert_space
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import is_wrapped, _patch_env
from stable_baselines3.common.vec_env import DummyVecEnv, is_vecenv_wrapped, VecTransposeImage
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space

from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC, DQN



from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer, MaskableRolloutBuffer
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from sb3_contrib.ppo_mask.policies import MlpPolicy

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO, RecurrentPPO
from sb3_contrib.common.maskable.utils import get_action_masks

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device, get_schedule_fn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

SelfMaskablePPO = TypeVar("SelfMaskablePPO", bound="MaskablePPO")
SelfBaseAlgorithm = TypeVar("SelfBaseAlgorithm", bound="BaseAlgorithm")
SelfLstmMaskablePPO = TypeVar("SelfLstmMaskablePPO", bound="LstmMaskablePPO")


from ts_mppo import *
import ts_transformer as tst


class ConcatLayer(nn.Module):
    def __init__(self, concat_output_size, dim=2):
        super().__init__()
        self.concat_output_size = concat_output_size
        self.dim = dim

    def forward(self, x, y):
        res = th.cat((x, y), dim=self.dim)
        return res.view(-1, self.concat_output_size)

class LSTMWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMWrapper, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        output, _ = self.lstm(x)
        return output

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, output_type):
        # print(x.shape)
        output, (hidden, cell) = self.lstm(x)
        if output_type == 'output':
            output = self.linear(output[:, -1, :])
            return output
        elif output_type == 'hidden':
            return hidden[-1]
        else:
            raise ValueError("Invalid output_type. Choose 'output', 'hidden', or 'cell'.")

    def save(self, checkpoint_path):
        th.save(self.state_dict(), checkpoint_path)
        print(f"Model weights saved to {checkpoint_path}")

    def load(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))
        print(f"Model weights loaded from {checkpoint_path}")

class LstmActorCriticNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        share_actor_critic: bool = False,
        lstm_output_size: int = 5,
        lstm_hidden_size: int = 200,
        lstm_num_layers: int = 2,
        use_mlp_extractor: bool = True, 
        extractor_sizes: Optional[List[int]] = [64, 64],
        use_lstm_extractor: bool = False,
        lstm_extractor_params: Optional[Tuple[int]] = (64, 2 ,64), #hiden_size, n_layer, output_size
        activation: str = 'relu',
        *args,
        **kwargs,

    ):
        super().__init__()
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.latent_dim_lstm = lstm_hidden_size
        self.lstm_output_size = lstm_output_size
        self.use_mlp_extractor = use_mlp_extractor

        #shared networks
        self.share_actor_critic = share_actor_critic

        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function. Choose 'relu' or 'tanh'.")

        # LSTM forecast
        # LSTM Layer
        self.lstm = LSTMModel(feature_dim, lstm_hidden_size, lstm_num_layers, lstm_output_size)

        # Concatenated input size for MLP
        concat_output_size = feature_dim + lstm_output_size
        self.concat_layer = ConcatLayer(concat_output_size, dim=2) 

        # MLP feature extractor
        if self.use_mlp_extractor and extractor_sizes:
            mlp_layers = []
            last_size = concat_output_size
            for size in extractor_sizes:
                mlp_layers.append(nn.Linear(last_size, size))
                mlp_layers.append(self.activation)
                last_size = size
            self.mlp_extractor = nn.Sequential(*mlp_layers)

        if self.use_lstm_extractor and extractor_sizes:
            ex_hidden_size, ex_num_layers, ex_output_size = lstm_extractor_params
            self.mlp_extractor = LSTMModel(concat_output_size, ex_hidden_size, ex_num_layers, ex_output_size)


        # Policy network
        policy_input_size = extractor_sizes[-1] if self.use_mlp_extractor and extractor_sizes else concat_output_size
        self.policy_net = nn.Sequential(
            nn.Linear(policy_input_size, last_layer_dim_vf),
            nn.ReLU()
        )

        # Value network
        value_input_size = extractor_sizes[-1] if self.use_mlp_extractor and extractor_sizes else concat_output_size
        self.value_net = nn.Sequential(
            nn.Linear(value_input_size, last_layer_dim_pi),
            nn.ReLU()
        )

        if share_actor_critic:
            self.value_net = self.policy_net

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features), self.forward_forecast(features)

    def forward_lstm_and_concat(self, features: th.Tensor) -> th.Tensor:
        lstm_output = self.lstm(features, output_type='output')
        concatenated_output = self.concat_layer(lstm_output.unsqueeze(1), features)
        return concatenated_output

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        concatenated_output = self.forward_lstm_and_concat(features)
        if self.use_mlp_extractor and hasattr(self, 'mlp_extractor'):
            concatenated_output = self.mlp_extractor(concatenated_output)
        return self.policy_net(concatenated_output)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        concatenated_output = self.forward_lstm_and_concat(features)
        if self.use_mlp_extractor and hasattr(self, 'mlp_extractor'):
            concatenated_output = self.mlp_extractor(concatenated_output)
        return self.value_net(concatenated_output)

    def forward_forecast(self, features: th.Tensor) -> th.Tensor:
        # print('forward forecast: input', features.shape)
        return self.lstm(features, output_type='output')
        
    def set_requires_grad(self, value: bool, module=None):
        """
        Sets the requires_grad attribute for all parameters in the model or a specific module.
        If module is None, applies the operation to the whole model.

        :param value: Boolean value to set the requires_grad attribute.
        :param module: Specific module within the model to apply this operation. If None, applies to the whole model.
        """
        if module is None:
            for param in self.parameters():
                param.requires_grad = value
        else:
            for param in module.parameters():
                param.requires_grad = value

class LstmMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        custom_policy_parameters={},
        *args,
        **kwargs,
    ):
        self.custom_policy_parameters = custom_policy_parameters
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :param action_masks: Action masks to apply to the action distribution
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        features = features.unsqueeze(1)
        
        if self.share_features_extractor:
            latent_pi, latent_vf, forecast = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            forecast = self.mlp_extractor.forward_forecast(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, forecast
    
    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        #TODO change feature extracor to add dimension for lstm ?
        features = super().extract_features(obs, self.vf_features_extractor)
        features = features.unsqueeze(1)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def predict_forecast(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated forecast values given the observations.

        :param obs: Observation
        :return: forecasted values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        features = features.unsqueeze(1)
        forecast = self.mlp_extractor.forward_forecast(features)
        return forecast


    def get_distribution(self, obs: th.Tensor, action_masks: Optional[np.ndarray] = None) -> MaskableDistribution:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation
        :param action_masks: Actions' mask
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        features = features.unsqueeze(1)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = LstmActorCriticNetwork(self.features_dim, **self.custom_policy_parameters)

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        features = self.extract_features(obs)
        if features.dim() == 2:
            features = features.unsqueeze(1)
        #should be called in train -> batch size, no need to reshape (only if multiple env ?) TODO
        
        if self.share_features_extractor:
            latent_pi, latent_vf, forecast = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            forecast = self.mlp_extractor.forward_forecast(features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy(), forecast

@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)

def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch

def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    # a bit like torchvision.transforms.Compose
    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # step 2: convert the data to NumPy (potentially not needed)
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                # we expect an extra dim for the multivariate case:
                expected_ndim=1 if config.input_size == 1 else 2,
            ),
            # step 3: handle the NaN's by filling in the target with zero
            # and return the mask (which is in the observed values)
            # true for observed values, false for nan's
            # the decoder uses this mask (no loss is incurred for unobserved values)
            # see loss_weights inside the xxxForPrediction model
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # step 4: add temporal features based on freq of the dataset
            # month of year in the case when freq="M"
            # these serve as positional encodings
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            # step 5: add another temporal feature (just a single number)
            # tells the model where in its life the value of the time series is,
            # sort of a running counter
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # step 6: vertically stack all the temporal features into the key FEAT_TIME
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config.num_dynamic_real_features > 0
                    else []
                ),
            ),
            # step 7: rename to match HuggingFace names
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )

def create_train_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    #print('config', config)

    transformation = create_transformation(freq, config)
    # print('transformation', transformation)
    transformed_data = transformation.apply(data, is_train=True)
    # print('transformed_data', transformed_data)
    
    if cache_data:
        transformed_data = Cached(transformed_data)

    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, "train")

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from the 366 possible transformed time series)
    # randomly from within the target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(stream)
    
    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=th.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )

def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
    past_length = None,
) -> Transformation:
    assert mode in ["train", "validation", "test", "infer"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(min_future=config.prediction_length),
        "test": TestSplitSampler(),
        "infer": TestSplitSampler(),
    }[mode]

    if past_length is None:
        past_length = config.context_length if mode=='infer' else config.context_length + max(config.lags_sequence)

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=past_length,
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )

def create_test_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    mode = 'test',
    **kwargs,
):
    # print(mode)
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # we create a Test Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, mode)

    # we apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)

    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=th.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )

class TransformerWrapper:
    def __init__(self, config, freq):
        self.config = config
        self.freq = freq
        self.model = TimeSeriesTransformerForPrediction(config) #nn.module
        self.prediction_length = config.prediction_length

        # accelerator = Accelerator()
        # self.device = accelerator.device
        # self.model.to(device)
        print(self.model.device)

    def get_prediction(self, infer_df, mode='eval'):
        last_row_df = infer_df.tail(1)
        test_data = Dataset.from_pandas(last_row_df, preserve_index=True)
        test_data.set_transform(partial(transform_start_field, freq=self.freq))

        test_dataloader = create_test_dataloader(
            config=self.config,
            freq=self.freq,
            data=test_data,
            batch_size=1
        )
        data_actual = next(iter(test_dataloader))
        # print(data_actual)
        if mode == 'eval':
            self.model.eval()

        self.model.to('cpu')
        outputs = self.model.generate(
            static_categorical_features=data_actual["static_categorical_features"]
            if self.config.num_static_categorical_features > 0
            else None,
            static_real_features=data_actual["static_real_features"]
            if self.config.num_static_real_features > 0
            else None,
            past_time_features=data_actual["past_time_features"],
            past_values=data_actual["past_values"],
            future_time_features=data_actual["future_time_features"],
            past_observed_mask=data_actual["past_observed_mask"],
        )
        pred = np.vstack(outputs.sequences.cpu().numpy())
        pred = np.median(pred, 0)
        return pred

    @staticmethod
    def create_infer_dataset(week_of_year, building_type, climate_zone, list_value):
        infer_df = pd.DataFrame(columns=['target', 'start', 'feat_static_cat', 'feat_dynamic_real', 'item_id'])
        infer_df.loc[0] = [list_value, datetime.datetime(2017,1,1,0,0), [week_of_year, building_type, climate_zone], None, 'T1'] 
        return infer_df


class TransformerWrapper2:
    def __init__(self, trained_model, config, freq):
        self.config = config
        self.freq = freq
        self.model = trained_model #nn.module
        self.prediction_length = config.prediction_length

        # accelerator = Accelerator()
        # self.device = accelerator.device
        # self.model.to(device)
        print(self.model.device)

    def get_prediction(self, infer_df, mode='eval'):
        last_row_df = infer_df.tail(1)
        test_data = Dataset.from_pandas(last_row_df, preserve_index=True)
        test_data.set_transform(partial(transform_start_field, freq=self.freq))

        test_dataloader = create_test_dataloader(
            config=self.config,
            freq=self.freq,
            data=test_data,
            batch_size=1
        )
        data_actual = next(iter(test_dataloader))
        # print(data_actual)
        if mode == 'eval':
            self.model.eval()

        self.model.to('cpu')
        outputs = self.model.generate(
            static_categorical_features=data_actual["static_categorical_features"]
            if self.config.num_static_categorical_features > 0
            else None,
            static_real_features=data_actual["static_real_features"]
            if self.config.num_static_real_features > 0
            else None,
            past_time_features=data_actual["past_time_features"],
            past_values=data_actual["past_values"],
            future_time_features=data_actual["future_time_features"],
            past_observed_mask=data_actual["past_observed_mask"],
        )
        pred = np.vstack(outputs.sequences.cpu().numpy())
        pred = np.median(pred, 0)
        return pred

    @staticmethod
    def create_infer_dataset(week_of_year, building_type, climate_zone, list_value):
        infer_df = pd.DataFrame(columns=['target', 'start', 'feat_static_cat', 'feat_dynamic_real', 'item_id'])
        infer_df.loc[0] = [list_value, datetime.datetime(2017,1,1,0,0), [week_of_year, building_type, climate_zone], None, 'T1'] 
        return infer_df
    
class TransformerActorCriticNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        transformer: TransformerWrapper,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        share_actor_critic: bool = False,
        type_extractor: str = 'mlp', 
        extractor_sizes: Optional[Union[List[int], Tuple[int]]] = [64, 64], #(64,2,64) for lstm
        activation: str = 'relu',
        *args,
        **kwargs,

    ):
        super().__init__()
        assert type_extractor in [None, 'mlp', 'lstm'], 'type extractor must be in [None, mlp, lstm]'
        self._extractor = False

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.type_extractor = type_extractor
        self.transformer = transformer

        #shared networks
        self.share_actor_critic = share_actor_critic

        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function. Choose 'relu' or 'tanh'.")

        # Transformer layer
        self.transformer_layer = transformer.model

        # Concatenated input size for MLP
        concat_output_size = feature_dim + self.transformer.config.prediction_length
        self.concat_layer = ConcatLayer(concat_output_size, dim=2) 

        # MLP feature extractor
        if self.type_extractor == 'mlp' and extractor_sizes:
            self._extractor = True
            mlp_layers = []
            last_size = concat_output_size
            for size in extractor_sizes:
                mlp_layers.append(nn.Linear(last_size, size))
                mlp_layers.append(self.activation)
                last_size = size
            self.extractor = nn.Sequential(*mlp_layers)

        # LSTM feature extractor
        if self.type_extractor == 'lstm' and extractor_sizes:
            assert len(extractor_sizes) == 3, 'lstm parameters must be tuple of size 3 (hidden_size, num_layers, output_size)'
            self._extractor = True
            ex_hidden_size, ex_num_layers, ex_output_size = lstm_extractor_params
            self.extractor = LSTMModel(concat_output_size, ex_hidden_size, ex_num_layers, ex_output_size)

        # Policy network
        policy_input_size = extractor_sizes[-1] if self._extractor else concat_output_size
        self.policy_net = nn.Sequential(
            nn.Linear(policy_input_size, last_layer_dim_vf),
            nn.ReLU()
        )

        # Value network
        value_input_size = extractor_sizes[-1] if self._extractor else concat_output_size
        self.value_net = nn.Sequential(
            nn.Linear(value_input_size, last_layer_dim_pi),
            nn.ReLU()
        )

        if share_actor_critic:
            self.value_net = self.policy_net

    def forward(self, infer_df: pd.DataFrame, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(infer_df, features), self.forward_critic(infer_df, features), self.forward_forecast(infer_df)

    def forward_transformer_and_concat(self, infer_df: pd.DataFrame, features: th.Tensor) -> th.Tensor:
        transformer_output = self.transformer.get_prediction(infer_df)
        concatenated_output = self.concat_layer(transformer_output, features)
        return concatenated_output

    def forward_actor(self, infer_df: pd.DataFrame, features: th.Tensor) -> th.Tensor:
        concatenated_output = self.forward_transformer_and_concat(infer_df, features)
        if self.extractor_:
            concatenated_output = self.extractor(concatenated_output)
        return self.policy_net(concatenated_output)

    def forward_critic(self, infer_df: pd.DataFrame, features: th.Tensor) -> th.Tensor:
        concatenated_output = self.forward_transformer_and_concat(infer_df, features)
        if self.extractor_:
            concatenated_output = self.extractor(concatenated_output)
        return self.value_net(concatenated_output)

    def forward_forecast(self, infer_df: pd.DataFrame) -> th.Tensor:
        return self.transformer.get_prediction(infer_df)
        
    def set_requires_grad(self, value: bool, module=None):
        """
        Sets the requires_grad attribute for all parameters in the model or a specific module.
        If module is None, applies the operation to the whole model.

        :param value: Boolean value to set the requires_grad attribute.
        :param module: Specific module within the model to apply this operation. If None, applies to the whole model.
        """
        if module is None:
            for param in self.parameters():
                param.requires_grad = value
        else:
            for param in module.parameters():
                param.requires_grad = value

class TransformerMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        custom_policy_parameters={},
        *args,
        **kwargs,
    ):
        self.custom_policy_parameters = custom_policy_parameters
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def forward(
        self,
        infer_df,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :param action_masks: Action masks to apply to the action distribution
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        # features = features.unsqueeze(1)
        
        if self.share_features_extractor:
            latent_pi, latent_vf, forecast = self.mlp_extractor(infer_df, features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(infer_df, pi_features)
            latent_vf = self.mlp_extractor.forward_critic(infer_df, vf_features)
            forecast = self.mlp_extractor.forward_forecast(infer_df)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, forecast
    
    def predict_values(self, infer_df, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        #TODO change feature extracor to add dimension for lstm ?
        features = super().extract_features(obs, self.vf_features_extractor)
        # features = features.unsqueeze(1)
        latent_vf = self.mlp_extractor.forward_critic(infer_df, features)
        return self.value_net(latent_vf)

    def predict_forecast(self, infer_df) -> th.Tensor:
        """
        Get the estimated forecast values given the observations.

        :param obs: Observation
        :return: forecasted values.
        """
        # features = super().extract_features(obs, self.vf_features_extractor)
        # features = features.unsqueeze(1)
        forecast = self.mlp_extractor.forward_forecast(infer_df)
        return forecast


    def get_distribution(self, infer_df, obs: th.Tensor, action_masks: Optional[np.ndarray] = None) -> MaskableDistribution:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation
        :param action_masks: Actions' mask
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        # features = features.unsqueeze(1)
        latent_pi = self.mlp_extractor.forward_actor(infer_df, features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def _build_mlp_extractor(self) -> None:
        transformer = TransformerWrapper(self.custom_policy_parameters['config'], self.custom_policy_parameters['freq'])
        self.mlp_extractor = TransformerActorCriticNetwork(transformer, self.features_dim)

    def evaluate_actions(
        self,
        infer_df,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        features = self.extract_features(obs)
        if features.dim() == 2:
            features = features.unsqueeze(1)
        #should be called in train -> batch size, no need to reshape (only if multiple env ?) TODO
        
        if self.share_features_extractor:
            latent_pi, latent_vf, forecast = self.mlp_extractor(infer_df, features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(infer_df, pi_features)
            latent_vf = self.mlp_extractor.forward_critic(infer_df, vf_features)
            forecast = self.mlp_extractor.forward_forecast(infer_df)

        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy(), forecast

class LstmMaskableRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor
    real_forecast: th.Tensor

class LstmMaskableRolloutBuffer(MaskableRolloutBuffer):
    """
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        env_episode_length: int,
        forecast_dim: int,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        shuffle: bool = True,
    ):
        self.forecast_dim = forecast_dim
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        
        self.buffer_size = buffer_size
        self.real_buffer_size = buffer_size
        self.env_episode_length = env_episode_length

        self.dones = None
        self.real_forecast = None
        self.keep_mask = None

        self.shuffle = shuffle
        # self.counter = 0

    def reset(self) -> None:
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self.real_forecast = np.ones((self.buffer_size, self.n_envs, self.forecast_dim), dtype=np.float32)
        keep_mask = np.ones(len(self.real_forecast), dtype=bool)
        super().reset()

    def add(self, *args, done, forecast: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        """
        # populate dones buffer to be able to remove non full prediction instances of buffer
        if isinstance(done, list):
            done = done[0]
        if done:
            self.dones[self.pos] = 1
        
        # reset_counter = False

        if forecast is not None:
            lookback = self.forecast_dim
            if self.pos % self.env_episode_length in range(self.forecast_dim) and self.pos > self.forecast_dim:
                lookback = 1 + self.pos % self.env_episode_length
                # self.counter = 0

            for i in range(lookback):
                if self.pos - i >= 0: #CHECK IF SELF.POS == TIMESTEP TODO
                    self.real_forecast[self.pos - i, :, i] = forecast                

        # self.counter += 1
        super().add(*args, **kwargs)
    

    def remove_incomplete_rows(self):
        # print('Removing incomplete rows due to forecast window.\nRemoving on all buffer storage lines.')
        ending_pos = [i for i, value in enumerate(self.dones) if value]
        keep_mask = np.ones(len(self.real_forecast), dtype=bool)

        # Initialize a mask for the buffer
        for pos in ending_pos:
            # Calculate the start index for the sequence, ensuring it's not negative
            start_index = max(pos - self.forecast_dim + 2, 0)
            # Mark the relevant rows in the mask
            keep_mask[start_index:pos + 1] = 0
        self.keep_mask = keep_mask

        # Apply the mask to the buffer
        self.observations = self.observations[keep_mask]
        self.actions = self.actions[keep_mask]
        self.rewards = self.rewards[keep_mask]
        self.advantages = self.advantages[keep_mask]
        self.returns = self.returns[keep_mask]
        self.episode_starts = self.episode_starts[keep_mask]
        self.log_probs = self.log_probs[keep_mask]
        self.values = self.values[keep_mask]
        self.action_masks  = self.action_masks[keep_mask]
        self.real_forecast  = self.real_forecast[keep_mask]
        
        self.real_buffer_size = len(self.observations)

    def get(self, batch_size: Optional[int] = None) -> Generator[LstmMaskableRolloutBufferSamples, None, None]:
        assert self.full, ""
        total_size = self.real_buffer_size * self.n_envs
        if self.shuffle:
            indices = np.random.permutation(total_size)
        else:
            start_index = np.random.randint(total_size)
            indices = [(start_index + i) % total_size for i in range(total_size)]
        # Prepare the data
        if not self.generator_ready:
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "action_masks",
                # "dones", not needed for training
                "real_forecast",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.real_buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.real_buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> LstmMaskableRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.action_masks[batch_inds].reshape(-1, self.mask_dims),
            # self.dones[batch_inds].flatten(), not needed for training
            self.real_forecast[batch_inds].reshape(-1, self.forecast_dim) #CHECK RESHAPE
        )
        return LstmMaskableRolloutBufferSamples(*map(self.to_torch, data))

class TransformerMaskableRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor
    real_forecast: th.Tensor

class TransformerMaskableRolloutBuffer(MaskableRolloutBuffer):
    """
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        env_episode_length: int,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        shuffle: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        
        self.buffer_size = buffer_size
        self.real_buffer_size = buffer_size
        self.env_episode_length = env_episode_length

        self.real_forecast = None

        self.shuffle = shuffle
        # self.counter = 0

    def reset(self) -> None:
        self.real_forecast = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    def add(self, *args, forecast: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        """
        self.forecast[self.pos % self.env_episode_length] = forecast
        super().add(*args, **kwargs)

    

    def get(self, batch_size: Optional[int] = None) -> Generator[TransformerMaskableRolloutBufferSamples, None, None]:
        assert self.full, ""
        total_size = self.real_buffer_size * self.n_envs
        if self.shuffle:
            indices = np.random.permutation(total_size)
        else:
            start_index = np.random.randint(total_size)
            indices = [(start_index + i) % total_size for i in range(total_size)]
        # Prepare the data
        if not self.generator_ready:
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "action_masks",
                # "dones", not needed for training
                "real_forecast",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.real_buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.real_buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> TransformerMaskableRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.action_masks[batch_inds].reshape(-1, self.mask_dims),
            # self.dones[batch_inds].flatten(), not needed for training
            self.real_forecast.flatten(), #CHECK RESHAPE
        )
        return TransformerMaskableRolloutBufferSamples(*map(self.to_torch, data))

class LstmMaskablePPO(MaskablePPO):
    def __init__(
        self,
        policy: Union[str, Type[MaskableActorCriticPolicy]],
        env: Union[GymEnv, str],
        forecast_dim: int,
        forecast_coef: float = 0.2,
        num_epochs_pretrain: int = 0,
        _init_setup_model: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            policy,
            env,
            _init_setup_model=False,
            *args,
            **kwargs,
        )
        # assert env.stop, "RolloutBuffer doesn't support action masking"
        if isinstance(env, DummyVecEnv):
            self.env_stop = env.envs[0].stop
        else:
            self.env_stop = env.stop
        self.forecast_dim = forecast_dim
        self.forecast_coef = forecast_coef
        self.num_epochs_pretrain = num_epochs_pretrain

        assert not (self.forecast_dim < 1 and self.num_epochs_pretrain), "Condition failed: If forecast_dim < 1, then num_epochs_pretrain must be 0 epochs." 

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )

        self.policy = self.policy.to(self.device)

        if not isinstance(self.policy, LstmMaskableActorCriticPolicy):
            raise ValueError("Policy must subclass LstmMaskableActorCriticPolicy")

        buffer_cls = LstmMaskableRolloutBuffer
        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.env_stop,
            self.forecast_dim,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            shuffle = False,
        )
        try:
            if self.env_stop > self.n_steps:
                print("/!\ Warning, episode legnth > n_steps (n_rollout_steps/buffer_size), buffer will contain false values of real_forecast.")
        except:
            print("/!\ Warning, cannot get max episode length, ensure max_episode_legnth <= n_steps, otherwise buffer will contain false values of real_forecast.")

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:

        assert isinstance(
            rollout_buffer, (MaskableRolloutBuffer, MaskableDictRolloutBuffer)
        ), "RolloutBuffer doesn't support action masking"
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()
        # print(rollout_buffer.buffer_size)

        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking. Consider using ActionMasker wrapper")

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                # This is the only change related to invalid action masking
                if use_masking:
                    action_masks = get_action_masks(env)
                # print('collect rollouts: policy')
                actions, values, log_probs, _ = self.policy(obs_tensor, action_masks=action_masks)
            # print('collect rollouts: after policy')
            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            # print('collect rollouts, VecEnv', dones)
            real_forecast = [i['real_forecast'] for i in infos]

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos) # info['real_forecast'] has no impact
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                done=dones,
                forecast=real_forecast,
                action_masks=action_masks
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            # Masking is not needed here, the choice of action doesn't matter.
            # We only want the value of the current observation.
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # self.policy.mlp_extractor.set_requires_grad(True)
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, forecast_losses = [], [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                real_forecast = rollout_data.real_forecast #CHECK SHAPE (see action longed +flatten) TODO

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # print(rollout_data.observations.shape, actions.shape, rollout_data.action_masks.shape)

                values, log_prob, entropy, forecast = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Forecast loss using real observed values
                forecast_loss = F.mse_loss(rollout_data.real_forecast, forecast)
                forecast_losses.append(forecast_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.forecast_coef * forecast_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/forecast_loss", np.mean(forecast_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfMaskablePPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        use_masking: bool = True,
        progress_bar: bool = False,
    ) -> SelfMaskablePPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            use_masking,
            progress_bar,
        )
        pretrained = False
        callback.on_training_start(locals(), globals())
        # print('learn: collect rollouts')
        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.n_steps, use_masking)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.rollout_buffer.remove_incomplete_rows()
            if self.num_epochs_pretrain > 0 and not pretrained:
                print('pretraining forecast')
                pretrained = self.forecast_pretrain()
            self.train()

        callback.on_training_end()

        return self

    @classmethod
    def load(  # noqa: C901
        cls: Type[SelfBaseAlgorithm],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfBaseAlgorithm:
        """
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if "net_arch" in data["policy_kwargs"] and len(data["policy_kwargs"]["net_arch"]) > 0:
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            forecast_dim=data["forecast_dim"],
            _init_setup_model=False,  # type: ignore[call-arg]
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]
        return model

    def forecast_pretrain(self):
        #modify or long sequences, buffer at beginning longer then normal size
        self.policy.set_training_mode(True)
        optimizer = optim.Adam(self.policy.mlp_extractor.lstm.parameters(), lr=0.002)
        loss_function = nn.MSELoss()
        losses = []

        for epoch in range(self.num_epochs_pretrain):
            epoch_losses = []  # Track losses for each epoch
            
            # Assuming buffer.get() is correctly implemented to yield batches of samples
            for samples in self.rollout_buffer.get(self.batch_size):  # You can define batch_size or leave it to get all data
                # Extract real_forecast from the samples
                features, real_forecast = samples.observations, samples.real_forecast

                # Reset gradients
                optimizer.zero_grad()

                # Forward pass through LSTM using real_forecast as target
                lstm_output = self.policy.predict_forecast(features)

                # Compute loss
                loss = loss_function(lstm_output, real_forecast)

                # Backward pass and update LSTM parameters
                loss.backward()
                optimizer.step()

                # Append loss to the epoch losses list
                epoch_losses.append(loss.item())

            # Calculate and print the average loss for the epoch
            epoch_loss = np.mean(epoch_losses)
            self.logger.record("train/forecast_pretrain_losses", epoch_loss)
        print(f"Last Pretrain loss at epoch: {epoch+1}, Loss: {epoch_loss}")
        return True

class TransformerMaskablePPO(MaskablePPO):
    def __init__(
        self,
        policy: Union[str, Type[MaskableActorCriticPolicy]],
        env: Union[GymEnv, str],
        forecast_coef: float = 0.2,
        num_epochs_pretrain: int = 0,
        _init_setup_model: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            policy,
            env,
            _init_setup_model=False,
            *args,
            **kwargs,
        )
        # assert env.stop, "RolloutBuffer doesn't support action masking"
        if isinstance(env, DummyVecEnv):
            self.env_stop = env.envs[0].stop
        else:
            self.env_stop = env.stop
        # self.forecast_dim = forecast_dim
        self.forecast_coef = forecast_coef
        self.num_epochs_pretrain = num_epochs_pretrain

        # assert not (self.forecast_dim < 1 and self.num_epochs_pretrain), "Condition failed: If forecast_dim < 1, then num_epochs_pretrain must be 0 epochs." 

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )

        self.policy = self.policy.to(self.device)

        if not isinstance(self.policy, TransformerMaskableActorCriticPolicy):
            raise ValueError("Policy must subclass TransformerMaskableActorCriticPolicy")

        buffer_cls = TransformerMaskableRolloutBuffer
        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.env_stop,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            shuffle = False,
        )
        try:
            if self.env_stop > self.n_steps:
                print("/!\ Warning, episode legnth > n_steps (n_rollout_steps/buffer_size), buffer will contain false values of real_forecast.")
        except:
            print("/!\ Warning, cannot get max episode length, ensure max_episode_legnth <= n_steps, otherwise buffer will contain false values of real_forecast.")

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        use_masking: bool = True,
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param use_masking: Whether or not to use invalid action masks during training
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return:
        """

        self.start_time = time.time_ns()
        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs= self.env.reset()
            self._reset_infos = self.env.reset_infos[0] #TODO check for multiple envs /!\ (maybe the same)
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, use_masking, progress_bar)

        return total_timesteps, callback

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:

        assert isinstance(
            rollout_buffer, (MaskableRolloutBuffer, MaskableDictRolloutBuffer)
        ), "RolloutBuffer doesn't support action masking"
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()
        # print(rollout_buffer.buffer_size)

        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking. Consider using ActionMasker wrapper")

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                #get last obs to predict from buffer for inference
                if rollout_buffer.pos > 0:
                    ts = rollout_buffer.real_forecast[:,0][:pos].tolist() #replace 0 with id env if multiple TODO /!\
                else:
                    ts = [self._reset_infos['real_forecast']]
                print(ts)
                _, zone, week, b_type = self._reset_infos.values()
                infer_df = self.policy.mlp_extractor.transformer.create_infer_dataset(week, b_type, zone, ts)
                print(infer_df)
                print(type(infer_df['start'][0]))

                # This is the only change related to invalid action masking
                if use_masking:
                    action_masks = get_action_masks(env)
                # print('collect rollouts: policy')
                actions, values, log_probs, _ = self.policy(infer_df, obs_tensor, action_masks=action_masks)
                raise AssertionError
            # print('collect rollouts: after policy')
            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            # print('collect rollouts, VecEnv', dones)
            real_forecast = [i['real_forecast'] for i in infos]

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos) # info['real_forecast'] has no impact
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                forecast=real_forecast,
                action_masks=action_masks
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            # Masking is not needed here, the choice of action doesn't matter.
            # We only want the value of the current observation.
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # self.policy.mlp_extractor.set_requires_grad(True)
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, forecast_losses = [], [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                real_forecast = rollout_data.real_forecast #CHECK SHAPE (see action longed +flatten) TODO
                

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # print(rollout_data.observations.shape, actions.shape, rollout_data.action_masks.shape)

                values, log_prob, entropy, forecast = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Forecast loss using real observed values
                forecast_loss = F.mse_loss(rollout_data.real_forecast, forecast)
                forecast_losses.append(forecast_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.forecast_coef * forecast_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/forecast_loss", np.mean(forecast_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfMaskablePPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        use_masking: bool = True,
        progress_bar: bool = False,
    ) -> SelfMaskablePPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            use_masking,
            progress_bar,
        )
        pretrained = False
        callback.on_training_start(locals(), globals())
        # print('learn: collect rollouts')
        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.n_steps, use_masking)
            raise AssertionError

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.rollout_buffer.remove_incomplete_rows()
            if self.num_epochs_pretrain > 0 and not pretrained:
                print('pretraining forecast')
                pretrained = self.forecast_pretrain()
            self.train()

        callback.on_training_end()

        return self

    @classmethod
    def load(  # noqa: C901
        cls: Type[SelfBaseAlgorithm],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfBaseAlgorithm:
        """
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if "net_arch" in data["policy_kwargs"] and len(data["policy_kwargs"]["net_arch"]) > 0:
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            forecast_dim=data["forecast_dim"],
            _init_setup_model=False,  # type: ignore[call-arg]
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]
        return model

    def forecast_pretrain(self):
        #modify or long sequences, buffer at beginning longer then normal size
        self.policy.set_training_mode(True)
        optimizer = optim.Adam(self.policy.mlp_extractor.lstm.parameters(), lr=0.002)
        loss_function = nn.MSELoss()
        losses = []

        for epoch in range(self.num_epochs_pretrain):
            epoch_losses = []  # Track losses for each epoch
            
            # Assuming buffer.get() is correctly implemented to yield batches of samples
            for samples in self.rollout_buffer.get(self.batch_size):  # You can define batch_size or leave it to get all data
                # Extract real_forecast from the samples
                features, real_forecast = samples.observations, samples.real_forecast

                # Reset gradients
                optimizer.zero_grad()

                # Forward pass through LSTM using real_forecast as target
                lstm_output = self.policy.predict_forecast(features)

                # Compute loss
                loss = loss_function(lstm_output, real_forecast)

                # Backward pass and update LSTM parameters
                loss.backward()
                optimizer.step()

                # Append loss to the epoch losses list
                epoch_losses.append(loss.item())

            # Calculate and print the average loss for the epoch
            epoch_loss = np.mean(epoch_losses)
            self.logger.record("train/forecast_pretrain_losses", epoch_loss)
        print(f"Last Pretrain loss at epoch: {epoch+1}, Loss: {epoch_loss}")
        return True

