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


import ts_transformer as tst

SelfMaskablePPO = TypeVar("SelfMaskablePPO", bound="MaskablePPO")
SelfBaseAlgorithm = TypeVar("SelfBaseAlgorithm", bound="BaseAlgorithm")
SelfLstmMaskablePPO = TypeVar("SelfLstmMaskablePPO", bound="LstmMaskablePPO")


# warnings.filterwarnings('ignore', message="WARN: env.action_masks to get variables from other wrappers is deprecated and will be removed in v1.0, to get this variable you can do `env.unwrapped.action_masks` for environment variables or `env.get_wrapper_attr('action_masks')` that will search the reminding wrappers.")
# warnings.filterwarnings("ignore")

def action_space_to_dict(aspace):
    """ Only for box space """
    return { "high": aspace.high,
             "low": aspace.low,
             "shape": aspace.shape,
             "dtype": str(aspace.dtype)
    }

def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    # building_info = env.get_building_information()
    # building_info = list(building_info)
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "observation": observations }
    return obs_dict

def include_buildings(schema, id_buildings, solar, only_battery, pricing):
    id_buildings = [i-1 for i in id_buildings]
    
    schema['observations']['electricity_pricing']['active'] = True
    # schema['reward_function']['type'] = '../../../../../../CL/reward_function.RewardSumCost'
    
    if only_battery:
        schema['actions'] = {'cooling_storage': {'active': False},
                              'heating_storage': {'active': False},
                              'dhw_storage': {'active': False},
                              'electrical_storage': {'active': True}}
    
    for i,e in enumerate(schema['buildings']):
        inactive_obs = schema['buildings'][e]['inactive_observations']
        if i not in id_buildings:
            schema['buildings'][e]['include'] = False
        else:
            if 'solar_generation' in inactive_obs:
                schema['buildings'][e]['pv'] = {'type': 'citylearn.energy_model.PV',
                             'autosize': False,
                             'attributes': {'nominal_power': 0.0}}
                schema['buildings'][e]['inactive_observations'] = [i for i in inactive_obs if i != 'solar_generation']
            else:
                if not solar:
                    schema['buildings'][e]['pv'] = {'type': 'citylearn.energy_model.PV',
                             'autosize': False,
                             'attributes': {'nominal_power': 0.0}}
            if 'dhw_storage_soc' in inactive_obs:
                schema['buildings'][e]["dhw_storage"] = {"type": "citylearn.energy_model.StorageTank",
                                                        "autosize": True,
                                                        "autosize_attributes": {
                                                          "safety_factor": 0.0
                                                        },
                                                        "attributes": {
                                                          "capacity": None,
                                                          "loss_coefficient": 1.0
                                                        }
                                                      }
            if pricing:
                schema['buildings'][e]['pricing'] = 'C:/Users/Theo/Documents/These/CL/pricing.csv'
            else:
                schema['buildings'][e]['pricing'] = 'C:/Users/Theo/Documents/These/CL/pricing_null.csv'
                
    return schema

class Device_:
  def __init__(self, device, storage_type):
    self.device = device
    # self.price_cost = 0
    # self.emission_cost = 0
    self.cost = 0
    self.storage_type = storage_type

  def loss(self, cost_t, pv_offset, battery_offset):
    """
    get avg price between (battery release, grid release and PV- direct consumption)
    add relative incertainty, but true in pratice as the energy is added up in a global consumption pool 

    battery: if battery releases, price = avg((total released by battery - remaining conso), grid) in the case of thermal
    in the case of battery, avg price with PV
    """
    if not self.device:
      print('not device')
      raise ValueError

    energy_used = self.device.energy_balance[-1]
    if isinstance(energy_used, np.ndarray):
      print('probleme energy used array instead of float')
      energy_used = energy_used[0]

    #charge
    if energy_used > 0:
      #if pv production, part of the energy is free
      if pv_offset > 0:
        energy_used = max(0, energy_used-pv_offset)
      #if usage of battery, part of energy has been already taken into account so free
      if battery_offset > 0:
        energy_used = max(0, energy_used-battery_offset)
      # self.price_cost = ((self.price_cost*self.device.soc[-2])+(energy_used*price))/self.device.soc[-1]
      # self.emission_cost = ((self.emission_cost*self.device.soc[-2])+(energy_used*emission))/self.device.soc[-1]
      
      total = self.device.soc[-1]
      if isinstance(total, np.ndarray):
        print('probleme soc-1 array instead of float')
        total = total[0]

      prev = self.device.soc[-2]
      if isinstance(prev, np.ndarray):
        print('probleme soc-2 array instead of float')
        prev = prev[0]

      self.cost = ((self.cost*prev) + (energy_used*cost_t)) / total
      return energy_used, None, None #energy_used > 0

    #discharge
    else:
      #energy_processed is total energy used during charge/discharge process including losses
      #energy_used is the energy_processed minus the losses (used by building)
      energy_processed = self.device.soc[-2]-self.device.soc[-1]
      return -energy_used, energy_processed, self.cost # -energy_used > 0, energy_processed > 0 

  def update_cost(self, energy_used, price_t, emission_t):
    prev_soc = 0 if len(self.device.soc)<2 else self.device.soc[-2]
    cost_t = price_t + emission_t
    self.cost = ((self.cost*prev_soc) + (energy_used*cost_t)) / self.device.soc[-1]

class BuildingDevices:
  """
    Keeps track of all storage devices of a building.
  """
  def __init__(self, building, num_building):
    self.num_building = num_building
    self.building = building
    self.devices = {'battery' : Device_(building.electrical_storage, 'battery'),
                    'cooling' : None,
                    'dhw' : None}
    
  def compute_bounds(self):
    bounds = [self.bounds_action(i) for i,j in self.devices.items() if j is not None]
    return gym.spaces.Box(low=np.array([i[0] for i in bounds]), high=np.array([i[1] for i in bounds]), dtype=np.float64)
  
# ACTION 0 :  cooling
# ACTION 1 : dhw
# ACTION 2 : battery
  
  def bounds_action(self, type_action):
    device = self.devices[type_action].device
    if device is None:
        return None # if return none building doest have battery
    if type_action == 'battery':
        capacity = device.capacity_history[-2] if len(device.capacity_history) > 1 else device.capacity
        #HIGH
        #get max energy that the storage unit can use to charge [kW]
        #if trying to put more than the battery can accept reject action
        high1 = device.get_max_input_power()/capacity
        high2 = (device.capacity - device.soc_init)/(0.95*device.capacity) #approxim (efficiency = 0.95)
        high = min(high1, high2, 1)

        #LOW
        low1 = -device.get_max_input_power()/capacity
        low2 = (-device.soc_init*0.95)/device.capacity #approxim (efficiency = 0.95)
        low = max(low1, low2, -1)

    else:
        bool_h2, bool_l2 = False, False
        if type_action == 'cooling':
            # print('\ncooling')
            space_demand = self.building.cooling_demand[self.building.time_step]
            max_output = self.building.cooling_device.get_max_output_power(self.building.weather.outdoor_dry_bulb_temperature[self.building.time_step], False)
            # print('space_demand',space_demand)
            # print('max_output', max_output)
            # print('capacity:', device.capacity)
        else: #dhw
            # print('\ndhw')
            space_demand = self.building.dhw_demand[self.building.time_step]
            max_output = self.building.dhw_device.get_max_output_power(self.building.weather.outdoor_dry_bulb_temperature[self.building.time_step], False)\
            if isinstance(self.building.dhw_device, HeatPump) else self.building.dhw_device.get_max_output_power()
            # print('space_demand',space_demand)
            # print('max_output', max_output)
        space_demand = 0 if space_demand is None or math.isnan(space_demand) else space_demand # case where space demand is unknown

        #HIGH
        high1 = (max_output-space_demand) / device.capacity
        # print('high1', high1)
        if device.max_input_power is not None:
            bool_h2 = True
            high2 = device.max_input_power / device.capacity
            # print('high2', high2)
        high3 = (device.capacity - device.soc_init) / (device.capacity*device.efficiency)
        # print(device.capacity, device.soc_init)
        # print('high3', high3)
        
        if bool_h2:
            high = min(high1, high2, high3, 0.5)
        else:
            high = min(high1, high3, 0.5)


        #LOW
        low1 = -space_demand / device.capacity
        # print('low1', low1)
        if device.max_output_power is not None:
            bool_l2 = True
            low2 = -device.max_output_power / device.capacity
            # print('low2',low2)
        low3 = (-device.soc_init*device.efficiency) / device.capacity
        # print('low3',low3)

        if bool_l2:
            low = max(low1, low2, low3, -0.5)
        else:
            low = max(low1, low3, -0.5)

    return (low, high)
  
  def cost(self, zeta, eta, carbon_price):
    """
    Other way to compute cost.
    1) we compute the total electrical consumption of the building,
    2) we the offset the PV generation if existant.
    3) we treat the case of charging and discharging the device 
    """
    #without dhw and cooling storage
    #net conso = cooling + dhw + electrical_storage + nsl - solar
    global_conso = 0
    building = self.building
    #convert kgCO2 in EUR
    carbon = building.carbon_intensity.carbon_intensity * carbon_price[:len(building.carbon_intensity.carbon_intensity)]

    price = building.pricing.electricity_pricing[building.time_step]
    carbon = carbon[building.time_step] * eta

    # print(building.time_step)
    cooling_demand = building.energy_simulation.cooling_demand[building.time_step] + building.cooling_storage.energy_balance[building.time_step]
    cooling_conso = building.cooling_device.get_input_power(cooling_demand, building.weather.outdoor_dry_bulb_temperature[building.time_step], heating=False)
    global_conso += cooling_conso

    dhw_demand = building.energy_simulation.dhw_demand[building.time_step] + building.dhw_storage.energy_balance[building.time_step]
    if isinstance(building.dhw_device, HeatPump):
            dhw_consumption = building.dhw_device.get_input_power(dhw_demand, building.weather.outdoor_dry_bulb_temperature[building.time_step], heating=True)
    else:
            dhw_consumption = building.dhw_device.get_input_power(dhw_demand)
    
    global_conso += dhw_consumption
    global_conso += building.energy_simulation.non_shiftable_load[building.time_step]
    global_conso -= building.pv.get_generation(building.energy_simulation.solar_generation)[building.time_step]

    #battery
    #discharge 
    #energy that can be used by building (< energy actually discharged)
    battery_conso_used = building.energy_from_electrical_storage[building.time_step]
    #remove from global conso the energy delivered by battery (not bought from the grid)
    # print('battery_conso_used', battery_conso_used)
    global_conso -= battery_conso_used

    #energy coming out of battery
    soc_t = building.electrical_storage.soc[-1]
    soc_t_1 = 0 if len(building.electrical_storage.soc) < 2 else building.electrical_storage.soc[-2]
    battery_net_conso =  max(0, soc_t_1 - soc_t) #keep only the case where we discharge
    # print('battery_net_conso', battery_net_conso)
    adjusted_battery_net_conso = battery_net_conso * (1 - zeta)
    
    #charge
    energy_used = building.energy_to_electrical_storage[building.time_step]
    global_conso += (energy_used * zeta)
    if energy_used > 0: #charging
        #update cost of 1 unit of energy in the device
        self.devices['battery'].update_cost(energy_used, price, carbon)
    
    # print('globa_conso', global_conso)
    #can be neagtive
    global_conso = max(0, global_conso)

    # print(global_conso)
    cost = (price + carbon) * global_conso
    cost += self.devices['battery'].cost * adjusted_battery_net_conso

    return -cost

def mask_fn(env: gymnasium.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()

def normalize_norm(mode, cat, value):
    """
    mode: random_mask
    cat: reward
    """
    if mode == 'random_discrete':
        if cat == 'reward':
            mean, std = -44.18679279288984, 34.31202033008597
        elif cat == 'temp':
            mean, std = 20.94367808219178, 7.28775091380609
        elif cat == 'nsl':
            mean, std = 24.83374885844749, 17.09060590705837
        elif cat =='net_conso':
            mean, std = 49.89326603300466, 61.492737733468275
        elif cat == 'soc_b':
            mean, std = 0.5886851198326962, 0.39343313608636826
        elif cat == 'cost':
            mean, std = 0.7850008976449486, 0.13398387077199095
        elif cat == 'pricing':
            mean, std = 0.2731312785388128, 0.1178026978827731
        elif cat == 'carbon_price':
            mean, std = 0.012705062874857505, 0.002384982850682248
        elif cat == 'cost_esu':
            mean, std = 0.8450902559653719, 0.12132634999196334
            
    return normalize_operation(value, mean, std)

def normalize_cycle(value, maxi):
    x_norm = 2 * math.pi * value / maxi
    return np.cos(x_norm), np.sin(x_norm)

def normalize_obs(obs, mode):
    normalized = []
    for i,e in enumerate(obs):
        if i == 0: #month
            normalized += normalize_cycle(e, 12)
        elif i == 1: #day type
            normalized += normalize_cycle(e, 8)
        elif i == 2: #hour
            normalized += normalize_cycle(e, 24)
        elif i == 3:
            normalized.append(normalize_norm(mode, 'temp', e))
        elif i == 4:
            normalized.append(normalize_norm(mode, 'nsl', e))
        elif i == 5:
            normalized.append(normalize_norm(mode, 'net_conso', e))
        elif i == 6:
            normalized.append(normalize_norm(mode, 'soc_b', e))
        # elif i == 7:
        #     normalized.append(normalize_norm(mode, 'cost', e))
        elif i == 8:
            normalized.append(normalize_norm(mode, 'cost_esu', e))
        
        else:
            normalized.append(e)
    return normalized

def denormalize_norm(cat, value):
    """
    mode: random_mask
    cat: reward
    """
    if cat == 'reward':
        mean, std = -44.18679279288984, 34.31202033008597
    elif cat == 'temp':
        mean, std = 20.94367808219178, 7.28775091380609
    elif cat == 'nsl':
        mean, std = 24.83374885844749, 17.09060590705837
    elif cat =='net_conso':
        mean, std = 49.89326603300466, 61.492737733468275
    elif cat == 'soc_b':
        mean, std = 0.5886851198326962, 0.39343313608636826
    elif cat == 'cost':
        mean, std = 0.7850008976449486, 0.13398387077199095
    elif cat == 'pricing':
        mean, std = 0.2731312785388128, 0.1178026978827731
    elif cat == 'carbon_price':
        mean, std = 0.012705062874857505, 0.002384982850682248
    elif cat == 'cost_esu':
        mean, std = 0.8450902559653719, 0.12132634999196334
    
    return value * std + mean

def normalize_operation(value, mean, std):
    return (value - mean)/std

def get_exp_name(model_name, env, total_timesteps):
    """
    get info about training session.
    """
    action_space = 'Discrete' if env.discrete else 'Continuous'
    if type(env.custom_reward) is type:
        reward = str(env.custom_reward).split(".")[-1][:-2]
    else:
        reward = f'customR_{int(env.custom_reward)}'
        if env.custom_reward in [1,3]:
            reward += f'_zeta_{env.zeta}'

    equipment = 'devices'+'-'.join([str(len(i)) for i in env.devices])

    p = [model_name, str(env.num_buildings)+'building', equipment, action_space,
        reward, 'sum_cost_'+str(int(env.sum_cost)), 'eta_'+str(env.eta),
        'cost_ESU_'+str(int(env.cost_ESU)), str(total_timesteps)]

    return '_'.join(p)

def train_save_model(model_name, zone, list_buildings, devices, discrete, custom_reward, solar, sum_cost,
                    cost_ESU, zeta, normalize, stop, variations, carbon_pricing, exp_name,
                    checkpoint_path='./results', total_timesteps=None, policy_kwargs=None):

    # first we initialize the environment (petting zoo)
    if not total_timesteps:
        total_timesteps = 1_500_000
    #carbon_pricing = np.array(pd.read_csv('./carbon_pricing.csv')['price_co2'])*1000
    
    if normalize:
        env = EnvCityGym2(zone=zone, list_buildings=list_buildings, devices=devices, discrete=discrete, custom_reward=custom_reward,
                                solar=solar, sum_cost=sum_cost, cost_ESU=cost_ESU, zeta=zeta, 
                                carbon_price=carbon_pricing, normalize=False, stop=None, variations=variations)

        # print(env.normalize)
        normalization_values = get_norm_values(env, 10, 'normalization_json_file')
        env = EnvCityGym2(zone=zone, list_buildings=list_buildings, devices=devices, discrete=discrete, custom_reward=custom_reward,
                                solar=solar, sum_cost=sum_cost, cost_ESU=cost_ESU, zeta=zeta, 
                                carbon_price=carbon_pricing, normalize=normalize, stop=stop, variations=variations,
                                normalization_values=normalization_values)

    if 'mask' in model_name:
        env = ActionMasker(env, mask_fn)
    obs, _ = env.reset()
    obs = np.array(obs)

    #exp_name = get_exp_name(model_name, env,total_timesteps)
    # load model if exist
    if model_name == 'dqn':
        model = DQN('MlpPolicy', env, verbose=0, gamma=0.99, tensorboard_log="./train_paper/", device='cuda')
    elif model_name == 'sac':
        model = SAC('MlpPolicy', env, verbose=0, gamma=0.99, tensorboard_log="./train_paper/", device='cuda')
    elif model_name == 'ppo_mask':
        model = MaskablePPO(MaskableActorCriticPolicy, env,
                        verbose=1, tensorboard_log='./train_paper', device='cuda', policy_kwargs=policy_kwargs)
    elif model_name == 'ppo':
        model = PPO('MlpPolicy', env, verbose=0, gamma=0.99, tensorboard_log="./train_paper/", device='cuda',
                    n_steps=9984, learning_rate=0.0005, clip_range=0.2, ent_coef=0.001, seed=0)
    elif model_name == 'ppo_lstm':
        model = RecurrentPPO(policy="MlpLstmPolicy", env=env, policy_kwargs=policy_kwargs,
                            verbose=1, tensorboard_log='./train_paper', device='cuda')
    elif model_name == 'ppo_mask_lstm':
        model = LstmMaskablePPO(policy=LstmMaskableActorCriticPolicy, env=env, forecast_dim=5,
                                forecast_coef=0.5, num_epochs_pretrain=0, n_steps = 2096, policy_kwargs=policy_kwargs,
                                verbose=1, device='cuda', tensorboard_log='./train_paper')
    else:
        print('model not recognized')
        return None

    print(f'Model: {model_name}\n')
    # Train the agent
    print(exp_name)
    model.learn(total_timesteps=total_timesteps, tb_log_name=exp_name,log_interval=5)

    print('saving model')
    x = str(len(os.listdir(checkpoint_path))+1)
    model.save(checkpoint_path+'/'+x+'_pricing_'+exp_name+'.zip')
    if 'mask' in model_name:
        env = env.env
    return model, env.rewards

def test_model(model_name, model_path, zone, list_buildings, devices, discrete, custom_reward, solar, sum_cost,
                cost_ESU, zeta, normalize, stop, variations, carbon, model=None):

    if normalize:
        env = EnvCityGym2(zone=zone, list_buildings=list_buildings, devices=devices, discrete=discrete, custom_reward=custom_reward,
                                solar=solar, sum_cost=sum_cost, cost_ESU=cost_ESU, zeta=zeta, 
                                carbon_price=carbon_pricing, normalize=False, stop=None, variations=variations)

        # print(env.normalize)
        normalization_values = get_norm_values(env, 10, 'normalization_json_file')

    # first we initialize the environment (petting zoo)
    if not model:
        try:
            if model_name == 'ppo':
                print('PPO')
                model = PPO.load(model_path)
            elif model_name == 'ddpg':
                print('DDPG')
                model = DDPG.load(model_path)
            elif model_name == 'a2c':
                print('A2C')
                model = A2C.load(model_path)
            elif model_name == 'sac':
                print('SAC')
                model = SAC.load(model_path)
            elif model_name == 'ppo_mask':
                model = MaskablePPO.load(model_path)
            elif model_name == 'ppo_lstm':
                model = RecurrentPPO.load(model_path)
            elif model_name == 'ppo_mask_lstm':
                print('ii')
                model = LstmMaskablePPO.load(model_path, env=env)
        except:
            print('not_found')

    for i in range(1):
        done = False
        print(f'Case {i}:', i)
        env = EnvCityGym2(zone=zone, list_buildings=list_buildings, devices=devices, discrete=discrete, custom_reward=custom_reward,
                                solar=solar, sum_cost=sum_cost, cost_ESU=cost_ESU, zeta=zeta, 
                                carbon_price=carbon_pricing, normalize=normalize, stop=stop, variations=variations,
                                normalization_values=normalization_values)

        if 'mask' in model_name:
            env = ActionMasker(env, mask_fn)
        print(env)
        print()
        obs, _ = env.reset()
        obs = np.array(obs)
        #lstm
        lstm_states = None
        num_envs = 1
        # Episode start signals are used to reset the lstm states
        episode_starts = np.ones((num_envs,), dtype=bool)

        action_list = []
        while not done:
            # print(obs)
            # obs = [i[0] if isinstance(i, np.array()) else i for i in obs]
            # obs = np.array(obs)
            # print(obs)
            if model_name == 'ppo_lstm':
                action, lstm_states = model.predict(obs[0], state=lstm_states, episode_start=episode_starts, deterministic=True)
            if model_name == 'ppo_mask_lstm':
                action_masks = get_action_masks(env)
                action, _states = model.predict(obs, action_masks=action_masks)
                # action, _ = model.predict(obs, deterministic=True)
            else:
                action, _state = model.predict(obs[0], deterministic=True)
            # print(type(action))
            obs, rewards, done, _, _ = env.step(action)
            episode_starts = done
            
            if isinstance(action, np.ndarray):
                action = int(action)
            action_list.append(action)

        if discrete:
            if 'mask' in model_name:
                env = env.env
            action_list = [env.action_conversion(i) for i in action_list]
                
            
        # print(action_list)
        x = pd.Series(action_list, name='action')
        print('List of different actions taken:')
        print(x.value_counts())
        print(f'Heure(s) de charge: {sum(x>0)} soit {np.round(sum(x>0)*100/len(x),2)}% du temps')
        print(f'Heure(s) de décharge: {sum(x<0)} soit {np.round(sum(x<0)*100/len(x),2)}% du temps')
        print(f'Heure(s) de noop: {sum(x==00)} soit {np.round(sum(x==0)*100/len(x),2)}% du temps')

        for n, nd in env.env.evaluate().groupby('name'):
            nd = nd.pivot(index='name', columns='cost_function', values='value').round(3)
            print(n, ':', nd.to_dict('records'))
        print()

    solar = env.env.buildings[0].energy_simulation.solar_generation
    solar = env.env.buildings[0].pv.get_generation(solar)
    conso = env.env.buildings[0].net_electricity_consumption
    price = env.env.buildings[0].pricing.electricity_pricing
    carbon = env.env.buildings[0].carbon_intensity.carbon_intensity


    df = pd.DataFrame()
    # df['Time [hours]'] = [i for i in range(len(conso))]
    df['Net conso [kWh]'] = conso
    df['SOC [kWh]'] = env.env.buildings[0].electrical_storage.soc
    df['Conso w/o storage [kWh]'] = env.env.buildings[0].net_electricity_consumption_without_storage
    df['Conso w/o storage and PV [kWh]'] = env.env.buildings[0].net_electricity_consumption_without_storage_and_pv
    df['Solar generation [kWh]'] = solar
    # df.iloc[0][0] = 24 #first is last day of july
    df['Cost sum(emission,price)x50'] = (price+carbon)*50
    df['Cost price x100'] = price*100
    df['Cost carbon x100'] = carbon*100

    return df, action_list, env #all vals of df in kWh

class EnvCityGym2(gymnasium.Env):
    """
    Env wrapper coming from the gym library.
    """

    def __init__(self, zone, list_buildings, carbon_price, devices, discrete, custom_reward, solar, sum_cost,
                cost_ESU, zeta, normalize, eta=1, stop=None, variations={'mu' : (-0.1, 0.1),'sigma' : (0, 0.1)},
                normalization_values=None):
        # print(schema_filepath)
        schema_filepath = '../clenv/Lib/site-packages/citylearn/data/' + 'citylearn_challenge_2020_climate_zone_' + str(zone)+'/schema.json'
        schema = read_json(schema_filepath)
        schema['root_directory'] = '../clenv/Lib/site-packages/citylearn/data/' + 'citylearn_challenge_2020_climate_zone_' + str(zone) +'/'
        self.schema = include_buildings(schema, list_buildings, solar, True, True)
        self.stop = 8759 if stop is None else stop
        self.zone = zone
        self.list_buildings = list_buildings

        self.variations = variations
        self.normalization_values = normalization_values

        self.obs = 'method_1'
        # new obs
        if solar:
            self.index_keep = [0,1,2,3,22,23,27]
            self.index_norm = [12,7,24,1,1,1,1,1]
        else:
            self.index_keep = [0,1,2,3,22,27]
            # self.index_norm = [12,7,24,1,1,1,1]
            self.index_norm = [1,1,1,1,1,1,1]

        self.custom_reward = custom_reward
        self.sum_cost = sum_cost
        self.cost_ESU = cost_ESU
        self.zeta = zeta
        self.eta = eta
        self.discrete = discrete
        self.normalize = normalize

        #t to kg
        self.carbon_price = np.array(carbon_price/1000).squeeze()

        self.env = CityLearnEnv(schema=self.schema, simulation_start_time_step=0,
                                simulation_end_time_step=self.stop)
        
        if isinstance(self.custom_reward, type) and issubclass(self.custom_reward, RewardFunction):
            self.env.reward_function = self.custom_reward(self.env)
        #list of names of devices [[]]
        self.devices = devices
        self.building_devices = []
        # get the number of buildings
        self.num_buildings = len(self.env.action_space)

        low = self.env.observation_space[0].low
        high = self.env.observation_space[0].high        

        #if sum cost
        if self.sum_cost:
            cost_l = low[19]+low[28]
            cost_h = high[19]+high[28]

        d_low, d_high = [], []
        for i in self.devices[0]:
            if i == 'battery':
                d_low.append(low[26])
                d_high.append(high[26])
            elif i == 'cooling':
                d_low.append(low[24])
                d_high.append(high[24])
            elif i == 'dhw':
                d_low.append(low[25])
                d_high.append(high[25])

        low = [low[i] for i in self.index_keep]
        high = [high[i] for i in self.index_keep]

        low = low + d_low
        high = high + d_high

        #if sum cost
        if self.sum_cost:
            low.append(cost_l)
            high.append(cost_h)

        #if cost ESU, chage if multiple buildings
        if self.cost_ESU:
            for i in range(len(self.devices[0])):
                low.append(0)
                high.append(cost_h)

        if self.discrete:
            self.action_space = gymnasium.spaces.discrete.Discrete(21)
            self.action_map = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,
                                0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        else:
            self.action_space = self.env.action_space[0]

        if self.obs == 'method_1':
            #TODO modify for proper nb of obs
            obs_shape = 12
            if not self.normalize:
                obs_shape = 9
            self.observation_space = gymnasium.spaces.Box(low=-np.inf,
                            high=np.inf,
                            shape=(obs_shape,), 
                            dtype=np.float32)
        else:
            self.observation_space = gymnasium.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

        #keep last outdoor temp for each building
        self.temp = []
        self.rewards = []

        #remove if test
        self.print_config()

        # TO THINK : normalize the observation space

    def reset(self, seed=None, options=None):
        # print('reset')
        self.new_pricing()
        obs_dict = env_reset(self.env)
        obs = self.env.reset()
        real_forecast_t = obs[0][4]
        info = {'real_forecast': real_forecast_t}

        for i,e in enumerate(self.env.buildings):
          self.building_devices.append(BuildingDevices(e,i))
        self.temp.append(obs[i][3])
        obs, real_forecast_t = self.get_obs(obs)
        # print(real_forecast_t)

        #TODO week is static
        return np.array(obs), {'real_forecast': real_forecast_t, 'zone': self.zone, 'week':1, 'building_type':self.get_building_class()}
        # return self.get_obs(obs), info

    def step(self, action):
        """
        we apply the same action for all the buildings
        """
        t = self.env.time_step
        # print('action', action,'\n')
        
        #if action is discrete convert using action mapping
        if self.discrete:
            action = [[self.action_conversion(action)]]
            action = action[0]
        # print('action', action)
        action = [action]

        # we do a step in the environment
        obs, reward, done, info = self.env.step(action)
        # print('envcitygym', done)
        # print('normal_reward', reward)
        if t == self.stop:
            done = True
        
        #custom reward 1 is the one where we can use zeta
        if self.custom_reward == 1:
            for i,e in enumerate(self.env.buildings):
                rewards = []
                rewards.append(compute_loss(e, self.building_devices[i], self.env.buildings[i].pricing.electricity_pricing[t-1],
                self.env.buildings[i].carbon_intensity.carbon_intensity[t-1], self.temp[i], self.zeta))
                self.temp[i] = obs[i][3]
                #TODO multiple buildings
                obs, real_forecast_t = self.get_obs(obs)
                return np.array(obs), -rewards[0], done, False, {'real_forecast': real_forecast_t}
            
        # custom reward 2 is cost without storage - cost with storage
        elif self.custom_reward == 2:
            for i in range(len(self.env.buildings)):
                rewards = self.reward_diff(i)
                # print('rewards', rewards)
                self.rewards.append(rewards)
                #TODO multiple buildings
                obs, real_forecast_t = self.get_obs(obs)
                return np.array(obs), -rewards[0], done, False, {'real_forecast': real_forecast_t}            
            
        # custom reward 3 is the same as 1 but coded in a different way (not coded to be used w/ thermal storage)
        elif self.custom_reward == 3:
            rewards = []
            for i in range(len(self.env.buildings)):
                #TODO multiple buildings
                rewards.append(self.building_devices[i].cost(self.zeta, self.eta, self.carbon_price))
                # print('reward3', self.building_devices[i].cost(self.zeta))
                # return np.array(self.get_obs(obs)), normalize_norm('random_discrete', 'reward', rewards[0]), done, False, info
                obs, real_forecast_t = self.get_obs(obs)
                #return np.array(obs), normalize_norm('random_discrete', 'reward', rewards[0]), done, False, {'real_forecast': real_forecast_t} #original one
                if self.normalize == 'all':
                    r = self.normalize_from_tuple(None, reward[0], reward=True)
                    return np.array(obs), r, done, False, {'real_forecast': real_forecast_t}
                return np.array(obs), reward[0], done, False, {'real_forecast': real_forecast_t}

        elif self.custom_reward == RampingCost:
            for i in range(len(self.env.buildings)):
                _ = self.building_devices[i].cost(self.zeta, self.eta, self.carbon_price)
            obs, real_forecast_t = self.get_obs(obs)
            #for cost ESU
            if self.normalize == 'all':
                r = self.normalize_from_tuple(None, reward[0], reward=True)
                return np.array(obs), r, done, False, {'real_forecast': real_forecast_t}
            return np.array(obs), reward[0], done, False, {'real_forecast': real_forecast_t}
        #else use normal reward 
        else:
            #TODO multiple buildings
            obs, real_forecast_t = self.get_obs(obs)
            return np.array(obs), -reward[0], done, False, {'real_forecast': real_forecast_t}

    def get_obs(self, obs):
        #keep common obs
        obs_ = [[o[i]/n for i,n in zip(self.index_keep, self.index_norm)] for o in obs]
        # obs_ = list(itertools.chain(*obs_))

        #add soc of each device for each building
        for o in range(len(obs_)):
            if 'battery' in self.devices[o]:
                i = obs[o][26]
                if isinstance(i, np.ndarray):
                    print('probleme array instead of float soc battery obs')
                    i = i[0]
                obs_[o].append(i)
            if 'cooling' in self.devices[o]:
                obs_[o].append(obs[o][24])
            if 'dhw' in self.devices[o]:
                obs_[o].append(obs[o][25])

        #add sum of costs (emission+price)
        if self.sum_cost is True:
            for o in range(len(obs_)):
                #modify for buildings that dont have same nb of obs
                price_co2 = obs[o][19]*self.carbon_price[self.env.time_step]
                price = obs[o][28]
                if self.normalize:
                    price_co2 = normalize_norm('random_discrete','carbon_price',price_co2)
                    price = normalize_norm('random_discrete','pricing',price)
                    obs_[o].append(price_co2 + price)
                else:
                    obs_[o].append(price_co2 + price)
            # print(obs)
        
        #add cost of energy in storage device for each device of each building
        if self.cost_ESU is True:
            for o in range(len(obs_)):
                for i in self.devices[o]:
                    obs_[o].append(self.building_devices[o].devices[i].cost)

        real_forecast_t = obs_[0][4]
        if self.normalize:
            obs__ = []
            for j in obs_:
                tmp = []
                for i,e in enumerate(j):
                    if i in range(3):
                        if i == 0: #month
                            tmp.append(normalize_cycle(e, 12))
                        elif i == 1: #day type
                            tmp.append(normalize_cycle(e, 8))
                        elif i == 2: #hour
                            tmp.append(normalize_cycle(e, 24))
                    else:
                        tmp.append(self.normalize_from_tuple(i, e, reward=False))
                tmp = [item for sublist in tmp for item in (sublist if isinstance(sublist, tuple) else [sublist])]
                obs__.append(tmp)
            # if len(obs_[0]) != 9:
            #     print('/!\ CHECK NORMALIZATION INDICES')
            # obs_ = [normalize_obs(i, 'random_discrete') for i in obs_]
            obs__
            real_forecast_t = obs__[0][7]
            # print(obs__[0], real_forecast_t)
            return np.array(obs__[0]), real_forecast_t
        return np.array(obs_[0]), real_forecast_t
    
    def action_conversion(self, action):
        return self.action_map[action]
    
    def valid_action_mask(self):
        mod_action_space = self.building_devices[0].compute_bounds()
        act = np.array(self.action_map)
        index = list(np.where((act>mod_action_space.low[0]) & (act<mod_action_space.high[0]))[0])
        act = [True if i in index else False for i in range(21)]
        act[10] = True #noop always valid
        return act

    def print_config(self):
        print('INIT ENV:')
        act = 'Discrete' if self.discrete else 'Continuous'
        print(f'ACTION SPACE: {act}')
        print(f'Use of custom reward: {self.custom_reward}')
        if self.custom_reward in [1,3]:
            print(f'    zeta: {self.zeta}')
        print('Observations kept:')
        for i in self.index_keep:
            print(f'    {i}: {self.env.observation_names[0][i]}')
        for i in self.devices[0]:
            if i == 'battery':
                print('    26: '+self.env.observation_names[0][26])
            elif i == 'cooling':
                print('    24: '+self.env.observation_names[0][24])
            elif i == 'dhw':
                print('    25: '+self.env.observation_names[0][25])
        if self.sum_cost or self.cost_ESU:
            print(f'Observations ADDED:')
            if self.sum_cost:
                print(f'    sum_cost: {self.env.observation_names[0][19]} + {self.env.observation_names[0][28]}')
            if self.cost_ESU:
                print('    cost_ESU: see Device.loss')

    def reward_diff(self, building_i):
        r = []
        building = self.env.buildings[building_i]
        c1 = building.net_electricity_consumption_cost[-1]
        c2 = building.net_electricity_consumption_emission[-1]
        c = c1 + c2

        # c1_ = building.net_electricity_consumption_without_storage_cost[-1]
        # c2_ = building.net_electricity_consumption_without_storage_emission[-1]
        # c_ = c1_ + c2_

        c1_ = building.net_electricity_consumption_without_storage_and_pv_cost[-1]
        c2_ = building.net_electricity_consumption_without_storage_and_pv_emission[-1]
        c_ = c1_ + c2_

        final_cost = c_ - c
        return final_cost

    def reward4(self):
        c1 = building.net_electricity_consumption_cost[-1]
        c2 = building.net_electricity_consumption_emission[-1]

        cost = c1 + c2
        cost += building.net_electricity_consumption

    @staticmethod
    def add_gaussian_noise(df, mu, sigma):
        noise = np.random.normal(mu, sigma, df.shape)
        df = df + noise
        df[df < 0] = 0
        return df

    def new_pricing(self):
        b = 'Building_' + str(self.list_buildings[0])
        # print(self.env.schema['buildings'][b])
        pricing = pd.read_csv(self.env.schema['buildings'][b]['pricing'])
        mu = self.variations['mu']
        sigma = self.variations['sigma']
        for i in self.env.buildings:
            pricing = self.add_gaussian_noise(pricing, random.uniform(mu[0], mu[1]),
                                        random.uniform(sigma[0], sigma[1]))
            new_pricing = Pricing(pricing['Electricity Pricing [$]'], pricing['6h Prediction Electricity Pricing [$]'],
                                pricing['12h Prediction Electricity Pricing [$]'], pricing['24h Prediction Electricity Pricing [$]'])
            i.pricing = new_pricing

    def normalize_from_tuple(self, index, value, reward=False):
        if reward:
            mean, std = self.normalization_values[1]
        else:
            mean, std = self.normalization_values[0][index]
        return normalize_operation(value, mean, std)
    
    def get_building_class(self):
        assert len(self.list_buildings) == 1
        id_building = self.list_buildings[0]
        assert id_building in range(1,10)
        if id_building - 1 == 0:
            return 3
        elif id_building - 1 in [1,2,3]:
            return 1
        else:
            return 2

def get_norm_values(env, nb_run, normalization_json_file):
    """
        Calculate and store normalization values for observations and rewards from the environment.

        This function runs the environment for a specified number of times, taking random actions at each step. 
        It collects observations, rewards, and actions during these runs. After collecting the data, it calculates 
        the mean and standard deviation for the rewards and each observation dimension. These statistics are then 
        written to a file using the `write_normalization_values` function.

        Parameters:
        - env (object): The environment object that provides the reset and step methods.
        - nb_run (int): The number of times the environment should be run to collect data..

        Notes:
        - The function contains a TODO comment about adding a mask with observation name.
        - The function assumes that the `write_normalization_values` function is defined elsewhere to store the calculated statistics.
        - The function also contains commented-out print statements.
    """
    # print(env.__dict__)
    #TODO ADD MASK W OBS NAME
    action_list = []
    reward_list = []
    obs_list = []
    print('env.normalize', env.normalize)
    assert env.normalize is False, 'env.normalize should be False'

    #TODO crash if not all obs
    
    for i in range(nb_run):
        done = False
        obs = env.reset()
        obs_list.append(obs[0])
        while not done:
            action = random.randint(0,20)
            obs, rewards, done, _, _ = env.step(action)
            obs_list.append(obs)
            reward_list.append(rewards)
            action_list.append(action)
            if isinstance(action, np.ndarray):
                action = int(action)
            action_list.append(action)
    
    stat_reward = (statistics.mean(reward_list), statistics.stdev(reward_list))
    obs_list = [i for i in obs_list]
    stat_obs = [[i[j] for i in obs_list] for j in range(len(obs_list[0]))]
    stat_obs = [(statistics.mean(i), statistics.stdev(i)) for i in stat_obs]    
    # write_normalization_values(env, stat_reward, stat_obs, normalization_json_file)
    return stat_obs, stat_reward

def find_spans_over_threshold(values, threshold):
    spans = []
    start_idx = None

    for i, val in enumerate(values):
        # Check if current value is above threshold and we're not already in a span
        if val > threshold and start_idx is None:
            start_idx = i
        # Check if current value is below threshold and we're currently in a span
        elif val <= threshold and start_idx is not None:
            spans.append((start_idx, i - 1))
            start_idx = None

    # Check if the last value is part of a span
    if start_idx is not None:
        spans.append((start_idx, len(values) - 1))

    return spans

class ConcatLayer(nn.Module):
    def __init__(self, concat_output_size, dim=2):
        super().__init__()
        self.concat_output_size = concat_output_size
        self.dim = dim

    def forward(self, x, y):
        res = th.cat((x, y), dim=self.dim)
        return res.view(-1, self.concat_output_size)


