### Differentiable Simulator

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict as DefaultDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import copy


class Simulator(gym.Env):
    metadata = {"render_modes": None}

    def __init__(self, device='cpu'):
        self.device = device
        self.problem_params, self.state_params = None, None
        # current_states include all info relating to the current period that we can observe
        # internal_data include all sequences, e.g., demand seq, feature seq, lt seq
        self.current_state, self.internal_data, self.batch_size = None, None, None

    def initialize(self, data, problem_params, state_params):

        self.problem_params = problem_params
        self.state_params = state_params
        self.batch_size = len(data['demands'])

        # Initialize internal_data, include state_params['period_state']
        self.internal_data = {
            'demands': data['demands'],
            'features': data['features'],
            'features_org': data['features_org'],
            'lead_times': data['lead_times'],
            'period_num': data['period_num'],
            'ignore_period_num': data['ignore_period_num'],
            'sales': data['demands'].clone(),
            # ALIBABA
            'reg_input': data['reg_input'],
        }

        # Initialize the first dynamic state
        self.current_state = {
            'current_period': torch.tensor([0]).to(self.device),
            'inventory': data['initial_inventory_pipeline'],  # shape (sample_num, 1 + max_lead_time)
            'inv_level_end': torch.zeros(self.batch_size, data['demands'].shape[2]).to(self.device),
            'lost_sales': torch.zeros(self.batch_size, data['demands'].shape[2]).to(self.device),
        }

        # 'holding_cost', 'backlog_cost', 'review_period'
        for k, v in state_params['static_state'].items():
            if v:
                self.current_state[k] = data[k]

        # 'past_arrivals', 'past_orders'
        # we do not use any past state in the current training
        for k, v in state_params['past_state'].items():
            if v > 0:
                self.current_state[k] = torch.zeros(self.batch_size, v).to(self.device)

        # 'demands', 'features', 'lead_times', 'reg_input'
        self.get_period_state(self.internal_data, self.current_state, self.state_params, current_period=0)

        # Initialize state space and action space
        self.state_space = self.init_state_space(self.current_state, problem_params)
        self.action_space = spaces.Box(low=0.0, high=float(problem_params['max_replenish']), shape=(self.batch_size, 1), dtype=np.float32)

        return self.current_state

    def transition(self, action):
        # action is a tensor of shape (sample_num, 1)
        # Event order: replenish pipeline, fulfill demand, incur loss, update inventory/features
        current_period = self.current_state['current_period'].item()
        # true action based on review period
        action = self.get_action_by_review_period(action, current_period)

        # update pipeline: works for L>= 0;
        lead_time = self.internal_data['lead_times'][:, :, current_period].long() #  shape (N, 1)
        self.current_state['inventory'] = self.update_pipeline(self.current_state['inventory'], lead_time, action)

        # update sales
        ini_inventory = self.current_state['inventory'][:, 0].unsqueeze(1).clone()
        demand = self.internal_data['demands'][:, :, current_period].clone()

        sales = torch.clip(demand, max=ini_inventory)
        self.internal_data['sales'][:, :, current_period] = sales.clone()

        # post_inventory; loss shape (N, 1)
        post_inventory = self.current_state['inventory'][:, 0].unsqueeze(1) - self.internal_data['demands'][:, :,
                                                                              current_period]
        # We now compute loss in run_epoch
        loss = 0
        # loss = self.current_state['holding_cost'] * torch.clip(post_inventory, min=0) \
        #        + self.current_state['backlog_cost'] * torch.clip(-post_inventory, min=0)
        # # incur loss if current_period <= sample's period_num and >= sample's ignore_period_num; and average over period_num
        # loss = self.get_true_avg_loss(loss, current_period, self.internal_data['ignore_period_num'], self.internal_data['period_num'])

        # update past state
        self.update_past_state(action)

        # Lost sales: clip inventory at 0 (excess demand is lost, not backlogged)
        if self.problem_params['lost_sales_demand']:
            self.current_state['lost_sales'][:, current_period] = torch.clip(-post_inventory, min=0).squeeze(1)
            post_inventory = torch.clip(post_inventory, min=0)

        # Alibaba obj: record inv level at end of period t
        self.current_state['inv_level_end'][:, current_period] = post_inventory.squeeze(1).clone()

        # update inventory pipeline
        # current_state['inventory'] = [on-hand inv at start of t+1, t+2 order, t+3 order, t+4 order, 0]
        # note: it is the net input to t+1
        if self.current_state['inventory'].shape[1] > 1:
            self.update_inventory_pipeline(self.current_state['inventory'], post_inventory)

        # update period features
        self.get_period_state(self.internal_data, self.current_state, self.state_params, current_period + 1)

        # next period
        self.current_state['current_period'] += 1
        return self.current_state, loss


    def get_tensor_start_end_index(self, obj, start_idx, end_idx):
        # obj (N, T); start_index, end_index (N,1)
        # to get flaged obj: demands satisfying start> t or t >= end_idx are set to be zero
        flag = (torch.arange(obj.size(1), device=obj.device).unsqueeze(0) >= start_idx) \
               & (torch.arange(obj.size(1), device=obj.device).unsqueeze(0) < end_idx)  # Shape (N, T)
        flaged_obj = obj * flag  # Shape (N, T), set unnecessary values to 0
        return flaged_obj, flag

    def ali_turnover_stockout_avg_syn(self, train):
        max_turnover = self.problem_params['max_turnover']
        start = self.internal_data['ignore_period_num']  # (N, 1)
        length = self.internal_data['period_num'] - start

        # turnover rate: sum of stock / sum of demand; sum from lt to the end
        stock, sales = self.ali_turnover_eval(start, length)
        stock /= length
        sales /= length
        turnover = torch.clip(stock / (sales + 1e-6), max=max_turnover)
        turnover_loss = turnover / max_turnover

        _, _, stock_out_rate = self.ali_short_rate_eval(start, length, train)

        return turnover_loss, stock_out_rate

    def ali_loss_syn(self, train):
        # underage and overage loss
        start_idx = self.internal_data['ignore_period_num']  # (N, 1)
        length = self.internal_data['period_num'] - start_idx

        end_idx = start_idx + length  # end_idx not included

        flag_inv, _ = self.get_tensor_start_end_index(self.current_state['inv_level_end'], start_idx, end_idx)  # (N,T)
        overage_inv = torch.clip(flag_inv, min=0)

        if self.problem_params['lost_sales_demand']:
            flag_lost, _ = self.get_tensor_start_end_index(self.current_state['lost_sales'], start_idx, end_idx)
            underage_inv = flag_lost
        else:
            underage_inv = torch.clip(-flag_inv, min=0)
        return overage_inv.sum(dim=1, keepdim=True) / length, underage_inv.sum(dim=1, keepdim=True) / length

    def ali_turnover_eval(self, start_idx, length):
        # start_idx, length (N, 1)
        end_idx = start_idx + length  # end_idx not included
        flag_inv, _ = self.get_tensor_start_end_index(self.current_state['inv_level_end'], start_idx, end_idx)  # (N,T)
        flag_sales, _ = self.get_tensor_start_end_index(self.internal_data['sales'].squeeze(1), start_idx, end_idx)
        return flag_inv.sum(dim=1, keepdim=True), flag_sales.sum(dim=1, keepdim=True)  # (N,1)

    def ali_short_rate_eval(self, start_idx, length, train):
        """Compute the stock out rate for the action taken on day t_action"""
        end_idx = start_idx + length  # (N, 1); end_idx not included
        flag_stock, _ = self.get_tensor_start_end_index(self.current_state['inv_level_end'], start_idx, end_idx)  # (N,T)

        # Use sigmoid to make indicator function differentiable
        if train:
            nonzero_stocks = torch.sigmoid(flag_stock)
            num_oos = length - nonzero_stocks.sum(dim=1, keepdim=True) + start_idx * torch.sigmoid(torch.tensor(0.0)).item()
        else:
            nonzero_stocks = torch.count_nonzero(flag_stock, dim=1).unsqueeze(1)
            num_oos = length - nonzero_stocks.sum(dim=1, keepdim=True)
        return num_oos, length, num_oos.float() / length

    def update_inventory_pipeline(self, inventory_pipeline, post_inventory):
        inventory_pipeline[:, 0] = post_inventory.squeeze(1) + inventory_pipeline[:, 1]
        inventory_pipeline[:, 1:-1] = inventory_pipeline[:, 2:].clone()
        inventory_pipeline[:, -1] = 0
        return inventory_pipeline

    def get_true_avg_loss(self, loss, current_period, ignore_period_num, period_num):
        if_condition = (ignore_period_num <= current_period) & (current_period <= period_num - 1)
        return loss * if_condition.float() / (period_num - ignore_period_num)

    def update_pipeline(self, current_inventory, lead_time, action):

        return current_inventory.scatter_add(1, lead_time, action)

    def get_action_by_review_period(self, action, current_period):
        bp = self.current_state['review_period'].long()
        if_review_period = (current_period % bp == 0) & (current_period >= bp)
        return action * if_review_period.float()

    def update_past_state(self, action):
        # update 'past_arrivals', 'past_orders'
        if self.state_params['past_state']['past_arrivals'] > 0:
            self.current_state['past_arrivals'] = torch.cat(
                (self.current_state['past_arrivals'][:, 1:], self.current_state['inventory'][:, 1].unsqueeze(1))
                , dim=1)
        if self.state_params['past_state']['past_orders'] > 0:
            self.current_state['past_orders'] = torch.cat((self.current_state['past_orders'][:, 1:], action), dim=1)

    def get_period_state(self, internal_data, current_state, state_params, current_period):
        for k in state_params['period_state']:
            # period state are all has 3 dim
            tmp_index = min(current_period, internal_data[k].shape[2] - 1)
            current_state[k] = internal_data[k][:, :, tmp_index]

    def init_state_space(self, initial_state, problem_params):
        box_values = DefaultDict(lambda: {'low': -np.inf, 'high': np.inf, 'dtype': np.float32})
        box_values.update({
            'current_period': {'low': 0, 'high': 10 * 365, 'dtype': np.int32},
            'inventory': {'low': 0 if problem_params['lost_sales_demand'] else np.inf, 'high': np.inf,
                          'dtype': np.float32},
            'holding_cost': {'low': 0, 'high': np.inf, 'dtype': np.float32},
            'backlog_cost': {'low': 0, 'high': np.inf, 'dtype': np.float32},
            'review_period': {'low': 0, 'high': 28, 'dtype': np.int32},
            'demands': {'low': 0, 'high': np.inf, 'dtype': np.float32},
            'features': {'low': -np.inf, 'high': np.inf, 'dtype': np.float32},
            'features_org': {'low': -np.inf, 'high': np.inf, 'dtype': np.float32},
            'reg_input': {'low': -np.inf, 'high': np.inf, 'dtype': np.float32},
            'lead_times': {'low': 0, 'high': 28, 'dtype': np.int32},
        })

        return spaces.Dict(
            {
                k: spaces.Box(low=box_values[k]['low'], high=box_values[k]['high'], shape=v.shape,
                              dtype=box_values[k]['dtype'])
                for k, v in initial_state.items()
            })
