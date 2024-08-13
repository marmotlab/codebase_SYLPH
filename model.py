import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
# from selfish_model import SelfishModel
from alg_parameters import *
from net import SCRIMPNet
import torch.nn as nn

from util import Loss

class Model(object):
    """model0 of agents"""

    def __init__(self, env_id, device, global_model=False):
        """initialization"""
        self.ID = env_id
        self.device = device
        self.network = SCRIMPNet().to(device)  # neural network
        # use MI Loss or not
        # self.selfish_model = SelfishModel(self.device)
        if global_model:
            self.net_optimizer = optim.Adam(self.network.parameters(), lr=TrainingParameters.lr)
            # self.multi_gpu_net = torch.nn.DataParallel(self.network) # training on multiple GPU
            self.net_scaler = GradScaler()  # automatic mixed precision

    def step(self, observation, vector, svo, comms_index, num_agent=EnvParameters.N_AGENTS):
        """using neural network in training for prediction"""
        observation = torch.from_numpy(observation).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        svo = torch.from_numpy(svo).to(self.device)
        comms_index = torch.from_numpy(comms_index).to(self.device)
        ps, v, block, _, output_state, _, svo_output = self.network(observation, vector, svo, comms_index)

        actions = np.zeros(num_agent)
        ps = np.squeeze(ps.cpu().detach().numpy())
        v = v.cpu().detach().numpy()  # intrinsic state values
        block = np.squeeze(block.cpu().detach().numpy())
        svo_output = np.squeeze(svo_output.cpu().detach().numpy())

        for i in range(num_agent):
            # choose action from complete action distribution
            actions[i] = np.random.choice(range(EnvParameters.N_ACTIONS), p=ps[i].ravel())
        return actions, ps, v, block, output_state, svo_output

    def evaluate(self, observation, vector, svo, comms_index, greedy=False, num_agent=EnvParameters.N_AGENTS):
        """using neural network in evaluations of training code for prediction"""
        eval_action = np.zeros(num_agent)
        observation = torch.from_numpy(np.asarray(observation)).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        svo = torch.from_numpy(svo).to(self.device)
        comms_index = torch.from_numpy(comms_index).to(self.device)
        ps, v, block, _, output_state, _, svo_output = self.network(observation, vector, svo, comms_index)

        ps = np.squeeze(ps.cpu().detach().numpy())
        block = np.squeeze(block.cpu().detach().numpy())
        greedy_action = np.argmax(ps, axis=-1)
        v = v.cpu().detach().numpy()
        svo_output = np.squeeze(svo_output.cpu().detach().numpy())

        for i in range(num_agent):
            if not greedy:
                eval_action[i] = np.random.choice(range(EnvParameters.N_ACTIONS), p=ps[i].ravel())
        if greedy:
            eval_action = greedy_action
        return eval_action, block, output_state, v, ps, svo_output

    def value(self, obs, vector, svo, comms_index):
        """using neural network to predict state values"""
        obs = torch.from_numpy(obs).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        svo = torch.from_numpy(svo).to(self.device)
        comms_index = torch.from_numpy(comms_index).to(self.device)
        _, v, _, _, _, _, _ = self.network(obs, vector, svo, comms_index)
        v = v.cpu().detach().numpy()
        return v

    def train(self, observation, vector, svo, svo_exe, comms_index, returns_svo, returns_action, returns, old_v, action,
              old_ps, train_valid, target_blockings):
        """train model0 by reinforcement learning"""
        self.net_optimizer.zero_grad()
        # from numpy to torch
        observation = torch.from_numpy(observation).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        svo = torch.from_numpy(svo).to(self.device)
        svo_exe = torch.from_numpy(svo_exe).to(self.device)
        svo_exe = torch.unsqueeze(svo_exe, -1)
        comms_index = torch.from_numpy(comms_index).to(self.device)
        returns_svo = torch.from_numpy(returns_svo).to(self.device)
        returns_action = torch.from_numpy(returns_action).to(self.device)
        returns = torch.from_numpy(returns).to(self.device)

        old_v = torch.from_numpy(old_v).to(self.device)

        action = torch.from_numpy(action).to(self.device)
        action = torch.unsqueeze(action, -1)
        old_ps = torch.from_numpy(old_ps).to(self.device)

        train_valid = torch.from_numpy(train_valid).to(self.device)
        target_blockings = torch.from_numpy(target_blockings).to(self.device)

        advantage_svo = returns_svo - old_v
        advantage_svo = (advantage_svo - advantage_svo.mean()) / (advantage_svo.std() + 1e-6)

        advantage_action = returns_action - old_v
        advantage_action = (advantage_action - advantage_action.mean()) / (advantage_action.std() + 1e-6)

        advantage = returns - old_v
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

        # dp_network = nn.DataParallel(self.network)

        with autocast():
            # new_ps_selfish = self.selfish_model.selfish_train_step(observation, vector)
            new_ps, new_v, block, policy_sig, _, _, svo_output = self.network(observation, vector, svo, comms_index)
            new_p = new_ps.gather(-1, action)
            old_p = old_ps.gather(-1, action)

            new_svo = svo_output.gather(-1, svo_exe)
            old_svo = svo.gather(-1, svo_exe)
            ratio = torch.exp(torch.log(torch.clamp(new_p, 1e-6, 1.0)) - torch.log(torch.clamp(old_p, 1e-6, 1.0)))
            ratio_svo = torch.exp(
                torch.log(torch.clamp(new_svo, 1e-6, 1.0)) - torch.log(torch.clamp(old_svo, 1e-6, 1.0)))

            entropy = torch.mean(-torch.sum(new_ps * torch.log(torch.clamp(new_ps, 1e-6, 1.0)), dim=-1, keepdim=True))
            entropy_svo = torch.mean(
                -torch.sum(svo_output * torch.log(torch.clamp(svo_output, 1e-6, 1.0)), dim=-1, keepdim=True))

            # critic loss
            new_v = torch.squeeze(new_v)
            new_v_clipped = old_v+ torch.clamp(new_v - old_v, - TrainingParameters.CLIP_RANGE,
                                               TrainingParameters.CLIP_RANGE)
            value_losses1 = torch.square(new_v - returns)
            value_losses2= torch.square(new_v_clipped - returns)
            critic_loss = torch.mean(torch.maximum(value_losses1, value_losses2))

            # todo: swap the advantage_svo and advantage_action
            # actor loss
            ratio = torch.squeeze(ratio)
            policy_losses = advantage_svo * ratio
            policy_losses2 = advantage_svo * torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE,
                                                     1.0 + TrainingParameters.CLIP_RANGE)
            policy_loss = torch.mean(torch.min(policy_losses, policy_losses2))

            # svo policy loss
            ratio_svo = torch.squeeze(ratio_svo)
            svo_losses = advantage_action * ratio_svo
            svo_losses2 = advantage_action * torch.clamp(ratio_svo, 1.0 - TrainingParameters.CLIP_RANGE,
                                                         1.0 + TrainingParameters.CLIP_RANGE)
            svo_loss = torch.mean(torch.min(svo_losses, svo_losses2))

            # valid loss and blocking loss decreased by supervised learning
            valid_loss = - torch.mean(torch.log(torch.clamp(policy_sig, 1e-6, 1.0 - 1e-6)) *
                                      train_valid + torch.log(torch.clamp(1 - policy_sig, 1e-6, 1.0 - 1e-6)) * (
                                              1 - train_valid))
            block = torch.squeeze(block)
            blocking_loss = - torch.mean(target_blockings * torch.log(torch.clamp(block, 1e-6, 1.0 - 1e-6))
                                         + (1 - target_blockings) * torch.log(torch.clamp(1 - block, 1e-6, 1.0 - 1e-6)))

            # # MI LOSS
            # coupling_loss = - torch.mean(new_ps_selfish * torch.log(torch.clamp(new_ps, 1e-6, 1.0 - 1e-6))
            #                              + (1 - new_ps_selfish) * torch.log(torch.clamp(1 - new_ps, 1e-6, 1.0 - 1e-6)))
            # greedy_svo = (5 * torch.argmax(svo_output, dim=2, keepdim=True) / 180 * torch.pi)
            # coef_svo = torch.mean(torch.sin(greedy_svo))
            # coupling_loss = coef_svo * coupling_loss

            # total loss
            all_loss = -policy_loss - entropy * TrainingParameters.ENTROPY_COEF + \
                TrainingParameters.VALUE_COEF * critic_loss  + TrainingParameters.VALID_COEF * valid_loss \
                + TrainingParameters.BLOCK_COEF * blocking_loss - svo_loss  \
                - TrainingParameters.ENTROPY_COEF * entropy_svo
                # + (-0.001 * coupling_loss)    # C MI loss

        clip_frac = torch.mean(torch.greater(torch.abs(ratio - 1.0), TrainingParameters.CLIP_RANGE).float())

        self.net_scaler.scale(all_loss).backward()
        self.net_scaler.unscale_(self.net_optimizer)

        # Clip gradient
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)

        self.net_scaler.step(self.net_optimizer)
        self.net_scaler.update()


        loss = Loss()
        loss.all_loss = all_loss.cpu().detach().numpy()
        loss.policy_loss = policy_loss.cpu().detach().numpy()
        loss.policy_entropy = entropy.cpu().detach().numpy()
        loss.critic_loss = critic_loss.cpu().detach().numpy()
        loss.blocking_loss = blocking_loss.cpu().detach().numpy()
        loss.valid_loss = valid_loss.cpu().detach().numpy()
        loss.clipfrac = clip_frac.cpu().detach().numpy()
        loss.grad_norm = grad_norm.cpu().detach().numpy()
        loss.advantage = torch.mean(advantage).cpu().detach().numpy()
        
        return loss

    def set_weights(self, weights):
        """load global weights to local models"""
        self.network.load_state_dict(weights)
