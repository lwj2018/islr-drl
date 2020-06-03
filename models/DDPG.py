import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,
        hidden_size=256,num_layers=1,bidirectional=True):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional)
        if bidirectional: hidden_size = 2*hidden_size
        self.fc1 = nn.Linear(hidden_size,256)
        self.fc2 = nn.Linear(256,action_dim)
    
    def forward(self, state):
        if not hasattr(self, '_flattened'):
            self.lstm.flatten_parameters()
            setattr(self, '_flattened', True)
        # shape of state is: N x T x 1024
        # RNN forward
        a, (h_n,c_n) = self.lstm(state)
        # Fc foward
        a = F.relu(self.fc1(a))
        a = self.fc2(a)
        # shape of a is: N x T x 3
        a = F.softmax(a,-1)
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,
        hidden_size=256,num_layers=1,bidirectional=True):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(input_size=state_dim+action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional)
        if bidirectional: hidden_size = 2*hidden_size
        self.fc1 = nn.Linear(hidden_size,256)
        self.fc2 = nn.Linear(256,1)


    def forward(self, state, action):
        if not hasattr(self, '_flattened'):
            self.lstm.flatten_parameters()
            setattr(self, '_flattened', True)
        # shape of state is: N x T x 1024
        # shape of action is: N x T x 3
        # RNN forward
        q, (h_n,c_n) = self.lstm(torch.cat([state, action], -1))
        # Fc foward
        q = F.relu(self.fc1(q))
        q = self.fc2(q)
        # shape of q is: N x 1
        q = q.mean(dim=1)
        return q


class DDPG(object):
    def __init__(self, state_dim, action_dim, discount=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.discount = discount
        self.tau = tau


    def select_action(self, state):
        state = torch.Tensor(state).unsqueeze(0).cuda()
        return self.actor(state).cpu().data.numpy().squeeze(0)


    def train(self, replay_buffer, batch_size=100):
        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)