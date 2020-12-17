'''
# Chapter 7 DQN Extensions

## Categorical DQN

* [source code](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter07/07_dqn_distrib.py)
* [Bellemare et al. 2017 Paper](https://arxiv.org/pdf/1707.06887v1.pdf)

### key idea
* Q value를  scalar value(기대값 형태의)가 아니라, 말그대로 가능한 Q value의 probability distribution으로 구하자는 아이디어.
* 참고 할 만한 리뷰 논문은 [Lowet et al. 2020](https://www.sciencedirect.com/science/article/pii/S0166223620301983)
'''

#%%
import gym
import ptan
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tensorflow as tf
import datetime

from lib import common
#%%
SAVE_STATES_IMG = False
SAVE_TRANSTIONS_IMG = False

if SAVE_STATES_IMG or SAVE_TRANSTIONS_IMG: # 그래프를 위한 code인듯
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pylab as plt # pyplot이 아님.
# %%
Vmax = 10
Vmin = -10
N_ATOMS = 51 # distribution의 bins number
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1) # size of bins

STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100
#%%
class DistributionalDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DistributionalDQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape) # conv가 아니라 input_shape이 들어가는 이유는 _get_conv_out function을 보면 이해 됨.
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions * N_ATOMS)
        )

        self.register_buffer("supports", torch.arange(Vmin, Vmax + DELTA_Z, DELTA_Z)) # histogram의 x축. 
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape)) # 변수명 앞의 *는 튜플의 인수를 풀어 각각의 매개변수로 함수로 전달한다. 
        return int(np.prod(o.size())) # np.prod 들어오는 array 원소들 간의 곱을 반환.
    
    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1) # tensor.view는 np.reshape과 유사함
        fc_out = self.fc(conv_out)
        return fc_out.view(batch_size, -1, N_ATOMS)

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports 
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())

#%%
def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net.qvals(states_v)
        best_action_values_v = action_values_v.max(1)[0] # torch.tensor에서 max는 최대값과 그것의 index를 반환한다. 
        # 매개변수로 받는 int는 max를 적용할 axis다.
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)
#%%
def save_state_images(frame_idx, states, net, device="cpu", max_states=200):
    ofs = 0
    p = np.arange(Vmin, Vmax + DELTA_Z, DELTA_Z)
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_prob = net.apply_softmax(net(states_v)).data.cpu().numpy()
        batch_size, num_actions, _ = action_prob.shape
        for batch_idx in range(batch_size):
            plt.clf() # clear current figure
            for action_idx in range(num_actions):
                plt.subplot(num_actions, 1, action_idx+1)
                plt.bar(p, action_prob[batch_idx, action_idx], width=0.5)
            plt.savefig("states/%05d_%08d.png" % (ofs + batch_idx, frame_idx))
        ofs += batch_size
        if ofs >= max_states:
            break
#%%
def save_transition_images(batch_size, predicted, projected, next_distr, dones, rewards, save_prefix):
    for batch_idx in range(batch_size):
        is_done = dones[batch_idx]
        reward = rewards[batch_idx]
        plt.clf()
        p = np.arange(Vmin, Vmax + DELTA_Z, DELTA_Z)
        plt.subplot(3, 1, 1)
        plt.bar(p, predicted[batch_idx], width=0.5)
        plt.title("Predicted")
        plt.subplot(3, 1, 2)
        plt.bar(p, projected[batch_idx], width=0.5)
        plt.title("Projected")
        plt.subplot(3, 1, 3)
        plt.bar(p, next_distr[batch_idx], width=0.5)
        plt.title("Next state")
        suffix = ""
        if reward != 0.0:
            suffix = suffix + "_%.0f" % reward
        if is_done:
            suffix = suffix + "_done"
        plt.savefig("%s_%02d%s.png" % (save_prefix, batch_idx, suffix))
# %%
def calc_loss(batch, net, tgt_net, gamma, device="cpu", save_prefix=None):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    batch_size = len(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.tensor(next_states).to(device)

    # next state distribution
    next_distr_v, next_qvals_v = tgt_net.both(next_states_v)
    next_actions = next_qvals_v.max(1)[1].data.cpu().numpy()
    next_distr = tgt_net.apply_softmax(next_distr_v).data.cpu().numpy()

    next_best_distr = next_distr[range(batch_size), next_actions]
    dones = dones.astype(np.bool)

    # project our distribution using Bellam update
    proj_distr = common.distr_projection(next_best_distr, rewards, dones, Vmin, Vmax, N_ATOMS, gamma)

    # calculate net output
    distr_v = net(states_v)
    state_action_values = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)
    proj_distr_v = torch.tensor(proj_distr).to(device)

    if save_prefix is not None:
        pred = F.softmax(state_action_values, dim=1).data.cpu().numpy()
        save_transition_images(batch_size, pred, proj_distr, next_best_distr, dones, rewards, save_prefix)

    loss_v = -state_log_sm_v * proj_distr_v
    return loss_v.sum(dim=1).mean()

# %%
if __name__ == "__main__":
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # it's convenient to record the date with current time
    LOGDIR = './tmp/ch7' + params['run_name'] + '/' + current_time + '/' 
    writer = tf.summary.create_file_writer(LOGDIR)

    net = DistributionalDQN(env.observation_space.shape, env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(lambda x: net.qvals(x), selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0
    eval_states = None
    prev_save = 0
    save_prefix = None

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params['replay_initial']:
                continue

            if eval_states is None:
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])

            save_prefix = None
            if SAVE_TRANSTIONS_IMG:
                interesting = any(map(lambda s: s.last_state is None or s.reward != 0.0, batch))
                if interesting and frame_idx // 30000 > prev_save:
                    save_prefix = "image/img_%08d" % frame_idx
                    prev_save = frame_idx // 30000

            loss_v = calc_loss(batch, net, tgt_net.target_model, gamma=params['gamma'], device=device, save_prefix=save_prefix)
            loss_v.backward()
            optimizer.step()

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()

            if frame_idx % EVAL_EVERY_FRAME == 0:
                mean_val = calc_values_of_states(eval_states, net, device=device)
                with writer.as_default():
                    tf.summary.scalar("value_mean", mean_val, frame_idx)

            if SAVE_STATES_IMG and frame_idx % 10000 == 0:
                save_state_images(frame_idx, eval_states, net, device=device)
        

    