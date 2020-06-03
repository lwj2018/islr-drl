import numpy as np
import torch
import os

from utils import ReplayBuffer
from utils.ioUtils import resume_model
from models import DDPG,TD3
from models import lstm
from datasets import CSL_Isolated_Openpose
from environment import Environment
import argparse
import time

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env, seed, videoInd, eval_episodes=5):
    eval_env = env
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for i in range(eval_episodes):
        print(f"Eval episode {i} / {eval_episodes}")
        start = time.time()
        state, done = eval_env.reset(videoInd), False
        while not done:
            action = policy.select_action(state)
            state, reward, done = eval_env.step(action,videoInd)
            avg_reward += reward
        end = time.time()

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Env {videoInd}, Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DDPG")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="video")                   # Our implementation of video enironment
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=2e3, type=int) # Time steps initial random policy is used, 25e3
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=16, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", default=1,type=int)        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--num_envs",default=18000,type=int)
    args = parser.parse_args()
    # Options
    state_dim =1024
    action_dim = 3 
    length = 32
    num_joints = 116
    device_list = '3'
    checkpoint = './checkpoint/20200513_LSTM_isolated_best.pth.tar'

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")

    # Use specific gpus
    os.environ["CUDA_VISIBLE_DEVICES"]=device_list
    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # init base network
    model = lstm(input_size=num_joints*2).to(device)
    resume_model(model,checkpoint)
    # init dataset
    dataset = CSL_Isolated_Openpose('trainval')

    env = Environment(model,dataset)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise
        kwargs["noise_clip"] = args.noise_clip
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./checkpoint/{policy_file}")

    replay_buffer = ReplayBuffer(state_dim,action_dim,length)
    
    # Random shuffle envs
    random_list = np.arange(args.num_envs)
    np.random.shuffle(random_list)
    videoInd = random_list[0]
    # Evaluate untrained policy
    evaluations = []
    # evaluations = [eval_policy(policy, env, args.seed, videoInd)]

    state, done = env.reset(videoInd), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for videoInd in [random_list[0]]:
        state = env.reset(videoInd)
        for t in range(int(args.max_timesteps)):
            
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                action = env.sample()
            else:
                action = (
                    policy.select_action(state)
                    + np.random.normal(0, 1 * args.expl_noise, size=(length,action_dim))
                )

            # Perform action
            next_state, reward, done = env.step(action,videoInd) 

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, float(done))
            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)

            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {t+1} Episode Num: {episode_num+1} \
                        Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Env: {videoInd}")
                # Reset environment
                state, done = env.reset(videoInd), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            # Evaluate episode
            if (t + 1) % args.eval_freq == 0:
                evaluations.append(eval_policy(policy, env, args.seed, videoInd))
                np.save(f"./results/{file_name}", evaluations)
                if args.save_model: policy.save(f"./checkpoint/{file_name}")