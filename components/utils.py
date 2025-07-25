import os
import cv2
import gymnasium as gym
import torch
import random
import shutil
import logging

import numpy as np

from scipy.stats import entropy


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def set_seed(seed):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_results_dir(exp_path, args):
    # make the result directory
    os.makedirs(exp_path, exist_ok=True)
    if args.opr == "train" and os.path.exists(exp_path) and os.listdir(exp_path):
        if not args.force:
            raise FileExistsError(
                "{} is not empty. Please use --force to overwrite it".format(exp_path)
            )
        else:
            print("Warning, path exists! Rewriting...")
            shutil.rmtree(exp_path)
            os.makedirs(exp_path)
    log_path = os.path.join(exp_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, "model"), exist_ok=True)
    return exp_path, log_path


def init_logger(base_path):
    # initialize the logger
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s][%(filename)s>%(funcName)s] ==> %(message)s"
    )
    for mode in ["train", "test", "train_test", "root"]:
        file_path = os.path.join(base_path, mode + ".log")
        logger = logging.getLogger(mode)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.FileHandler(file_path, mode="a")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)


def select_action(visit_counts, temperature=1, deterministic=True):
    """select action from the root visit counts.
    Parameters
    ----------
    temperature: float

        the temperature for the distribution
    deterministic: bool
        True -> select the argmax
        False -> sample from the distribution
    """
    action_probs = [
        visit_count_i ** (1 / temperature) for visit_count_i in visit_counts
    ]
    total_count = sum(action_probs)
    action_probs = [x / total_count for x in action_probs]
    if deterministic:
        # best_actions = np.argwhere(visit_counts == np.amax(visit_counts)).flatten()
        # action_pos = np.random.choice(best_actions)
        action_pos = np.argmax([v for v in visit_counts])
    else:
        action_pos = np.random.choice(len(visit_counts), p=action_probs)

    count_entropy = entropy(action_probs, base=2)
    return action_pos, count_entropy


def prepare_observation_lst(observation_lst):
    """Prepare the observations to satisfy the input fomat of torch"""
    # B, S, N_obj, N_features
    observation_lst = np.array(observation_lst, dtype=np.float32)

    shape = observation_lst.shape
    # Return the observation in the shape of (B, N_obj, N_features) if S == 1
    if len(shape) == 4 and shape[1] == 1:
        return observation_lst[:, 0, :, :]
    return observation_lst
