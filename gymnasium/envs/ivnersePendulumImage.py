## Class
import gym
from gym import spaces
import numpy as np
import cv2


class env(gym.Env):
    def __init__(self):
        super(env, self).__init__()
        self.env = gym.make("InvertedPendulum-v2")
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        self.action_space = self.env.action_space

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        image_observation = self.env.render(mode="rgb_array")
        image_observation = cv2.resize(
            image_observation, dsize=(64, 64), interpolation=cv2.INTER_CUBIC
        )
        return image_observation, reward, done, info

    def reset(self):
        obs = self.env.reset()
        image_observation = self.env.render(mode="rgb_array")
        image_observation = cv2.resize(
            image_observation, dsize=(64, 64), interpolation=cv2.INTER_CUBIC
        )
        return image_observation

    def render(self, mode="human"):
        if mode == "rgb_array":
            screen = self.env.render("rgb_array")
            return screen
        else:
            return self.env.render(mode)

    def close(self):
        self.env.close()

def defaultConfig(config):
    config = config.update(
        {
            "logdir": f"danijar_CodeBase/basicGymTesting/logdir/PendulumImageGPU",  # this was just changed to generate a new log dir every time for testing
            "run.train_ratio": 64,
            "run.log_every": 50,
            "batch_size": 8,
            "jax.prealloc": False,
            "encoder.mlp_keys": ".*",
            "decoder.mlp_keys": ".*",
            "encoder.cnn_keys": "image",
            "decoder.cnn_keys": "image",
            # "jax.platform": "cpu",  # I don't have a gpu locally
        }
    )
    return config
