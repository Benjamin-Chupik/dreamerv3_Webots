import gym
from gym import spaces
import numpy as np
import cv2


class LunarLanderImageEnv(gym.Env):
    def __init__(self):
        super(LunarLanderImageEnv, self).__init__()
        self.env = gym.make("LunarLander-v2")
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        self.action_space = self.env.action_space

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        image_observation = self.render(mode="rgb_array")
        return image_observation, reward, done, info

    def reset(self):
        obs = self.env.reset()
        image_observation = self.render(mode="rgb_array")
        return image_observation

    def render(self, mode="human"):
        if mode == "rgb_array":
            screen = self.env.render("rgb_array")
            return screen
        else:
            return self.env.render(mode)

    def close(self):
        self.env.close()


if __name__ == "__main__":
    env = LunarLanderImageEnv()
    env2 = gym.make("LunarLander-v2")

    print(env.action_space)
    print(env2.action_space)
    # observation = env.reset()
    # done = False
    # while not done:
    #     action = env.action_space.sample()
    #     observation, reward, done, info = env.step(action)
    #     print(action)

    # env.close()
    # # cv2.imshow("Lunar Lander Image", observation)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # print(env.action_space)
    # print(action)
