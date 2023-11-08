def main():
    import warnings
    import dreamerv3
    from dreamerv3 import embodied

    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs["defaults"])
    config = config.update(dreamerv3.configs["medium"])
    config = config.update(
        {
            "logdir": f"basicGymTesting/logdir/LunarLanderImageGPU",  # this was just changed to generate a new log dir every time for testing
            "run.train_ratio": 64,
            "run.log_every": 30,
            "batch_size": 16,
            "jax.prealloc": False,
            "encoder.mlp_keys": ".*",
            "decoder.mlp_keys": ".*",
            "encoder.cnn_keys": "$^",
            "decoder.cnn_keys": "$^",
            # "jax.platform": "cpu",  # I don't have a gpu locally
        }
    )
    config = embodied.Flags(config).parse()

    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
            # embodied.logger.WandBOutput(logdir.name, config),
            # embodied.logger.MLFlowOutput(logdir.name),
        ],
    )

    # import crafter
    import gym

    # from ..envs import LunarLanderImage
    from embodied.envs import from_gym

    # env = crafter.Env()  # Replace this with your Gym env.
    env = LunarLanderImageEnv  # this needs box2d-py installed also
    # env = gym.make("LunarLander-v2")
    env = from_gym.FromGym(
        env, obs_key="image"
    )  # I found I had to specify a different obs_key than the default of 'image'
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    print("here---------------------------------------------")
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    print("---------------------------------------------")
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / "replay"
    )
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length,
    )
    embodied.run.train(agent, env, replay, logger, args)
    # embodied.run.eval_only(agent, env, logger, args)


## Class
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
    main()
