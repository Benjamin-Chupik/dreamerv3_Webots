def main():
    import warnings
    import dreamerv3
    from dreamerv3 import embodied
    import os
    import sys
    # adds the parent directory to the path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  
    # imports the environment class
    import envs.ivnersePendulumImage as envFile
 

    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    # See configs.yaml  in dreamer for all options
    config = embodied.Config(dreamerv3.configs["defaults"])
    config = config.update(dreamerv3.configs["medium"])
    config = envFile.defaultConfig(config)
    config = config.update({"logdir": f"danijar_CodeBase/basicGymTesting/logdir/inversePendulumImage"})

    # innitial log written
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

    # make the ENV object
    env = envFile.env()
    from embodied.envs import from_gym
    env = from_gym.FromGym(
        env, obs_key="image"
    )  # I found I had to specify a different obs_key than the default of 'image'
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    # make the agent
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)

    # make the replay buffer
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / "replay"
    )

    # Set up training
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length,
    )

    # Train
    embodied.run.train(agent, env, replay, logger, args)
    # embodied.run.eval_only(agent, env, logger, args)


if __name__ == "__main__":
    main()
