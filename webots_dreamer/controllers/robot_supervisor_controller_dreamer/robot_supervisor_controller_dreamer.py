def main():
    from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
    from utilities import normalize_to_range

    from gym.spaces import Box, Discrete
    import numpy as np

    class CartpoleRobot(RobotSupervisorEnv):
        def __init__(self):
            super().__init__()
            
            # setting up obseration and action space
            self.observation_space = Box(low=np.array([-0.4, -np.inf, -1.3, -np.inf]),
                                        high=np.array([0.4, np.inf, 1.3, np.inf]),
                                        dtype=np.float64)
            self.action_space = Discrete(2)
            
            # Setting up robot sensors
            self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
            self.position_sensor = self.getDevice("polePosSensor")
            self.position_sensor.enable(self.timestep)
            self.pole_endpoint = self.getFromDef("POLE_ENDPOINT")

            self.wheels = []
            for wheel_name in ['wheel1', 'wheel2', 'wheel3', 'wheel4']:
                wheel = self.getDevice(wheel_name)  # Get the wheel handle
                wheel.setPosition(float('inf'))  # Set starting position
                wheel.setVelocity(0.0)  # Zero out starting velocity
                self.wheels.append(wheel)
            
            # initializing training variables    
            self.steps_per_episode = 200  # Max number of steps per episode
            self.episode_score = 0  # Score accumulated during an episode
            self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        
        def get_observations(self):
            # Position on x-axis
            cart_position = normalize_to_range(self.robot.getPosition()[0], -0.4, 0.4, -1.0, 1.0)
            # Linear velocity on x-axis
            cart_velocity = normalize_to_range(self.robot.getVelocity()[0], -0.2, 0.2, -1.0, 1.0, clip=True)
            # Pole angle off vertical
            pole_angle = normalize_to_range(self.position_sensor.getValue(), -0.23, 0.23, -1.0, 1.0, clip=True)
            # Angular velocity y of endpoint
            endpoint_velocity = normalize_to_range(self.pole_endpoint.getVelocity()[4], -1.5, 1.5, -1.0, 1.0, clip=True)

            return [cart_position, cart_velocity, pole_angle, endpoint_velocity]
        
        def get_default_observation(self):
            # This method just returns a zero vector as a default observation
            return [0.0 for _ in range(self.observation_space.shape[0])]
        
        def get_reward(self, action=None):
            return 1
            
        def is_done(self):
            if self.episode_score > 195.0:
                return True

            pole_angle = round(self.position_sensor.getValue(), 2)
            if abs(pole_angle) > 0.261799388:  # more than 15 degrees off vertical (defined in radians)
                return True

            cart_position = round(self.robot.getPosition()[0], 2)  # Position on x-axis
            if abs(cart_position) > 0.39:
                return True

            return False
            
        def solved(self):
            if len(self.episode_score_list) > 100:  # Over 100 trials thus far
                if np.mean(self.episode_score_list[-100:]) > 195.0:  # Last 100 episodes' scores average value
                    return True
            return False
        
        def get_info(self):
            return None

        def render(self, mode='human'):
            pass
            
        def apply_action(self, action):
            print(action)
            action = int(action)

            if action == 0:
                motor_speed = 5.0
            else:
                motor_speed = -5.0

            for i in range(len(self.wheels)):
                self.wheels[i].setPosition(float('inf'))
                self.wheels[i].setVelocity(motor_speed)

    ###################### DREAMER TIME ########################
    import warnings
    import dreamerv3
    from dreamerv3 import embodied

    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs["defaults"])
    config = config.update(dreamerv3.configs["medium"])
    config = config.update(
        {
            "logdir": f"logdir/pendulum",  # this was just changed to generate a new log dir every time for testing
            "run.train_ratio": 64,
            "run.log_every": 30,
            "batch_size": 16,
            "jax.prealloc": False,
            "encoder.mlp_keys": ".*",
            "decoder.mlp_keys": ".*",
            "encoder.cnn_keys": "$^",
            "decoder.cnn_keys": "$^",
            "jax.platform": "cpu",  # I don't have a gpu locally
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

    from embodied.envs import from_gym

    # env = crafter.Env()  # Replace this with your Gym env.
    env = CartpoleRobot()
    env = from_gym.FromGym(
        env, obs_key="state_vec"
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

if __name__ == "__main__":
    main()
