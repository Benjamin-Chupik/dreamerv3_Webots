from controller import Robot, DistanceSensor, Motor, Supervisor
from controller import Camera

import matplotlib.pyplot as plt
import numpy as np

# time in [ms] of a simulation step

# setting camera up
# feedback loop: step simulation until receiving an exit event

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

class PendulumEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self):
        # INITIALIZING ROBOT
        self.timestep = 500
        self.maxspeed = 6.28

        # create the Robot instance.
        self.robot = Robot()
        self.camera = Camera('camera')
        self.camera.enable(self.timestep)

        # Supervisor setup
        self.supervisor = Supervisor()
        self.robot_node = self.supervisor.getFromDef("_BEEPY_")
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field = self.robot_node.getField("rotation")

        self.ball = self.supervisor.getFromDef("BALL")
        self.ball_trans= self.ball.getField("translation")

        # print(self.robot_node)
        # initialize devices
        # self.ps = []
        # self.psNames = [
        #     'ps0', 'ps1', 'ps2', 'ps3',
        #     'ps4', 'ps5', 'ps6', 'ps7'
        # ]

        # for i in range(8):
        #     self.ps.append(self.robot.getDevice(self.psNames[i]))
        #     self.ps[i].enable(self.timestep)

        self.leftMotor = self.robot.getDevice('left wheel motor')
        self.rightMotor = self.robot.getDevice('right wheel motor')
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)


        # SPACEY
        self.action_space = spaces.Box(low=-1, high=1,  shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(32, 32, 3), dtype=np.uint8
        )
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        ## SNEAKY WEBOTS STEP
        self.img = np.asarray(self.camera.getImageArray())

        self.robot.step(self.timestep)
        self.timespent += self.timestep
        print(self.timespent)
        
        # write actuators inputs
        self.leftMotor.setVelocity(u[0] * self.maxspeed)
        self.rightMotor.setVelocity(u[1] * self.maxspeed)

        done = False
        if self.timespent > 2e4: # time in ms
            done = True
        
        # REWARDS
        self.robot_pos = np.asarray(self.trans_field.getSFVec3f())
        self.reward = 1/np.sum((self.robot_pos-self.goal)**2)

        return self._get_obs(), self.reward, done, {}

    def reset(self):

        self.goal = np.asarray(self.ball_trans.getSFVec3f())

        self.img = np.zeros((32,32,3), dtype=np.uint8)
        self.timespent = 0

        # reset robot
        self.trans_field.setSFVec3f([0,0,0])
        self.rot_field.setSFRotation([1,0,0,0])
        self.robot_node.resetPhysics()
        
        return self._get_obs()

    def _get_obs(self):
        return self.img
    
    def render(self, mode="human"):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

################################# DREAMER TIME ################################


import warnings
import dreamerv3
from dreamerv3 import embodied

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

# See configs.yaml for all options.
config = embodied.Config(dreamerv3.configs["defaults"])
config = config.update(dreamerv3.configs["medium"])
config = config.update(
    {
        "logdir": f"logdirtest",  # this was just changed to generate a new log dir every time for testing
        "run.train_ratio": 64,
        "run.log_every": 30,
        "batch_size": 16,
        "jax.prealloc": False,
        "encoder.mlp_keys": ".*",
        "decoder.mlp_keys": ".*",
        "encoder.cnn_keys": "image",
        "decoder.cnn_keys": "image",
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


import gym
from embodied.envs import from_gym

env = PendulumEnv() # 
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
