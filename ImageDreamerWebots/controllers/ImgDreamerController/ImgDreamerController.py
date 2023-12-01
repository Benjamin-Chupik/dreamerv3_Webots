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
        self.timestep = 200
        self.maxspeed = 6.28
        self.epsilon = 0.15

        # create the Robot instance.
        self.robot = Robot()
        self.camera = Camera('camera')
        self.camera.enable(self.timestep)

        # Supervisor setup
        self.supervisor = Supervisor()
        self.robot_node = self.supervisor.getFromDef("_BEEPY_")
        self.robot_trans = self.robot_node.getField("translation")
        self.robot_rot = self.robot_node.getField("rotation")
        
        # Ball object
        self.ball = self.supervisor.getFromDef("BALL")
        self.ball_trans= self.ball.getField("translation")


        # Setting up  motors
        self.leftMotor = self.robot.getDevice('left wheel motor')
        self.rightMotor = self.robot.getDevice('right wheel motor')
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)

        # Obstacles
        self.Obs1 = self.supervisor.getFromDef("Obstacle1")
        self.Obs1_trans= self.Obs1.getField("translation")
        self.Obs1_pos = np.asarray(self.Obs1_trans.getSFVec3f())

        self.Obs2 = self.supervisor.getFromDef("Obstacle2")
        self.Obs2_trans= self.Obs2.getField("translation")
        self.Obs2_pos = np.asarray(self.Obs2_trans.getSFVec3f())

        self.Obs3 = self.supervisor.getFromDef("Obstacle3")
        self.Obs3_trans= self.Obs3.getField("translation")
        self.Obs3_pos = np.asarray(self.Obs3_trans.getSFVec3f())
        print(self.Obs3_pos)

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

        # Update timestep
        self.robot.step(self.timestep)
        self.timespent += self.timestep
        
        # write actuators inputs
        self.leftMotor.setVelocity(u[0] * self.maxspeed)
        self.rightMotor.setVelocity(u[1] * self.maxspeed)

        done = False
        if self.timespent > 2e4: # time in ms
            done = True
        
        # REWARDS
        self.robot_pos = np.asarray(self.robot_trans.getSFVec3f())
        self.d_to_goal = np.sqrt(np.sum((self.robot_pos-self.goal)**2))
        
        if self.d_to_goal < self.epsilon:
            self.reward = 1000
            done = True
        else:
            self.reward = 0

        self.reward += np.sum(self.img[:,:,0])/(32*32*255)
        self.reward+= self._obs_avoidance()
        
        return self._get_obs(), self.reward, done, {}
    

    def reset(self):

        self.goal = np.asarray(self.ball_trans.getSFVec3f())
        self.timespent = 0

        # reset robot
        self.robot_trans.setSFVec3f([0,0,0])
        self.robot_rot.setSFRotation([1,0,0,0])
        self.robot_node.resetPhysics()
        self.robot_pos = np.asarray(self.robot_trans.getSFVec3f())

        # reset ball
        corners = np.array([-0.45, 0.45])
        pos = np.random.choice(corners, 2)
        newpos = np.zeros(3)
        newpos[2] = 0.05
        newpos[:2] = pos
        self.ball_trans.setSFVec3f(list(newpos))

        # Get observation
        self.img = np.asarray(self.camera.getImageArray())
        np.save('beepyview', self.img)
        return self._get_obs()

    def _get_obs(self):
        return self.img
    
    def _obs_avoidance(self):
        total = 0
        d1 = self._distance(self.robot_pos, self.Obs1_pos)
        d2 = self._distance(self.robot_pos, self.Obs2_pos)
        d3 = self._distance(self.robot_pos, self.Obs3_pos)
        prox = self.epsilon

        if d1 < prox:
            total += -0.5*(1/d1-1/prox)**2
        if d2 < prox:
            total += -0.5*(1/d2-1/prox)**2
        if d3 < prox:
            total += -0.5*(1/d3-1/prox)**2
        return total
        

    def _distance(self, pos1, pos2):
        return np.sqrt(np.sum((pos1-pos2)**2))
    
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
