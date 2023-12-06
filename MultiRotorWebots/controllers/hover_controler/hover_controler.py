from controller import Robot, DistanceSensor, Motor, Supervisor, InertialUnit
from controller import Camera

import numpy as np

# time in [ms] of a simulation step

# setting camera up
# feedback loop: step simulation until receiving an exit event

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import logging


# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

class HoverEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, hoverHeight=5):
        # INITIALIZING ROBOT
        self.timestep = 64
        self.timespent = 0
        # self.maxspeed = 6.28
        self.epsilon = 0.15
        self.maxMotorSpeed = 560

        # create the Robot instance.
        self.robot = Robot()
        self.camera = Camera('camera')
        self.camera.enable(self.timestep)

        # Supervisor setup
        self.supervisor = Supervisor()
        self.robot_node = self.supervisor.getFromDef("Mavic2PRO")
        self.robot_trans = self.robot_node.getField("translation")
        self.robot_rot = self.robot_node.getField("rotation")
        
        # set up sensors
        # self.compass = self.robot.getDevice("compass") # [E, N, U]  ( wb_compass_get_values())
        # self.gyro = self.robot.getDevice("gyro") # [roll accel, pitch, yaw accel] (roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()) rad/sec
        # self.gps = self.robot.getDevice("gps") # [x,y,z] (gps.getValues())
        # self.imu = self.robot.getDevice("inertial unit") # [roll, pitch, yaw] (imu.getRollPitchYaw())
        # self.sensors = [self.compass, self.gyro, self.gps, self.imu]
        # for sensor in self.sensors:
        #     sensor.enable(self.timestep)

        # Setting up  motors
        flMotor = self.robot.getDevice("front left propeller")
        frMotor = self.robot.getDevice("front right propeller")
        blMotor = self.robot.getDevice("rear left propeller")
        brMotor = self.robot.getDevice("rear right propeller")
        self.motors = [flMotor, frMotor, blMotor, brMotor]

        for motor in self.motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

        # Gym Space Setup
        self.action_space = spaces.Box(low=0, high=self.maxMotorSpeed,  shape=(4,), dtype=np.float32)
        # self.observation_space = spaces.Box(
        #     low=-500, high=500, shape=(4*3,), dtype=np.float32
        # )
        # self.seed()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(32, 32, 3), dtype=np.uint8
        )

        # set up goal
        self.goal = np.array([0,0,hoverHeight])

    # def _updateObs(self):
    #     obs = []
    #     # updates aall the sensors in the self.sensors list.
    #     for sensor in self.sensors:
    #         if isinstance(sensor, InertialUnit):
    #             value = sensor.getRollPitchYaw()  
    #         else:
    #             value = sensor.getValues()                  
    #         obs.append(value)

    #     obs = np.array(obs).flatten()

    #     nanMask = np.isnan(obs)
    #     if np.any(nanMask, axis=None):
    #         print("There are NAN observations!:")
    #         print("NAN at: ", nanMask)
    #         print("observation: ", obs)

    #     self.obs=obs
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        #print(u)
        # get the obeservation

        # Update timestep
        self.robot.step(self.timestep)
        self.timespent += self.timestep

        self.img = np.asarray(self.camera.getImageArray())

        # write actuators inputs
        for i in range(0,len(self.motors)):
            self.motors[i].setVelocity(u[i])

        if self.timespent > 3e4: # time in ms
            done = True
        else:
            done = False
        
        # REWARDS
        self.reward=0
        self.robot_pos = np.asarray(self.robot_trans.getSFVec3f())
        self.d_to_goal = np.sqrt(np.sum((self.robot_pos-self.goal)**2))
        # self._updateObs()

        if np.any(np.isnan(self.robot_pos)):
            done = True

        if self.d_to_goal < self.epsilon:
            self.reward += 100
            done = True
        
        # End the sim if it travledd farther than 20 meters away from goal (assume a bad crash)
        
        if self.d_to_goal > 20:
            self.reward -= 1000
            done=True


        # Add a -1 reward for every step to incentivise getting there fast
        self.reward+=-self.d_to_goal
        
        return self._get_obs(), self.reward, done, {}
    

    def reset(self):

        self.timespent = 0
        
        # reset motors
        for motor in self.motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

        # reset robot
        self.robot_node.resetPhysics()
        self.robot_trans.setSFVec3f([0,0,0.5])
        self.robot_rot.setSFRotation([1,0,0,0])
        self.robot_pos = np.asarray(self.robot_trans.getSFVec3f())
        print("Reset: New position:", self.robot_pos)

        # Get observation
        self.robot.step(self.timestep)
        self.img = np.asarray(self.camera.getImageArray())

        return self._get_obs()

    def _get_obs(self):
        #print(self.obs)
        return self.img
    
    def render(self, mode="human"):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None




""" Test ENV Code
try: 
    env = HoverEnv()
    obs = env.reset()
    print("Obs: ", obs)
    obs, reward, done, _ = env.step(np.array([500,500,500,500]))
    print("Obs: ", obs) 
    print("Reward: ", reward)
    print("done: ", done)

except Exception as e:
    # Log the error
    logging.error(f"An error occurred: {str(e)}", exc_info=True)
"""


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

env = HoverEnv() # 
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


# import warnings
# import dreamerv3
# from dreamerv3 import embodied


# logging.basicConfig(filename='error_log.txt', level=logging.ERROR, format='%(asctime)s - %(levelname)s: %(message)s')

# try:
#     warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

#     # See configs.yaml for all options.
#     config = embodied.Config(dreamerv3.configs["defaults"])
#     config = config.update(dreamerv3.configs["medium"])
#     config = config.update(
#         {
#             "logdir": f"logdir/testing1",  # this was just changed to generate a new log dir every time for testing
#             "run.train_ratio": 64,
#             "run.log_every": 30,
#             "batch_size": 16,
#             "jax.prealloc": False,
#             "encoder.mlp_keys": ".*",
#             "decoder.mlp_keys": ".*",
#             "encoder.cnn_keys": "$^",
#             "decoder.cnn_keys": "$^",
#             "jax.platform": "cpu",  # I don't have a gpu locally
#         }
#     )
#     config = embodied.Flags(config).parse()

#     logdir = embodied.Path(config.logdir)
#     step = embodied.Counter()
#     logger = embodied.Logger(
#         step,
#         [
#             embodied.logger.TerminalOutput(),
#             embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
#             embodied.logger.TensorBoardOutput(logdir),
#             # embodied.logger.WandBOutput(logdir.name, config),
#             # embodied.logger.MLFlowOutput(logdir.name),
#         ],
#     )


#     from embodied.envs import from_gym

#     env = HoverEnv() # 
#     env = from_gym.FromGym(
#         env, obs_key="state_vec"
#     )  # I found I had to specify a different obs_key than the default of 'image'
#     env = dreamerv3.wrap_env(env, config)
#     env = embodied.BatchEnv([env], parallel=False)

#     print("here---------------------------------------------")
#     agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
#     print("---------------------------------------------")
#     replay = embodied.replay.Uniform(
#         config.batch_length, config.replay_size, logdir / "replay"
#     )
#     args = embodied.Config(
#         **config.run,
#         logdir=config.logdir,
#         batch_steps=config.batch_size * config.batch_length,
#     )
#     embodied.run.train(agent, env, replay, logger, args)
# except Exception as e:
#     # Log the error
#     logging.error(f"An error occurred: {str(e)}", exc_info=True)
