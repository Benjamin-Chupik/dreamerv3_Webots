from controller import Robot, DistanceSensor, Motor, Supervisor
from controller import Camera
import logging

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

    def __init__(self, gamma=0.997):
        self.gamma = gamma
        
        # INITIALIZING ROBOT
        self.timestep = 200
        self.maxtime = 3*60e3
        self.maxspeed = 6.28
        self.epsilon = 0.5

        # create the Robot instance.
        self.robot = Robot()
        self.camera = Camera('camera')
        self.camera.enable(self.timestep)

        # Supervisor setup
        self.supervisor = Supervisor()
        self.robot_node = self.supervisor.getFromDef("_BEEPY_")
        self.robot_trans = self.robot_node.getField("translation")
        self.robot_rot = self.robot_node.getField("rotation")
        self.robot_node.enableContactPointsTracking(self.timestep)

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

        # Pedestrians
        self.pedestrian_range = 0.75

        # Pedestrian 1
        self.ped1 = self.supervisor.getFromDef("PED1")
        self.ped1_trans= self.ped1.getField("translation")
        self.ped1_rot = self.ped1.getField("rotation")
        self.ped1_cargs = self.ped1.getField("controllerArgs")
        self.ped1_pos = np.asarray(self.ped1_trans.getSFVec3f())

        # Pedestrian 2
        self.ped2 = self.supervisor.getFromDef("PED2")
        self.ped2_trans= self.ped2.getField("translation")
        self.ped2_rot = self.ped2.getField("rotation")
        self.ped2_pos = np.asarray(self.ped2_trans.getSFVec3f())
        self.ped2_cargs = self.ped2.getField("controllerArgs")


        # SPACEY
        self.action_space = spaces.Box(low=-1, high=1,  shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        lastPos = np.asarray(self.robot_trans.getSFVec3f())
        # Update timestep
        self.robot.step(self.timestep)
        self.timespent += self.timestep

        curPos = np.asarray(self.robot_trans.getSFVec3f())

        # udpate
        self.ped1_pos = np.asarray(self.ped1_trans.getSFVec3f())
        self.ped2_pos = np.asarray(self.ped2_trans.getSFVec3f())

        # write actuators inputs
        self.leftMotor.setVelocity(u[0] * self.maxspeed)
        self.rightMotor.setVelocity(u[1] * self.maxspeed)

        # update robot position
        self.robot_pos = np.asarray(self.robot_trans.getSFVec3f())
        self.d_to_goal = self._distance(self.robot_pos, self.goal)
        
        # check if done
        done = False

        if self.timespent > self.maxtime: # time in ms
            self.reward = 0
            done = True
        if self.d_to_goal < self.epsilon:
            self.reward = 100
            done = True
        else:
            self.reward = 0
        
        # Reward Shaping
        philp = self._phi(lastPos) 
        phicp = self._phi(curPos)
        rewardShapeTerm = philp-self.gamma*phicp
        self.reward += rewardShapeTerm

        # REWARDS
        self.reward += self._obs_avoidance()
        self.reward -= 0.2

        return self._get_obs(), self.reward, done, {}
    
    def _phi(self, pos): # distance to goal
        return 10*self._distance(pos, self.goal)

    def reset(self):
        self.robot.step(self.timestep)
        self.timespent = 0
        
        # somewhat random trajectories
        xperts = np.random.rand(4)
        if np.random.rand() <0.5:
            ypert = 1
        else:
            ypert = -1

        self.ped1_cargs.setMFString(0, f"--trajectory={-1.5+xperts[0]} {ypert*-2.25}, {-1.5+xperts[1]} {ypert*2.25}")
        self.ped1_cargs.setMFString(1, "--speed=0.1")
        
        self.ped2_cargs.setMFString(0, f"--trajectory={0.5+xperts[2]} {ypert*2.25}, {0.5+xperts[3]} {ypert*-2.25}")
        self.ped2_cargs.setMFString(1, "--speed=0.1")

        # update pedestrian position vectors
        self.ped1_pos = np.asarray(self.ped1_trans.getSFVec3f())
        self.ped2_pos = np.asarray(self.ped2_trans.getSFVec3f())

        
        self.ped1_trans.setSFVec3f([self.ped1_pos[0],self.ped1_pos[1],1.27])
        self.ped2_trans.setSFVec3f([self.ped2_pos[0],self.ped2_pos[1],1.27])

        self.ped1_pos = np.asarray(self.ped1_trans.getSFVec3f())
        self.ped2_pos = np.asarray(self.ped2_trans.getSFVec3f())

        # reset robot
        self.robot_trans.setSFVec3f([-2.25,0,0])
        self.robot_rot.setSFRotation([0,0,1,0])
        self.robot_node.resetPhysics()
        self.robot_pos = np.asarray(self.robot_trans.getSFVec3f())

        
        # randomize ball pos
        xpos = np.array([0, 2.2])
        ypos = np.array([-2.2, 2.2])
        newpos = np.zeros(3)
        newpos[0] = np.random.choice(xpos, 1)[0]
        newpos[1] = np.random.choice(ypos, 1)[0]
        newpos[2] = 0.3
        
        # set ball pos
        self.ball_trans.setSFVec3f(list(newpos))

        # set goal value
        self.goal = np.asarray(self.ball_trans.getSFVec3f())
        # Get observation
        self.robot.step(self.timestep)
        
        # np.save('beepyview', self.img)
        return self._get_obs()

    def _get_obs(self):
        self.img = np.asarray(self.camera.getImageArray()).astype(np.uint8)
        if np.sum(self.img) == 0:
            print('uh oh')
        return self.img
    
    def _obs_avoidance(self):
        total = 0
        d1 = self._distance(self.robot_pos, self.ped1_pos)
        d2 = self._distance(self.robot_pos, self.ped2_pos)

        if d1 < self.pedestrian_range or d2 < self.pedestrian_range:
            total += -1

        return total
        

    def _distance(self, pos1, pos2):
        #xy distance
        return np.sqrt(np.sum((pos1[:2]-pos2[:2])**2))
    
    def render(self, mode="human"):
        return self.img

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

################################# DREAMER TIME ################################
logging.basicConfig(filename='error_log.txt', level=logging.ERROR, format='%(asctime)s - %(levelname)s: %(message)s')

try:

    import warnings
    import dreamerv3
    from dreamerv3 import embodied

    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs["defaults"])
    config = config.update(dreamerv3.configs["large"])
    config = config.update(
        {
            "logdir": f"logdir/finalHope",  # this was just changed to generate a new log dir every time for testing
            "run.train_ratio": 64,
            "run.log_every": 30,
            "batch_size": 8,
            "jax.prealloc": False,
            "encoder.mlp_keys": ".*",
            "decoder.mlp_keys": ".*",
            "encoder.cnn_keys": "image",
            "decoder.cnn_keys": "image",
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

    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length,
    )
    
    isTrain=True

    if isTrain:
        replay = embodied.replay.Uniform(
            config.batch_length, config.replay_size, logdir / "replay"
            )
        embodied.run.train(agent, env, replay, logger, args)
    else:
        args = args.update({'from_checkpoint':'logdir/Lasthope/checkpoint.ckpt'
            })
        #env.render()
        embodied.run.eval_only(agent, env, logger, args)

except Exception as e:
    # Log the error
    logging.error(f"An error occurred: {str(e)}", exc_info=True)
