import numpy as np
import gym
from matplotlib.pyplot import imread


def make_env(env_name, seed=-1, render_mode=False):
  # -- Bullet Environments ------------------------------------------- -- #
  if "Bullet" in env_name:
    import pybullet as p # pip install pybullet
    import pybullet_envs
    import pybullet_envs.bullet.kukaGymEnv as kukaGymEnv

  # -- Lunar Lander ------------------------------------------------ -- #
  elif (env_name.startswith("LunarLander")):
    if (env_name.startswith("LunarLanderHardcore")):
      import Box2D
      from domain.lunar_lander import LunarLanderHardcore
      env = LunarLanderHardcore()
    elif (env_name.startswith("LunarLanderMedium")):
      from domain.lunar_lander import LunarLander
      env = LunarLander()
      env.accel = 3
    else:
      from domain.lunar_lander import LunarLander
      env = LunarLander()

  # -- Cart Pole Swing up -------------------------------------------- -- #
  elif (env_name.startswith("CartPoleSwingUp")):
    from domain.cartpole_swingup import CartPoleSwingUpEnv
    env = CartPoleSwingUpEnv()
    if (env_name.startswith("CartPoleSwingUp_Hard")):
      env.dt = 0.01
      env.t_limit = 200

  # -- Other  -------------------------------------------------------- -- #
  else:
    env = gym.make(env_name)

  if (seed >= 0):
    domain.seed(seed)

  return env
