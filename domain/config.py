from collections import namedtuple
import numpy as np

Game = namedtuple('Game', ['env_name', 'time_factor', 'actionSelect',
  'input_size', 'output_size', 'layers', 'i_act', 'h_act',
  'o_act', 'weightCap','noise_bias','output_noise','max_episode_length','in_out_labels'])

games = {}


# -- Cart-pole Swingup --------------------------------------------------- -- #

# > Slower reaction speed
cartpole_swingup = Game(env_name='CartPoleSwingUp_Hard',
  actionSelect='all', # all, soft, hard
  input_size=5,
  output_size=1,
  time_factor=0,
  layers=[5, 5],
  i_act=np.full(5,1),
  h_act=[1,2,3,4,5,6,7,8,9,10],
  o_act=np.full(1,1),
  weightCap = 2.0,
  noise_bias=0.0,
  output_noise=[False, False, False],
  max_episode_length = 200,
  in_out_labels = ['x','x_dot','cos(theta)','sin(theta)','theta_dot',
                   'force']
)
games['swingup_hard'] = cartpole_swingup

# > Normal reaction speed
cartpole_swingup = cartpole_swingup._replace(\
    env_name='CartPoleSwingUp', max_episode_length=1000)
games['swingup'] = cartpole_swingup

# -- Lunar Lander ------------------------------------------------------ -- #

# > Flat terrain
lunar = Game(env_name='LunarLander-v2',
  actionSelect='all', # all, soft, hard
  input_size=8,
  output_size=2,
  time_factor=0,
  layers=[40, 40],
  i_act=np.full(8,1),
  h_act=[1,2,3,4,5,6,7,8,9,10],
  o_act=np.full(2,1),
  weightCap = 2.0,
  noise_bias=0.0,
  output_noise=[False, False, False],
  max_episode_length = 400,
  in_out_labels = [
  'pos_x','pos_y','vel_x','vel_y',
  'angle','angular_vel','first_leg_attach','second_leg_attach',
  'Main engine','Orientation engines']
)
games['lunar'] = lunar

# > Hilly Terrain
lunarmed = lunar._replace(env_name='LunarLanderMedium-v2')
games['lunarmedium'] = lunarmed

# > Obstacles, hills, and pits
lunarhard = lunar._replace(env_name='LunarLanderHardcore-v2')
games['lunarhard'] = lunarhard

