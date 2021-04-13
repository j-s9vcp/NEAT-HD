import os
import sys
import time
import math
import datetime
import argparse
import subprocess
import numpy as np
from multiprocessing import Pool
np.set_printoptions(precision=2, linewidth=160)

# # MPI
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# prettyNeat
from neat_src import * # NEAT
from domain import *   # Task environments

generation = 300
iter_num = 50
output_dir = 'log/history_control_scenario2'


# -- Run NEAT ------------------------------------------------------------ -- #
def master():
  """Main NEAT optimization script
  """
  global fileName, hyp, generation, rep_type, gen

  start_t = datetime.datetime.now()
  print('started at ', start_t)

  for iter_i in range(iter_num):
      print('iter ', iter_i)
      make_new_file(iter_i)

      data = DataGatherer(fileName, hyp)
      neat = Neat(hyp, rep_type)

      for gen in range(generation):
          pop = neat.ask()            # Get newly evolved individuals from NEAT
          reward = batchMpiEval(pop, gen=neat.gen, sp_count=len(neat.species))  # Send pop to be evaluated by workers
          neat.tell(reward)           # Send fitness to NEAT
          neat.probMoo()      # Rank population according to objectivess
          neat.speciate()     # Divide population into species

          data = gatherData(data,neat,gen,iter_i,hyp,savePop=True)
          print(gen, '\t - \t', data.display())
          print('\t', len(neat.species))

          neat.gen += 1

      # Clean up and data gathering at run end
      data = gatherData(data,neat,gen,iter_i,hyp,savePop=False)
      data.save()
      data.savePop(neat.pop,fileName) # Save population as 2D numpy arrays

  print('finish at ', datetime.datetime.now())
  print('total time ', datetime.datetime.now()-start_t)

  stopAllWorkers()

def make_new_file(iter_i):
    pref = output_dir + '/iter_{}'.format(iter_i)
    os.mkdir(pref)

def gatherData(data,neat,gen,iter_i,hyp,savePop=False):
  """Collects run data, saves it to disk, and exports pickled population

  Args:
    data       - (DataGatherer)  - collected run data
    neat       - (Neat)          - neat algorithm container
      .pop     - [Ind]           - list of individuals in population
      .species - (Species)       - current species
    gen        - (ind)           - current generation
    hyp        - (dict)          - algorithm hyperparameters
    savePop    - (bool)          - save current population to disk?

  Return:
    data - (DataGatherer) - updated run data
  """
  data.gatherData(neat.pop, neat.species)
  if (gen%hyp['save_mod']) == 0:
    data = checkBest(data)
    data.save(gen)

  if savePop is True: # Get a sample pop to play with in notebooks
    global fileName
    pref = output_dir + '/iter_{}'.format(iter_i) + '/gen_' + str(gen).zfill(4)
    import pickle
    with open(pref+'.obj', 'wb') as fp:
      pickle.dump(neat.pop,fp)

  return data

def checkBest(data):
  """Checks better performing individual if it performs over many trials.
  Test a new 'best' individual with many different seeds to see if it really
  outperforms the current best.

  Args:
    data - (DataGatherer) - collected run data

  Return:
    data - (DataGatherer) - collected run data with best individual updated


  * This is a bit hacky, but is only for data gathering, and not optimization
  """
  global filename, hyp
  if data.newBest is True:
    bestReps = hyp['bestReps']
    rep = np.tile(data.best[-1], bestReps)
    fitVector = batchMpiEval(rep, gen=None, sp_count=None, sameSeedForEachIndividual=False)
    trueFit = np.mean(fitVector)
    if trueFit > data.best[-2].fitness:  # Actually better!
      data.best[-1].fitness = trueFit
      data.fit_top[-1]      = trueFit
      data.bestFitVec = fitVector
    else:                                # Just lucky!
      prev = hyp['save_mod']
      data.best[-prev:]    = data.best[-prev]
      data.fit_top[-prev:] = data.fit_top[-prev]
      data.newBest = False
  return data

def wrap_calc_fit(args):
    return calc_fit(*args)

def calc_fit(wVec, aVec, gen, sp_count, run_type, hyp):

    task = GymTask(games[hyp['task']], nReps=hyp['alg_nReps'])
    result = task.getFitness(wVec, aVec, gen, sp_count, run_type) # process it
    return result

# -- Parallelization ----------------------------------------------------- -- #
def batchMpiEval(pop, gen=None, sp_count=None, sameSeedForEachIndividual=True):
  """Sends population to workers for evaluation one batch at a time.

  Args:
    pop - [Ind] - list of individuals
      .wMat - (np_array) - weight matrix of network
              [N X N]
      .aVec - (np_array) - activation function of each node
              [N X 1]

  Return:
    reward  - (np_array) - fitness value of each individual
              [N X 1]

  Todo:
    * Asynchronous evaluation instead of batches
  """

  global nWorker, hyp, n_processor

  if gen == None:
      gen = 1000
      sp_count = 1000
      print('ALERT gen is None')

  args = []
  for ind in pop:
      wVec   = ind.wMat.flatten()
      aVec   = ind.aVec.flatten()
      args.append([wVec, aVec, gen, sp_count, run_type, hyp])

  p = Pool(processes=n_processor)
  result = p.map(wrap_calc_fit, args)
  result = np.array(result)
  return result
  # global nWorker, hyp
  # nSlave = nWorker-1
  # nJobs = len(pop)
  # nBatch= math.ceil(nJobs/nSlave) # First worker is master
  #
  # # Set same seed for each individual
  # if sameSeedForEachIndividual is False:
  #   seed = np.random.randint(1000, size=nJobs)
  # else:
  #   seed = np.random.randint(1000)
  #
  # reward = np.empty(nJobs, dtype=np.float64)
  # i = 0 # Index of fitness we are filling
  # for iBatch in range(nBatch): # Send one batch of individuals
  #   for iWork in range(nSlave): # (one to each worker if there)
  #     if i < nJobs:
  #       wVec   = pop[i].wMat.flatten()
  #       n_wVec = np.shape(wVec)[0]
  #       aVec   = pop[i].aVec.flatten()
  #       n_aVec = np.shape(aVec)[0]
  #
  #       comm.send(n_wVec, dest=(iWork)+1, tag=1)
  #       comm.Send(  wVec, dest=(iWork)+1, tag=2)
  #       comm.send(n_aVec, dest=(iWork)+1, tag=3)
  #       comm.Send(  aVec, dest=(iWork)+1, tag=4)
  #       if sameSeedForEachIndividual is False:
  #         comm.send(seed.item(i), dest=(iWork)+1, tag=5)
  #       else:
  #         comm.send(  seed, dest=(iWork)+1, tag=5)
  #       if gen is not None:
  #           comm.send(gen, dest=(iWork)+1, tag=6)
  #       else:
  #           comm.send(1000, dest=(iWork)+1, tag=6)
  #       if sp_count is not None:
  #           comm.send(sp_count, dest=(iWork)+1, tag=7)
  #       else:
  #           comm.send(1000, dest=(iWork)+1, tag=7)
  #     else: # message size of 0 is signal to shutdown workers
  #       n_wVec = 0
  #       comm.send(n_wVec,  dest=(iWork)+1)
  #     i = i+1
  #
  #   # Get fitness values back for that batch
  #   i -= nSlave
  #   for iWork in range(1,nSlave+1):
  #     if i < nJobs:
  #       workResult = np.empty(1, dtype='d')
  #       comm.Recv(workResult, source=iWork)
  #       reward[i] = workResult
  #     i+=1
  #
  # return reward

def slave():
  """Evaluation process: evaluates networks sent from master process.

  PseudoArgs (recieved from master):
    wVec   - (np_array) - weight matrix as a flattened vector
             [1 X N**2]
    n_wVec - (int)      - length of weight vector (N**2)
    aVec   - (np_array) - activation function of each node
             [1 X N]    - stored as ints, see applyAct in ann.py
    n_aVec - (int)      - length of activation vector (N)
    seed   - (int)      - random seed (for consistency across workers)

  PseudoReturn (sent to master):
    result - (float)    - fitness value of network
  """
  global hyp, gen, run_type
  task = GymTask(games[hyp['task']], nReps=hyp['alg_nReps'])

  # Evaluate any weight vectors sent this way
  while True:
    n_wVec = comm.recv(source=0,  tag=1)# how long is the array that's coming?
    if n_wVec > 0:
      wVec = np.empty(n_wVec, dtype='d')# allocate space to receive weights
      comm.Recv(wVec, source=0,  tag=2) # recieve weights

      n_aVec = comm.recv(source=0,tag=3)# how long is the array that's coming?
      aVec = np.empty(n_aVec, dtype='d')# allocate space to receive activation
      comm.Recv(aVec, source=0,  tag=4) # recieve it
      seed = comm.recv(source=0, tag=5) # random seed as int
      gen = comm.recv(source=0, tag=6) # random seed as int
      sp_count = comm.recv(source=0, tag=7) # random seed as int

      result = task.getFitness(wVec, aVec, gen, sp_count, run_type) # process it
      comm.Send(result, dest=0)            # send it back

    if n_wVec < 0: # End signal recieved
      print('Worker # ', rank, ' shutting down.')
      break

def stopAllWorkers():
  """Sends signal to all workers to shutdown.
  """
  global nWorker
  nSlave = nWorker-1
  print('stopping workers')
  for iWork in range(nSlave):
    comm.send(-1, dest=(iWork)+1, tag=1)

def mpi_fork(n):
  """Re-launches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/garymcintire/mpi_util/)
  """
  if n<=1:
    return "child"
  if os.getenv("IN_MPI") is None:
    env = os.environ.copy()
    env.update(
      MKL_NUM_THREADS="1",
      OMP_NUM_THREADS="1",
      IN_MPI="1"
    )
    print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
    subprocess.check_call(["mpirun", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
    return "parent"
  else:
    global nWorker, rank
    nWorker = comm.Get_size()
    rank = comm.Get_rank()
    #print('assigning the rank and nworkers', nWorker, rank)
    return "child"


# -- Input Parsing ------------------------------------------------------- -- #

def main(argv):
  """Handles command line input, launches optimization or evaluation script
  depending on MPI rank.
  """
  global fileName, hyp, run_type, rep_type, n_processor # Used by both master and slave processes
  fileName    = args.outPrefix
  hyp_default = args.default
  hyp_adjust  = args.hyperparam
  n_processor = args.num_worker
  run_type    = args.type1
  rep_type    = args.type2

  hyp = loadHyp(pFileName=hyp_default)
  updateHyp(hyp,hyp_adjust)
  print(hyp)

  # # Launch main thread and workers
  # if (rank == 0):
  #   master()
  # else:
  #   slave()
  # Launch main thread and workers
  master()

if __name__ == "__main__":
  ''' Parse input and launch '''
  parser = argparse.ArgumentParser(description=('Evolve NEAT networks'))

  parser.add_argument('-d', '--default', type=str,\
   help='default hyperparameter file', default='p/default_neat.json')

  parser.add_argument('-p', '--hyperparam', type=str,\
   help='hyperparameter file', default=None)

  parser.add_argument('-o', '--outPrefix', type=str,\
   help='file name for result output', default='test')

  parser.add_argument('-n', '--num_worker', type=int,\
   help='number of cores to use', default=8)

  parser.add_argument('-t1', '--type1', type=str,\
   help='which type to use static or dynamic', default='static')

  parser.add_argument('-t2', '--type2', type=str,\
   help='which type to use normal or hybrid', default='normal')

  args = parser.parse_args()


  # Use MPI if parallel
  # if "parent" == mpi_fork(args.num_worker+1): os._exit(0)

  main(args)
