import numpy as np
import copy
import itertools
from .ind import Ind
from utils import *


def evolvePop(self):
  """ Evolves new population from existing species.
  Wrapper which calls 'recombine' on every species and combines all offspring into a new population. When speciation is not used, the entire population is treated as a single species.
  """
  newPop = []
  for i in range(len(self.species)):
    children, self.innov = self.recombine(self.species[i],\
                           self.innov, self.gen)
    newPop.append(children)
  self.pop = list(itertools.chain.from_iterable(newPop))

def recombine(self, species, innov, gen):
  """ Creates next generation of child solutions from a species

  Procedure:
    ) Sort all individuals by rank
    ) Eliminate lower percentage of individuals from breeding pool
    ) Pass upper percentage of individuals to child population unchanged
    ) Select parents by tournament selection
    ) Produce new population through crossover and mutation

  Args:
      species - (Species) -
        .members    - [Ind] - parent population
        .nOffspring - (int) - number of children to produce
      innov   - (np_array)  - innovation record
                [5 X nUniqueGenes]
                [0,:] == Innovation Number
                [1,:] == Source
                [2,:] == Destination
                [3,:] == New Node?
                [4,:] == Generation evolved
      gen     - (int) - current generation

  Returns:
      children - [Ind]      - newly created population
      innov   - (np_array)  - updated innovation record

  """
  p = self.p
  nOffspring = int(species.nOffspring)
  pop = species.members
  all_pop = np.copy(self.pop).tolist()
  children = []

  if self.rep_type == 'normal':
      # Sort by rank
      pop.sort(key=lambda x: x.rank)

      # Cull  - eliminate worst individuals from breeding pool
      numberToCull = int(np.floor(p['select_cullRatio'] * len(pop)))
      if numberToCull > 0:
        pop[-numberToCull:] = []

      # Elitism - keep best individuals unchanged
      nElites = int(np.floor(len(pop)*p['select_eliteRatio']))
      for i in range(nElites):
        children.append(pop[i])
        nOffspring -= 1
        if nOffspring == 0:
            return children, innov

      all_pop_rank = [all_pop[r].rank for r in range(len(all_pop))]
      ind_pop_rank = [pop[r].rank for r in range(len(pop))]

      # Get parent pairs via tournament selection
      # -- As individuals are sorted by fitness, index comparison is
      # enough. In the case of ties the first individual wins
      parentA = np.random.randint(len(pop),size=(nOffspring,p['select_tournSize']))
      parentB = np.random.randint(len(pop),size=(nOffspring,p['select_tournSize']))
      parentA = np.min(parentA,1)
      parentB = np.min(parentB,1)
      parentA = [ind_pop_rank[i] for i in parentA]
      parentA = [all_pop_rank.index(i) for i in parentA]
      parentB = [ind_pop_rank[i] for i in parentB]
      parentB = [all_pop_rank.index(i) for i in parentB]

  elif self.rep_type == 'hybrid':
      # Sort by rank
      pop.sort(key=lambda x: x.rank)

      # Cull  - eliminate worst individuals from breeding pool
      numberToCull = int(np.floor(p['select_cullRatio'] * len(pop)))
      if numberToCull > 0:
        for i_drop in range(numberToCull):
          for j_drop in range(len(all_pop)):
            if all_pop[j_drop].rank == pop[-numberToCull:][i_drop].rank:
                del all_pop[j_drop]
                break

        pop[-numberToCull:] = []

      # Elitism - keep best individuals unchanged
      nElites = int(np.floor(len(pop)*p['select_eliteRatio']))
      for i in range(nElites):
        children.append(pop[i])
        nOffspring -= 1
        if nOffspring == 0:
            return children, innov

      all_pop_rank = [all_pop[r].rank for r in range(len(all_pop))]
      ind_pop_rank = [pop[r].rank for r in range(len(pop))]

      parentA = np.random.randint(len(pop),size=(nOffspring,p['select_tournSize']))
      parentA = np.min(parentA,1)

      parentB = []
      for i in range(len(parentA)):
          prob_list = []
          for j in range(len(all_pop)):
              cDist = self.compatDist(pop[parentA[i]].conn,all_pop[j].conn)
              cDist = self.cDist_func(cDist)
              prob_list.append(cDist)
          for t in range(len(prob_list)):
              if t == parentA[i]:
                  prob_list[t] == 0
                  continue
              prob_list[t] = 1- prob_list[t]
          summing = sum(prob_list)
          prob_list = np.array(prob_list)/summing
          prob_list = prob_list.tolist()
          choose_him = np.random.choice(list(range(len(all_pop))), p=prob_list)
          parentB.append(choose_him)
      parentB = np.array(parentB)

      parentA = [ind_pop_rank[i] for i in parentA]
      parentA = [all_pop_rank.index(i) for i in parentA]


  parents = np.vstack( (parentA, parentB ) )
  # parents = np.sort(parents,axis=0) # Higher fitness parent first

  # Breed child population
  for i in range(nOffspring):
    if np.random.rand() > p['prob_crossover']:
      # Mutation only: take only highest fit parent
      child, innov = all_pop[parents[0,i]].createChild(p,innov,gen)

      # parent history
      child.p1_rank = copy.deepcopy(all_pop[parents[0,i]].rank)
      child.p2_rank = -1

    else:
      # Crossover
      try:
          child, innov = all_pop[parents[0,i]].createChild(p,innov,gen,\
                mate=all_pop[parents[1,i]])
      except:
          print('ind_pop {}'.format(len(pop)))
          print('all_pop {}'.format(len(all_pop)))
          print('parents\n{}'.format(parents))
          print('num ', i)

      child.p1_rank = copy.deepcopy(all_pop[parents[0,i]].rank)
      child.p2_rank = copy.deepcopy(all_pop[parents[1,i]].rank)


    child.express()
    children.append(child)

  return children, innov

def cDist_func(self, x):
    p = self.p
    return 1/(1 + np.exp(-50*(-p['spec_thresh'] + x)))
