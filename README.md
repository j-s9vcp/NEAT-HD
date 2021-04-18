# hyb-speciation

These files are reused from [prettyNEAT](https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease/prettyNEAT) by Adam Gaier et.al (2019) (https://arxiv.org/pdf/1906.04358.pdf). The code can compare normal NEAT and Hybridization model. 

To run normal NEAT, 
```
python neat_train.py -p/swingup.json -n 3 -o log/history_normal -t1 dynamic -t2 normal
```

To run hybridization model, 
```
python neat_train.py -p/swingup.json -n 3 -o log/history_hyb -t1 dynamic -t2 hybrid
```

## Visualization
to see the figures in the context, ckeck .ipynb file.

## Note 
the code for hybridization model is only implemented in swingup task.
