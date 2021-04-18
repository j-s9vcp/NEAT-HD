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

`-n`:  the number of workers
`-o`:  output directory
`-t1`:  static or dynamic
`-t2`:  normal or hybridization

## Visualization
Ckeck out `hyb_model_demo.ipynb` to see the figures in the context.


## Note 
the code for hybridization model is only implemented in swingup task.
