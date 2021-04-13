# hyb-speciation

## Note
These files are reused from [prettyNEAT](https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease/prettyNEAT). The code can compare normal NEAT and Hybridization model. 

To run normal NEAT, 
```
python neat_train.py -p/swingup.json -n 3 -o log/history_normal -t1 dynamic -t2 normal
```

To run hybridization model, 
```
python neat_train.py -p/swingup.json -n 3 -o log/history_hyb -t1 dynamic -t2 hybrid
```
