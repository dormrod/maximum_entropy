## Network Analysis of 2D Ring Systems

This repository contains python scripts to calculate maximum entropy
node degree distributions and edge joint degree distributions.

### Requirements
* Python 3.6 or higher
* Numpy 
* Scipy
* Matplotlib

### Node degree distribution (Lemaitre's Law)
Calculates maximum entropy node degree distributions
with a specified mean across a range of variances.

This is achieved through the ```NodeME``` class found in ```node_me.py```.
Examples can be found in the examples directory, but simply running:
```python
python node_me.py
```
will produce a graph of Lemaitre's law and save maximum entropy 
distributions and associated variances to ```node_dist.dat```.



