## Maximum Entropy Distributions

This repository contains python scripts to calculate maximum entropy
node degree distributions and edge joint degree distributions.

### Requirements
* Python 3.6 or higher
* Numpy 
* Scipy
* Matplotlib

### Node degree distribution (Lemaitre's Law)
Calculates maximum entropy node degree distributions
with a specified mean and principle value (e.g. p<sub>6</sub>).

This is achieved through the ```NodeME``` class found in ```node_me.py```.
A model is set up with a given mean and node size limits:
```python
from node_me import NodeME

model = NodeME(k_mean=6.0, k_limits=(3,12))
```
The maximum entropy distribution can then be obtained given the value 
for a single node size (suggested to be the most common) via:
```python
distribution = model(0.5,k=6)
```
Alternatively distributions can be calculated across a range of 
variances using:
```python
model.scan()
model.write()
```
which will write the distributions and variances to ```node_dist.dat```.
These will not necessarily be evenly spaced, and so it may be preferable
to use the call function in a loop.

### Edge joint degree distribution
To come...



