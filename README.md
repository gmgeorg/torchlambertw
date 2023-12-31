# torchlambertw: Lambert W function and Lambert W x F distributions in pytorch


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
![Github All Releases](https://img.shields.io/github/downloads/gmgeorg/torchlambertw/total.svg)


**IMPORTANT**: This is the very first prototype for an implementation of the Lambert W function and Lambert W x F distributions in `torch`. For now this is a prototype serving as reference for discussion in https://github.com/pytorch/pytorch/issues/108948.  Use this only for prototyping/R&D (see also `LICENSE`).

See https://github.com/gmgeorg/torchlambertw/issues for remaining issues/TODOs.

---

# Overview

This library is a native implementation in `pytorch` of

 * the Lambert W function (`special.lambertw`)

 * Lambert W x F distributions (`torch.distributions`)

While this library is for now standalone, the goal is to get both the mathematical function as well as the distributions into `torch` core package.

See also https://github.com/pytorch/pytorch/issues/108948.

**IMPORTANT**: See also the accompanying [**pylambertw**](https://github.com/gmgeorg/pylambertw) 
module which uses `torchlambertw` under the hood to train distribution parameters and
can be used to Gaussianize skewed, heavy-tailed data.

The `torchlambertw` module here is solely focused on providing the building blocks
for Lambert W functions and Lambert W x F distributions.  If you are interested
in using Transformations and estimating parameters of these distributions, take a look
at the **pylambertw** instead.

## Installation

It can be installed directly from GitHub using:

```python
pip install git+https://github.com/gmgeorg/torchlambertw.git
```


## Lambert W function (math)

Implementation of the Lambert W function (special function) in `torch`:

```python
import torchlambertw as tw
import numpy as np
tw.special.lambertw(torch.tensor([-1., 0., 1., -np.exp(-1)]))
```
output:
```bash
tensor([nan,  0.0000,  0.5671, -1.0000], dtype=torch.float64)
```

As a more interesting example you can use this implementation to replicate the figure on the [Lambert W Function](https://en.wikipedia.org/wiki/Lambert_W_function) Wikipedia page:

```python
import numpy as np
import matplotlib.pyplot as plt
from torchlambertw import special

def plot_lambertW(range_start, range_end, num_points=2000):
    x_values = np.linspace(range_start, range_end, num_points)
    x_values_torch = torch.tensor(x_values)
    principal_branch_values = special.lambertw(x_values_torch, k=0).numpy()
    non_principal_branch_values = special.lambertw(x_values_torch, k=-1).numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, principal_branch_values, label="Principal Branch", color='blue')
    plt.plot(x_values, non_principal_branch_values, label="Non-Principal Branch", color='red')

    plt.title("Lambert W Function")
    plt.xlabel("x")
    plt.ylabel("W(x)")
    plt.xlim(range_start, range_end)
    plt.ylim(-4, 2)  # same range as wiki figure
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
    plt.legend()

    plt.grid(True)
    plt.show()

# Example usage:
plot_lambertW(-1, 6)

```
![Lambert W Function](imgs/lambertw_plot.png)

## Lambert W x F distributions

For the original papers see Goerg 2011 & 2015. If you want to jump into applications and examples I suggest looking at the [**LambertW** R package](https://github.com/gmgeorg/LambertW) for detailed references and links to many external examples on Stackoverflow / cross-validated and other external blogs.


**Important**: The `torch.distributions` framework allows you to easily build *any* Lambert W x F
distribution by just using the skewed & heavy tail Lambert W transform here implemented here and pass whatever `base_distribution` -- that's F -- makes sense to you. Voila! You have just built a Lambert W x F distribution.

See [demo notebook](notebooks/demo-lambertw-f-distributions.ipynb) for details.


### In a nutshell

Lambert W x F distributions are a generalized family of distributions, that take an "input" X ~ F and transform it to a skewed and/or heavy-tailed output, Y ~ Lambert W x F, via a particularly parameterized transformation.  See Goerg (2011, 2015) for details.

![Lambert W Function](imgs/input_output_system_tail.png)


For parameter values of 0, the new variable collapses to X, which means that Lambert W x F distributions always contain the original base distribution F as a special case.  Ie it does not hurt to impose a Lambert W x F distribution on your data; worst case, parameter estimates are 0 and you get F back; best case: you properly account for skewness & heavy-tails in your data and can even remove it (by transforming data back to having X ~ F). The such obtained random variable / data / distribution is then a Lambert W x F distribution.

The convenient part about this is that when working with data y1, ..., yn, you can estimate the transformation from the data and transform it back into the (unobserved) x1, ..., xn.  This is particularly useful when X ~ Normal(loc, scale), as then you can "Gaussianize" your data.

### Heavy-tail Lambert W x F distributions

Here is an illustration of a heavy-tail Lambert W x Gaussian distribution, which takes a Gaussian input and turns it into something heavy-tailed. If `tailweight = 0` then its just a Gaussian again.

```python
from torchlambertw import distributions as tlwd

# Implements a Lambert W x Normal distribution with (loc=1, scale=3, tailweight=0.75)
m = tlwd.TailLambertWNormal(loc=1.0, scale=3.0, tailweight=0.75)
m.sample((2,))
```
```
tensor([[ 0.0159], [-0.9322]])
```

This distribution is quite heavy-tailed with moments existing only up to `1 / tailweight = 1.33`, ie this random variable / distribution has infinite (population) variance.

```python
m.tailweight, m.support, m.mean, m.variance
```
```
(tensor([0.7500]), Real(), tensor([1.]), tensor([inf]))
```

Let's draw a random sample from distribution and plot density / ecdfplot.

```python
torch.manual_seed(0)
# Use a less heavy-tailed distribution with a tail parameter of 0.25 (ie moments < 1/0.25 = 4 exist).
m = tlwd.TailLambertWNormal(loc=1.0, scale=3.0, tailweight=0.25)
y = m.sample((1000,)).numpy().ravel()

import seaborn as sns
import statsmodels.api as sm

sns.displot(y, kde=True)
plt.show()
sm.qqplot(y, line='45', fit=True)
plt.grid()
plt.show()
```
![Lambert W x Gaussian histogram and KDE](imgs/lambertw_gauss_hist_kde.png)

![Lambert W x Gaussian qqnorm plot](imgs/lambertw_gauss_qqnorm.png)

#### Back-transformation

The parameters `(loc, scale, tailweight)` can be estimated from the data using the accompanying [**pylambertw**](https://github.com/gmgeorg/pylambertw) module (see also [**LambertW** R package](https://github.com/gmgeorg/LambertW)).

Let's say you have the estimated parameters; then you can obtain the unobserved, Gaussian data using:

```python
torch.manual_seed(0)

m = tlwd.LambertWNormal(loc=1.0, scale=3.0, tailweight=0.25)

y = m.sample((1000,)).numpy().ravel()
x = m.transforms[0]._inverse(torch.tensor(y)).numpy().ravel()
sns.displot(x, kde=True)
plt.show()
sm.qqplot(x, line='45', fit=True)
plt.grid()
plt.show()

```

![Lambert W x Gaussian histogram and KDE](imgs/lambertw_gauss_latent_hist_kde.png)

![Lambert W x Gaussian qqnorm plot](imgs/lambertw_gauss_latent_qqplot.png)

### Skewed Lambert W x F distributions

For examples of skewed Lambert W x F distributions, for F = Normal, Exponential, or Gamma
see [demo notebook](notebooks/demo-lambertw-f-distributions.ipynb).

## Implementation details

This implementation closely follows the TensorFlow Probability version in [`tfp.special.lambertw`](https://www.tensorflow.org/probability/api_docs/python/tfp/math/lambertw).


## Related Implementations

* TensorFlow Probability: [**LambertWDistribution**](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LambertWDistribution)

* R package: [**LambertW**](https://github.com/gmgeorg/LambertW)

* R package / C++: [**lamw**](https://github.com/aadler/lamW)


See also [here](https://github.com/thibsej/unbalanced-ot-functionals/blob/13f2203b3993d973f929578085ea458c5c1a7a78/common/torch_lambertw.py) and [here](
https://github.com/AminJun/BreakingCertifiableDefenses/blob/cc469fa48f7efba21f3584e233c4db0c9a4856c1/RandomizedSmoothing/projected_sinkhorn/lambertw.py
)) for minimum example `pytorch` implementations [not optimized for fast iteration though and good starting points.]


## References

* Goerg (2011). *Lambert W random variables—a new family of generalized skewed distributions with applications to risk estimation.* Ann. Appl. Stat. 5 (3) 2197 - 2230, 2011. https://doi.org/10.1214/11-AOAS457

* Goerg (2015) *The Lambert Way to Gaussianize Heavy-Tailed Data with the Inverse of Tukey’s h Transformation as a Special Case*. The Scientific World Journal. Volume 2015 | Article ID 909231 | https://doi.org/10.1155/2015/909231

* Goerg (2016) *Rebuttal of the 'Letter to the Editor' of Annals of Applied Statistics on Lambert W x F Distributions and the IGMM Algorithm*. https://arxiv.org/abs/1602.02200

* Corless, R.M., et al. (1996) On the LambertW Function. Advances in Computational Mathematics, 5, 329-359.
https://doi.org/10.1007/BF02124750

* Lambert W implementation in TensorFlow: https://www.tensorflow.org/probability/api_docs/python/tfp/math/lambertw

* Käärik, Meelis & Selart, Anne & Puhkim, Tuuli & Tee, Liivika. (2023). *Lambert W random variables and their applications in loss modelling*. https://arxiv.org/pdf/2307.05644.pdf

# License

This project is licensed under the terms of the [MIT license](LICENSE).
