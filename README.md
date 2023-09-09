# torchlambertw: Lambert W function and Lambert W x F distributions in pytorch


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
![Github All Releases](https://img.shields.io/github/downloads/gmgeorg/pypsps/total.svg)


**IMPORTANT**: This is *NOT* ready for use.  This is just a prototype serving as basis of discussion in https://github.com/pytorch/pytorch/issues/108948. 

This has not been properly tested w/ all pytorch functionality and should not be used other than for prototyping/R&D.

---


```python
import torchlambertw as tw
import numpy as np
special.lambertw(torch.tensor([-1., 0., 1., -np.exp(-1)]))
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


## Implementation

This implementation closely follows the TensorFlow Probability version in [`tfp.special.lambertw`](https://www.tensorflow.org/probability/api_docs/python/tfp/math/lambertw).


See also [here](https://github.com/thibsej/unbalanced-ot-functionals/blob/13f2203b3993d973f929578085ea458c5c1a7a78/common/torch_lambertw.py) and [here](
https://github.com/AminJun/BreakingCertifiableDefenses/blob/cc469fa48f7efba21f3584e233c4db0c9a4856c1/RandomizedSmoothing/projected_sinkhorn/lambertw.py
)) for minimum example `pytorch` implementations.




# Installation

It can be installed directly from GitHub using:

```python
pip install git+https://github.com/gmgeorg/torchlambertw.git
```


## References

* Corless, R.M., et al. (1996) On the LambertW Function. Advances in Computational Mathematics, 5, 329-359.
https://doi.org/10.1007/BF02124750 

* Lambert W implementation in TensorFlow: https://www.tensorflow.org/probability/api_docs/python/tfp/math/lambertw

## License

This project is licensed under the terms of the [MIT license](LICENSE).
