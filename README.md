> **Note: `skaggs` has not yet been refactored as an independent package. You should have no expectation that it will run or import correctly in its current state.**
>
> **I hope to ameliorate this as soon as I have time; of course, PRs welcome.**

<p>&nbsp;</p>

# skaggs

The `skaggs` package supports the import, storage, data structures, preprocessing, and analysis of neurobehavioral datasets of the type produced by rodent spatial navigation (e.g., place cell) experiments, for cluster-sorted single-unit data and 2D head-position trajectories. Data analysis functionality includes both signal processing and information theoretic calculations.

## Origin

This code was used to conduct the data analysis of hippocampal and subcortical recording data presented in this paper:

* Monaco JD, De Guzman RM, Blair HT, and Zhang K. (2019). [Spatial synchronization codes from coupled rate-phase neurons](https://dx.doi.org/10.1371/journal.pcbi.1006741). *PLOS Computational Biology*, **15**(1), e1006741. doi:&nbsp;[10.1371/journal.pcbi.1006741](https://dx.doi.org/10.1371/journal.pcbi.1006741)

The complete code archive for the paper is available on figshare (doi:&nbsp;[10.6084/m9.figshare.6072317.v1](https://doi.org/10.6084/m9.figshare.6072317.v1)) and the dataset is archived on OSF (doi:&nbsp;[10.17605/osf.io/psbcw](https://doi.org/10.17605/osf.io/psbcw)). The `skaggs` package is based on the `spc.lib` subpackage in that code archive; the name is an homage, of course, to [Bill Skaggs](https://neurotree.org/beta/publications.php?pid=11486). 

## Dependencies

*Note: This section will be updated as the packaging and dependencies are fixed.*

The `skaggs` package requires a typical scientific python computing environment, which can be set up using Anaconda or similar distributions (see the `requirements.txt`). 

## Todo

- [ ] Fix dependencies for other packages of mine (e.g., remove or add as submodules)
- [ ] Update the `setup.py` to ensure correct installation, etc.
- [ ] Improve function and class APIs to enhance usabililty and convenience
- [ ] Code style and formatting consistency (e.g., `flake8` validation)
