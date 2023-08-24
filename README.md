# Basis Functions Expansions in Python (BFE-py) 

             __                __              __                       __      __       __    
            / /\              /\ \            /\ \                     /\ \    /\ \     /\_\   
           / /  \            /  \ \          /  \ \                   /  \ \   \ \ \   / / /   
          / / /\ \          / /\ \ \        / /\ \ \                 / /\ \ \   \ \ \_/ / /    
         / / /\ \ \        / / /\ \_\      / / /\ \_\   ____        / / /\ \_\   \ \___/ /     
        / / /\ \_\ \      / /_/_ \/_/     / /_/_ \/_/ /\____/\     / / /_/ / /    \ \ \_/      
       / / /\ \ \___\    / /____/\       / /____/\    \/____\/    / / /__\/ /      \ \ \   
      / / /  \ \ \__/   / /\____\/      / /\____\/               / / /_____/        \ \ \    
     / / /____\_\ \    / / /           / / /______              / / /                \ \ \     
    / / /__________\  / / /           / / /_______\            / / /                  \ \_\    
    \/_____________/  \/_/            \/__________/            \/_/                    \/_/  



BFE-py is a Python package specialized in utilizing basis function expansions (BFE) from N-body simulations. The package is designed to work and interact with specialized codes such as EXP, and to work with various BFE outputs from codes such as AGAMA, Gala, Galpy, and EXP. 

# Some of the BFE-py features include: 
  - Read and write coefficients from Gala, AGAMA, and EXP.
  - Smoothing coefficients following Weinverg+98
  - Compute BFE expansion from a collection of N-body simulation snapshots using the SCF expansion
  - Computed bound masses of satellite galaxies.
  - Compute energy of the coefficients
  - Visualization of coefficients and fields

  
# Code structure:
  - io
    - handle I/O libraries for different simulation formats using [pygadgetreader](https://bitbucket.org/rthompson/pygadgetreader/src/default/)
  - satellites
    - Finds bound particles of a Satellite
  - coefficients
    - Compute coefficients in parallel using [gala](https://github.com/adrn/gala) and schwimmbad
  - analysis
    - Energy of coefficients
    - plotting routines
  - exp
    - a variety of functions to interface with EXP
  - noise
    - routines regarding noise
    


 
   - Parallelization:
        - Implement parallel structure discussed in articles.adsabs.harvard.edu/pdf/1995ApJ...446..717H
        - Fix results=pool() returns empty list if --ncores==1 When running in a single processor
- Known issues:
    - For Python versions < 3.8 multiprocessing returns the following error when many
    particles are used:
    struct.error: 'i' format requires -2147483648 <= number <= 2147483647
    This is a known issue of multiprocessing has been solved in
    python3.8
    see :
    https://stackoverflow.com/questions/47776486/python-struct-error-i-format-requires-2147483648-number-2147483647


# dependencies:

  - python3.8 or up
  - pyyaml (parameter file format)
  - scipy
  - numpy
  - gala
  - schwimmbad (python parallelization)
  - openmp
  - pyEXP (optional)


# Installation:

```pyhton -m pip install .```
