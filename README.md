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



BFE-py is a python package specialized in analyzing snapshots from idealized N-body simulations using analytic Basis Function Expansions (BFE).

# Features: 
  - Compute BFE expansion from a collection of snapshots
  - Separates a satellite from its host by finding bound
    satellite particles.
  - Recenter Host and Satellite to its COM
  - Sample satellite particles to have the same mass of the host.
  - Run in parallel for the nlm list.
  - Write particle data in Gadget format if desired.
  
# Code structure:
  - io
    - handle I/O libraries for different simulations formats using [pygadgetreader](https://bitbucket.org/rthompson/pygadgetreader/src/default/)
  - satellites
    - Finds bound particles of a Satellite
  - coefficients
    - compute coefficients in parallel using [gala](https://github.com/adrn/gala) and schwimmbad
  - analysis
    - energy of coefficients
    - plotting routines
  - noise
    - routines regaring noise
    
  
# TODO:
  - Parameter file:
      - Make different parameter categories for Host and Satellites.
      - Generalize to multiple satellites.
      - Read COM if provided
      - Remove gala dependency 

  - Optional outputs:
      - track bound mass fraction
      - write output on Gadget format file
   
   - Implement tests
   - Parallelization:
        - Try parallel structure discussed in articles.adsabs.harvard.edu/pdf/1995ApJ...446..717H
        - Fix results=pool() returns empty list if --ncores==1? when running in a single processor
- known issues:
    - currently multiprocessing return the following error when many
    particles are used:
    struct.error: 'i' format requires -2147483648 <= number <= 2147483647
    This is a known issue of multiprocessing that apparently is solved in
    python3.8
    see :
    https://stackoverflow.com/questions/47776486/python-struct-error-i-format-requires-2147483648-number-2147483647


# dependencies:

  - pyyaml (parameter file format)
  - scipy
  - numpy
  - gala
  - schwimmbad (python parallelization)
  - openmp 
  - pynbody 
  - pygadgetreader (to read old gadget formats 1/2/3)


# Installation:

```pyhton -m pip install .```
