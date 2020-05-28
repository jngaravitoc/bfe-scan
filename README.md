# bfe-scan
Pipeline to analyze N-body simulations using Basis Field Expansions (BFE)

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

  - pyyaml (parameter file )
  - scipy
  - numpy
  - gala
  - schwimmbad
  - openmp
  - pygadgetreader


# Installation:

pyhton setup.py install
