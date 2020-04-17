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
  
# TODO:
  -Parameter file:
      - Make different categories for Host and Satellite?
      - Think that if this is going to be general we might need more than
      one satellite.
  -Implement all optional outputs:
      - random satellite sample *
      - output ascii files
      - what if the COM is provided?
      - use ids to track bound - unbound particles -- think about cosmo
      zooms
      - track bound mass fraction
      - write Gadget format file
      - Check if all the flags are working
      - Flag: write_snaps_ascii
        - ids_tr is needed? or can be deleted to use less memory?
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
