/*
Author: J. Nicolas Garavito-Camargo
08/19/2016
University of Arizona.

Code to compute the covariance matrix of the
coefficients of the basis expansions functions.
Bases on Weinberg 1996 Algorithm.

Requirements:
-------------
gnu/gsl
openmp

Usage:
-------
./a.out nmax lmax #ofparticles snapshot outputfile

To-Do:
------
1. Put comments and organize code!
2. Make # of particles not an argument.

*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_gegenbauer.h>
#include <gsl/gsl_sf_legendre.h>

#include "covariance_matrix.h"



double Anl_tilde(int n ,int l){
    /*
    Function that computes the A_nl from Eq.16 in Lowing+16 paper

    Parameters:
    ----------
    n, l

    return:
    ------- 

    A_nl
    */
    
    double K_nl, factor, A_nl;
    double gamma_factor;

    // Definition of K_nl
    K_nl = 0.5*n*(n+4.*l+3.) + (l+1.)*(2.*l+1.);

    factor = pow(2,8.*l+6.) / (4.*M_PI*K_nl);
    gamma_factor = pow(gsl_sf_gamma(2.0*l+1.5),2) / gsl_sf_gamma(n+4.*l+3.);
    A_nl =-factor * gsl_sf_fact(n) * (n+2.*l+1.5) * gamma_factor;

    printf("%f", A_nl);
    return A_nl;
}


int main(){
 Anl_tilde(0, 3);    
}




