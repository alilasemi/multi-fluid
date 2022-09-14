#include <defines.h>

// Compute the left and right star states for an exact Riemann problem.
void compute_exact_riemann_problem(double rL, double pL, double uL, double rR,
        double pR, double uR, double gL, double gR, double psgL, double psgR,
        vector_ref<double> result);
