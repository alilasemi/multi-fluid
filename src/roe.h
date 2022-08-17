#include <defines.h>

void compute_flux(matrix_ref<double> U_L,
        matrix_ref<double> U_R, matrix<double>& area_normal, double g,
        vector<double>& F);
