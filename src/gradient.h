#include <string>

#include <defines.h>

// Compute the gradient of the state with respect to physical space.
void compute_gradient(matrix_ref<double> U, matrix_ref<double> xy,
        std::vector<vector<long>>& stencil, vector_ref<double> gradV,
        std::vector<double> g, std::vector<double> psg,
        vector_ref<long> fluid_ID);

// Compute the gradient of the level set with respect to physical space.
void compute_gradient_phi(matrix_ref<double> phi, matrix_ref<double> xy,
        std::vector<vector<long>>& stencil, matrix_ref<double> grad_phi);
