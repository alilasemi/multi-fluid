#include <string>

#include <defines.h>

// Compute the gradient of the state with respect to physical space.
void compute_gradient(matrix_ref<double> U, matrix_ref<double> xy,
        std::vector<vector<long>>& stencil, vector_ref<double> gradU);

// Compute the gradient of the level set with respect to physical space.
void compute_gradient_phi(matrix_ref<double> phi, matrix_ref<double> xy,
        std::vector<vector<long>>& stencil, matrix_ref<double> grad_phi);
