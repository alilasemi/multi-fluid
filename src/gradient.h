#include <string>

#include <defines.h>

// Compute the gradient of the state with respect to physical space.
void compute_gradient(matrix_ref<double> U, matrix_ref<double> xy,
        std::vector<vector<long>>& stencil, vector_ref<double> gradU);
