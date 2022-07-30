#include <string>

#include <defines.h>

// Compute the interior faces' contributions to the residual.
void compute_interior_face_residual(matrix_ref<double> U,
        matrix_ref<long> edge, matrix_ref<double> quad_wts,
        std::vector<double> quad_pts_phys, matrix_ref<double> limiter,
        std::vector<double> gradU, matrix_ref<double> xy,
        std::vector<double> area_normals_p2, matrix_ref<double> area,
        double g, matrix_ref<double> residual);

// Compute the boundary faces' contributions to the residual.
void compute_boundary_face_residual(matrix_ref<double> U,
        matrix_ref<long> bc_type, matrix_ref<double> quad_wts,
        std::vector<double> quad_pts_phys, matrix_ref<double> limiter,
        std::vector<double> gradU, matrix_ref<double> xy,
        std::vector<double> area_normals_p2, matrix_ref<double> area,
        double g, long num_boundaries, matrix<double> bc_data,
        std::string problem_name, double t, matrix_ref<double> residual);
