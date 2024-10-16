#include <string>

#include <defines.h>

// Compute the interior faces' contributions to the residual.
void compute_interior_face_residual(matrix_ref<double> U,
        matrix_ref<double> U_L, matrix_ref<double> U_R,
        vector_ref<long> interior_face_IDs, matrix_ref<long> edge,
        matrix_ref<double> limiter, std::vector<double> gradV,
        matrix_ref<double> xy, matrix_ref<double> area_normals_p1,
        matrix_ref<double> area, vector_ref<long> fluid_ID,
        std::vector<double> g, std::vector<double> psg,
        matrix_ref<double> residual);

// Compute the fluid-fluid interfaces' contributions to the residual.
void compute_fluid_fluid_face_residual(matrix_ref<double> U,
        vector_ref<double> U_L_p2, vector_ref<double> U_R_p2,
        vector_ref<long> interface_IDs, matrix_ref<long> edge,
        matrix_ref<double> quad_wts, std::vector<double> quad_pts_phys,
        matrix_ref<double> limiter, std::vector<double> gradV,
        matrix_ref<double> xy, std::vector<double> area_normals_p2,
        matrix_ref<double> area, vector_ref<long> fluid_ID,
        std::vector<double> g, std::vector<double> psg,
        matrix_ref<double> residual, vector_ref<double> U_L_Rie,
        vector_ref<double> U_R_Rie);

// Compute the boundary faces' contributions to the residual.
void compute_boundary_face_residual(matrix_ref<double> U,
        matrix_ref<long> bc_type, matrix_ref<double> quad_wts,
        std::vector<double> quad_pts_phys, matrix_ref<double> limiter,
        std::vector<double> gradV, matrix_ref<double> xy,
        std::vector<double> area_normals_p2, matrix_ref<double> area,
        vector_ref<long> fluid_ID, std::vector<double> g,
        std::vector<double> psg, long num_boundaries, matrix<double> bc_data,
        std::string problem_name, double t, matrix_ref<double> residual);

void evaluate_solution_at_interior_faces(matrix_ref<double> U,
        vector_ref<long> interior_face_IDs, matrix_ref<long> edge,
        matrix_ref<double> limiter, std::vector<double> gradV,
        matrix_ref<double> xy, vector_ref<long> fluid_ID,
        std::vector<double> g, std::vector<double> psg,
        matrix_ref<double> U_L, matrix_ref<double> U_R);

void evaluate_solution_at_interfaces(matrix_ref<double> U,
        vector_ref<long> interface_IDs, matrix_ref<long> edge,
        matrix_ref<double> quad_wts, std::vector<double> quad_pts_phys,
        matrix_ref<double> limiter, std::vector<double> gradV,
        matrix_ref<double> xy, vector_ref<long> fluid_ID,
        std::vector<double> g, std::vector<double> psg,
        vector_ref<double> U_L_p2, vector_ref<double> U_R_p2);

vector<double> conservative_to_primitive(vector<double> U, double g,
        double psg);

vector<double> primitive_to_conservative(vector<double> V, double g,
        double psg);
