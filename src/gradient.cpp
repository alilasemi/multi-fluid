#include <math.h>
#include <iostream>
using std::cout, std::endl;

#include <defines.h>
#include <gradient.h>

void compute_gradient(matrix_ref<double> U, matrix_ref<double> xy,
        std::vector<vector<long>>& stencil, vector_ref<double> gradU) {
    // Number of cells
    int n = U.rows();
    // Loop over all cells
    for (int i = 0; i < n; i++) {
        // Get this cell's gradU
        matrix_map<double> gradU_i(&gradU(i*4*2), 4, 2);

        auto n_points = stencil[i].rows();
        // If there are no other points in the stencil, then set the gradient to
        // zero
        if (n_points == 1) {
            gradU_i = matrix<double>::Zero(4, 2);
        // Otherwise, solve with least squares
        } else {
            // Construct A matrix: [x_i, y_i, 1]
            matrix<double> A = matrix<double>::Constant(n_points, 3, 1);
            A(all, seq(0, 1)) = xy(stencil[i], all);
            // We desired [x_i, y_i, 1] @ [c0, c1, c2] = U[i], therefore Ax=b.
            // However, there are more equations than unknowns (for most points)
            // so instead, solve the normal equations: A.T @ A x = A.T @ b
            matrix<double> c = (A.transpose() * A).partialPivLu().solve(
                    A.transpose() * U(stencil[i], all));
            // Since U = c0 x + c1 y + c2, then dU/dx = c0 and dU/dy = c1.
            gradU_i = c(seq(0, 1), all).transpose();
            // If any NaNs are found, that means the matrix inverse failed
            // (probably a singular combination of stencil points, such as all
            // points being in a straight line). In this case, set gradient to
            // zero.
            if (c.array().isNaN().any()) {
                cout << "Gradient calculation failed! Stencil = "
                        << stencil[i].transpose() << endl;
                gradU_i = matrix<double>::Zero(4, 2);
            }
        }
    }
}

void compute_gradient_phi(matrix_ref<double> phi, matrix_ref<double> xy,
        std::vector<vector<long>>& stencil, matrix_ref<double> grad_phi) {
    // Number of cells
    int n = phi.rows();
    // Loop over all cells
    for (int i = 0; i < n; i++) {
        auto n_points = stencil[i].rows();
        // If there are no other points in the stencil, then set the gradient to
        // zero
        if (n_points == 1) {
            grad_phi(i, all) = vector<double>::Zero(2);
        // Otherwise, solve with least squares
        } else {
            // Construct A matrix: [x_i, y_i, 1]
            matrix<double> A = matrix<double>::Constant(n_points, 3, 1);
            A(all, seq(0, 1)) = xy(stencil[i], all);
            // We desired [x_i, y_i, 1] @ [c0, c1, c2] = U[i], therefore Ax=b.
            // However, there are more equations than unknowns (for most points)
            // so instead, solve the normal equations: A.T @ A x = A.T @ b
            matrix<double> c = (A.transpose() * A).partialPivLu().solve(
                    A.transpose() * phi(stencil[i], all));
            // Since phi = c0 x + c1 y + c2, then dphi/dx = c0 and dphi/dy = c1.
            grad_phi(i, all) = c(seq(0, 1), 0);
            // If any NaNs are found, that means the matrix inverse failed
            // (probably a singular combination of stencil points, such as all
            // points being in a straight line). In this case, set gradient to
            // zero.
            if (c.array().isNaN().any()) {
                cout << "phi gradient calculation failed! Stencil = "
                        << stencil[i].transpose() << endl;
                grad_phi(i, all) = vector<double>::Zero(2);
            }
        }
    }
}
