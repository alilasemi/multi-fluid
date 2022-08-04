#include <iostream>
#include <cmath>
using std::cout, std::endl;

#include <forced_interface.h>

// Compute interface velocity for an interface advected at constant velocity.
vector<double> ComputeAdvectionInterfaceVelocity::operator() (
        [[maybe_unused]] double x, [[maybe_unused]] double y,
        [[maybe_unused]] double t, vector<double>& data) {
    vector<double> velocity(2);
    // data(0) stores the x-component of the advection velocity
    velocity(0) = data(0);
    // data(1) stores the y-component of the advection velocity
    velocity(1) = data(1);
    return velocity;
}

// Compute interface velocity for a collapsing cylinder.
vector<double> ComputeCollapsingCylinderVelocity::operator() (double x,
        double y, double t, vector<double>& data) {
    vector<double> velocity(2);
    // Compute the angle of this point
    auto theta = atan2(y, x);
    // Radius of cylinder
    auto R = data(0);
    // Precompute factor. data(1) stores the total time to collapse and return
    // back to a circle
    auto a = M_PI / data(1);
    // TODO below not needed, but document the math
    // Compute r
    //auto r = (1/3 * (2 + cos(4 * theta)))
    //        * pow(sin(a*t), 2) + pow(cos(a*t), 2);
    // r *= R;
    // Compute dr/dt
    double drdt = 2./3*a*sin(a*t)*cos(a*t) * ( cos(4 * theta) - 1 );
    drdt *= R;
    // Compute dx/dt
    velocity(0) = drdt * cos(theta);
    // Compute dy/dt
    velocity(1) = drdt * sin(theta);
    return velocity;
}

vector<double> ComputeStarVelocity::operator() (double x,
        double y, double t, vector<double>& data) {
    vector<double> velocity(2);
    // Frequency
    auto f = data(0);
    // Compute A
    auto sign_x = copysign(1.0, x);
    auto sign_y = copysign(1.0, y);
    auto norm = sqrt(x*x + y*y);
    auto A = abs(x) + abs(y) - norm;
    // Compute dB/dt
    auto dB_dt = -2 * M_PI * f * sin(2 * M_PI * f * t);
    // Compute dx/dt
    velocity(0) = .5 * A * sign_x * dB_dt;
    // Compute dy/dt
    velocity(1) = .5 * A * sign_y * dB_dt;
    return velocity;
}
