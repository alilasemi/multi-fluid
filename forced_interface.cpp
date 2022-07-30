#include <iostream>
#include <cmath>
using std::cout, std::endl;

#include <forced_interface.h>

// Compute interface velocity for an interface advected at constant velocity.
vector<double> ComputeAdvectionInterfaceVelocity::operator() (double x,
        double y, double t, vector<double>& data) {
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
    // Precompute factor
    auto a = 100 * M_PI;
    // Radius of cylinder
    auto R = data(0);
    // TODO below not needed, but document the math
    // Compute r
    //auto r = (1/3 * (2 + cos(4 * theta)))
    //        * pow(sin(a*t), 2) + pow(cos(a*t), 2);
    // r *= R;
    // Compute dr/dt
    auto drdt = (1/3 * (2 + cos(4 * theta)))
            * 2*a*sin(a*t) + 2*a*cos(a*t);
    drdt *= R;
    // Compute dx/dt
    velocity(0) = drdt * cos(theta);
    // Compute dy/dt
    velocity(1) = drdt * sin(theta);
    return velocity;
}
