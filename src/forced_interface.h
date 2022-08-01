#include <defines.h>

// Abstract base class for a functor to compute the interface velocity of a
// forced interface.
class ComputeForcedInterfaceVelocity {
    public:
        virtual vector<double> operator() (double x, double y, double t,
                vector<double>& data) = 0;
};

// Compute interface velocity for an interface advected at constant velocity.
class ComputeAdvectionInterfaceVelocity : public ComputeForcedInterfaceVelocity {
    public:
        vector<double> operator() (double x, double y, double t,
                vector<double>& data);
};

// Compute interface velocity for a collapsing cylinder.
class ComputeCollapsingCylinderVelocity : public ComputeForcedInterfaceVelocity {
    public:
        vector<double> operator() (double x, double y, double t,
                vector<double>& data);
};
