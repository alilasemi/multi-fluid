cd cache/
# TODO: Unhack this
python='/home/alilasemi/software/anaconda3/bin/python'
#!/bin/bash
flags=-g
g++ $flags -Wall -shared -std=c++11 -fPIC $($python -m pybind11 --includes) compute_A_RL.cpp -o compute_A_RL.so
g++ $flags -Wall -shared -std=c++11 -fPIC $($python -m pybind11 --includes) compute_Lambda.cpp -o compute_Lambda.so
g++ $flags -Wall -shared -std=c++11 -fPIC $($python -m pybind11 --includes) compute_Q_inv.cpp -o compute_Q_inv.so
g++ $flags -Wall -shared -std=c++11 -fPIC $($python -m pybind11 --includes) compute_Q.cpp -o compute_Q.so
cd ../
