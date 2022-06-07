The convolutional neural network (CNN) based wave-functions for J1-J2 model and t-J model are presented in a recent paper, and the interested reader can verify the energy results using the Python scripts in this repository.

The J1-J2 quantum spin model on the square lattice has strong frustrations. In the frustrated region, the ground state of J1-J2 model is inferred as the quantum spin liquids identified by several independent numerical methods. Because of the competitions between J1 terms and J2 terms, solving the ground states of J1-J2 model is very challenging, thus the model is a candidate model for benchmarking numerical methods. The t-J model is also a challenging model, because of long-range entanglement of the Fermions, and the high energy degeneracy near the ground state, numerically solving the t-J model requires an efficient numetical method and a huge comutational effort to achieve the real ground energy. 

The newly emerged neural network based quantum wave-functions are promising tools for numerically solving the quantum many-body systems. Benchmarking the neural network ansatz on the J1-J2 model and the t-J model can demonstrate the strong state representation ability of neural networks, thus other quantum models can be solved in a easier fashion. Therefore, the authors in this work challenged the CNN-based wave-function on both models, and obtained faithful energy results. 

To make the verification generic, the Python scripts can run on a CPU machine with the environments below:\
1, Python-3.6.15\
2, openmpi-3.1.6\
3, numpy-1.19.5, numba-0.31.1, scipy-1.5.4, torch-1.10, mpi4py-3.1.1

In the file "fire.py", the string value of variable "Case" on line 20 determines the quantum model (J1J2 or tJ). On line 55, the number passing to the function "MC_sequence" is the sample number per MPI rank, multiplying this number to the MPI process number is then the total sample number. To run the sampling process, just:

mpirun -np NPROC python fire.py

After the sampling completes, the "energy" file reports the energy value for the quantum model evaluated. Because of the large neural network, MCMC may takes a long time without GPUs. For example, collecting 945 samples by one Xeon Silver 4215R core takes two hours. 
