#ifndef DYNEARTHSOL3D_FIELDS_CUDA_HPP
#define DYNEARTHSOL3D_FIELDS_CUDA_HPP

void allocate_gpu_variable(const Param &param, Variables& var);
void delete_gpu_variable(Variables& var);

#endif
