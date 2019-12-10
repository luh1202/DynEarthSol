#ifndef DYNEARTHSOL3D_CUDA_HPP
#define DYNEARTHSOL3D_CUDA_HPP

void launch_cuda_device_synchronize();

//int launch_set_constant_parameters(double dt, double surface_temperature, double pls0, double pls1, double friction_angle0, double friction_angle1, double dilation_angle0, double dilation_angle1, double cohesion0, double conhesion1, double tension_max);
int launch_set_constant_parameters(double dt, double surface_temperature);
int launch_cudaMemset(void *devPtr, int value, size_t count);
int launch_cudaMemcpy(void *dst, const void *src, size_t count, int direction);
int launch_cudaMemcpyAsync(void *dst, const void *src, size_t count, int direction);
int launch_cudaMalloc(void **devPtr, size_t size);
void launch_cudaFree(void *devPtr);
int launch_cudaMallocHost(void **hostPtr, size_t size);
void launch_cudaFreeHost(void *hostPtr);

void launch_update_temperature(double *temperature, double *temp_support, double *shpdx, double *shpdy, double *shpdz, double *volume, int *connectivity, double *tmass, unsigned int *bcflag, int nelem, int nnode);
void launch_update_stress(double *stress, double *strain_rate, double *edvoldt, double *strain, double *plstrain, double *delta_plstrain, double *dpressure, int nelem);
void launch_update_strain_rate(int *connectivity, double *shpdx, double *shpdy, double *shpdz, double *s, double *vel, double *dvoldt, double *edvoldt, double *dvoldt_support, double *volume, double *volume_n, int nnode, int nelem);
void launch_NMD_stress(int *connectivity, double *dp_nd, double *dpressure, double *volume, double *volume_n, double *stress, int nelem, int nnode);
void launch_update_force(int *connectivity, double *shpdx, double *shpdy, double *shpdz, double *stress, double *volume, double *temperature, double *force, double *vel, double *coord, int *eindex, int *findex, int nelem, int nnode, int nsize);
void launch_update_mesh(int *connectivity, double *coord, double *vel, double *volume, double *volume_n, double *shpdx, double *shpdy, double *shpdz, double *mass, double *tmass, double *temperature, double pseudo_speed, int nelem, int nnode, int bce, int bcn, int *bceindex, int *bcfinex, int *bcnindex, double *total_dx, double *total_slop, bool update_coordinate);
void launch_rotate_stress(int *connectivity, double *strain, double *stress, double *shpdx, double *shpdy, double *shpdz, double *vel, int nelem);
void launch_update_velocity(double *mass, double *vel, double *force, int nnode);
void launch_apply_vbcs(unsigned int *bcflag, double *vel, int nnode);

double launch_get_minh(int *connectivity, double *coord, double *volume, int nelem);
int launch_get_worst_quality(int *connectivity, double *coord, double *global_minv, double *global_minq, int nelem);
double launch_max_bottom_distance(unsigned int *bcflag, double *coord, int nnode, double bottom);

#endif