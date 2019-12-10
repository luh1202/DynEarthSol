#ifndef __CUDAKERNEL_H
#define __CUDAKERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>

//int set_constant_parameters(double dt, double surface_temperature, double pls0, double pls1, double friction_angle0, double friction_angle1, double dilation_angle0, double dilation_angle1, double cohesion0, double conhesion1, double tension_max);
int set_constant_parameters(double dt, double surface_temperature);

__global__ void update_stress_kernel(double *stress, double *strain_rate, double *edvoldt, double *strain, double *plstrain, double *delta_plstrain, double *dpressure, int nelem);
__global__ void update_temperature_kernel0(double *temperature, double *temp_support, double *shpdx, double *shpdz, double *volume, int *connectivity, int nelem);
__global__ void update_temperature_kernel1(double *temperature, double *temp_support, double *tmass, unsigned int *bcflag, int nnode);
__global__ void update_strain_rate_kernel(int *connectivity, double *shpdx, double *shpdz, double *strain_rate, double *vel, double *dvoldt_support, double *volume, int nelem);
__global__ void compute_dvoldt_kernel(double *dvoldt, double *dvoldt_support, double *volume_n, int nnode);
__global__ void compute_edvoldt_kernel(int *connectivity, double *dvoldt, double *edvoldt, int nelem);
__global__ void NMD_stress_kernel0(int *connectivity, double *dp_nd, double *dpressure, double *volume, int nelem);
__global__ void NMD_stress_kernel1(double *dp_nd, double *volume_n, int nnode);
__global__ void NMD_stress_kernel2(int *connectivity, double *dp_nd, double *stress, double *dpressure, int nelem);
__global__ void update_force_kernel(int *connectivity, double *shpdx, double *shpdz, double *stress, double *volume, double *rho, double *force_support, double *force, int nelem);
__global__ void update_velocity_kernel(double *mass, double *vel, double *force, int nnode);
__global__ void update_force_kernel(int *connectivity, double *shpdx, double *shpdz, double *stress, double *volume, double *temperature, double *force, int nelem);
__global__ void update_velocity_kernel(double *mass, double *vel, double *force, int nnode);
__global__ void update_coordinate_kernel(double *coord, double *vel, int nnode);
__global__ void compute_volume_kernel(int *connectivity, double *coord, double *volume, double *shpdx, double *shpdz, int nelem);
__global__ void compute_mass_kernel(int *connectivity, double *volume, double *volume_n, double *mass, double *tmass, double *temperature, double pseudo_speed, int nelem);
__global__ void rotate_stress_kernel(int *connectivity, double *strain, double *stress, double *shpdx, double *shpdz, double *vel, int nelem);
__global__ void apply_damping_kernel(double *force, double *vel, int nnode);
__global__ void apply_stress_bcs_kernel(double *coord, int *eindex, int *findex, int *connectivity, double *force, double *temperature, int nsize, int nelem);
__global__ void apply_vbcs_kernel(unsigned int *bcflag, double *vel, int nnode);
__global__ void surface_processes_kernel0(int *connectivity, int *eindex, int *findex, double *total_dx, double *total_slope, double *coord, int nsize, int nelem);
__global__ void surface_processes_kernel1(int *nindex, double *total_dx, double *total_slope, double *coord, int nsize);

__global__ void get_minh_kernel(int *connectivity, double *coord, double *volume, double *global_minh, int nelem);
__global__ void worst_elem_quality_kernel(int *connectivity, double *coord, double *q, double *global_minv, int nelem);
__global__ void get_max_bottom_kernel(unsigned int *bcflag, double *coord, double *gloab_maxd, int nnode, double bottom);

//3D version
__global__ void update_stress_3D_kernel(double *stress, double *strain_rate, double *edvoldt, double *strain, double *plstrain, double *delta_plstrain, double *dpressure, int nelem);
__global__ void update_temperature_3D_kernel0(double *temperature, double *temp_support, double *shpdx, double *shpdy, double *shpdz, double *volume, int *connectivity, int nelem);
__global__ void update_temperature_3D_kernel1(double *temperature, double *temp_support, double *tmass, unsigned int *bcflag, int nnode);
__global__ void update_strain_rate_3D_kernel(int *connectivity, double *shpdx, double *shpdy, double *shpdz, double *strain_rate, double *vel, double *dvoldt_support, double *volume, int nelem);
__global__ void compute_dvoldt_3D_kernel(double *dvoldt, double *dvoldt_support, double *volume_n, int nnode);
__global__ void compute_edvoldt_3D_kernel(int *connectivity, double *dvoldt, double *edvoldt, int nelem);
__global__ void NMD_stress_3D_kernel0(int *connectivity, double *dp_nd, double *dpressure, double *volume, int nelem);
__global__ void NMD_stress_3D_kernel1(double *dp_nd, double *volume_n, int nnode);
__global__ void NMD_stress_3D_kernel2(int *connectivity, double *dp_nd, double *stress, double *dpressure, int nelem);
__global__ void update_force_3D_kernel(int *connectivity, double *shpdx, double *shpdy, double *shpdz, double *stress, double *volume, double *rho, double *force_support, double *force, int nelem);
__global__ void update_velocity_3D_kernel(double *mass, double *vel, double *force, int nnode);
__global__ void update_force_3D_kernel(int *connectivity, double *shpdx, double *shpdy, double *shpdz, double *stress, double *volume, double *temperature, double *force, int nelem);
__global__ void update_velocity_3D_kernel(double *mass, double *vel, double *force, int nnode);
__global__ void update_coordinate_3D_kernel(double *coord, double *vel, int nnode);
__global__ void compute_volume_3D_kernel(int *connectivity, double *coord, double *volume, double *shpdx, double *shpdy, double *shpdz, int nelem);
__global__ void compute_mass_3D_kernel(int *connectivity, double *volume, double *volume_n, double *mass, double *tmass, double *temperature, double pseudo_speed, int nelem);
__global__ void rotate_stress_3D_kernel(int *connectivity, double *strain, double *stress, double *shpdx, double *shpdy, double *shpdz, double *vel, int nelem);
__global__ void apply_damping_3D_kernel(double *force, double *vel, int nnode);
__global__ void apply_stress_bcs_3D_kernel(double *coord, int *eindex, int *findex, int *connectivity, double *force, double *temperature, int nsize, int nelem);
__global__ void apply_vbcs_3D_kernel(unsigned int *bcflag, double *vel, int nnode);
__global__ void surface_processes_3D_kernel0(int *connectivity, int *eindex, int *findex, double *total_dx, double *total_slope, double *coord, int nsize, int nelem);
__global__ void surface_processes_3D_kernel1(int *nindex, double *total_dx, double *total_slope, double *coord, int nsize);

__global__ void get_minh_3D_kernel(int *connectivity, double *coord, double *volume, double *global_minh, int nelem);
__global__ void worst_elem_quality_3D_kernel(int *connectivity, double *coord, double *q, double *global_minv, int nelem);
__global__ void get_max_bottom_3D_kernel(unsigned int *bcflag, double *coord, double *gloab_maxd, int nnode, double bottom);

#endif