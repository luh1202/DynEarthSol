#include <iostream>

#include "cudaKernel.cuh"
#include "../constants.hpp"

#define THREAD_PER_BLOCK 1024
#define THREAD_PER_BLOCK1 512

float totaltime = 0;

extern "C++"
int launch_cudaMemset(void *devPtr, int value, size_t count) {
	if (cudaMemset(devPtr, value, count) != cudaSuccess) {
		return -1;
	}

	return 0;
}

extern "C++"
int launch_cudaMemcpy(void *dst, const void *src, size_t count, int direction) {
	if (direction == 0) {
		if (cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice) != cudaSuccess) {
			return -1;
		}
	} else {
		if (cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost) != cudaSuccess) {
			return -1;
		}
	}

	return 0;
}

extern "C++"
int launch_cudaMallocHost(void **hostPtr, size_t size) {
    if (cudaMallocHost(hostPtr, size) != cudaSuccess) {
        return -1;
    }

    return 0;
}

extern "C++"
void launch_cudaFreeHost(void *hostPtr) {
    cudaFreeHost(hostPtr);
}

extern "C++"
int launch_cudaMemcpyAsync(void *dst, const void *src, size_t count, int direction) {
    /*
    if (direction == 0) {
        if (cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, 0) != cudaSuccess) {
            return -1;
        }
    } else {
        if (cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, 0) != cudaSuccess) {
            return -1;
        }
    }

    return 0;
    */
    if (direction == 0) {
		if (cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice) != cudaSuccess) {
			return -1;
		}
	} else {
		if (cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost) != cudaSuccess) {
			return -1;
		}
	}

	return 0;
}

extern "C++"
int launch_cudaMalloc(void **devPtr, size_t size) {
	if (cudaMalloc(devPtr, size) != cudaSuccess) {
		return -1;
	}

	return 0;
}

extern "C++"
void launch_cudaFree(void *devPtr) {
	cudaFree(devPtr);
}

extern "C++"
void launch_cuda_device_synchronize() {
    cudaDeviceSynchronize();
}

extern "C++"
int launch_set_constant_parameters(double dt, double surface_temperature) {
    return set_constant_parameters(dt, surface_temperature);
}

extern "C++"
void launch_update_temperature(double *temperature, double *temp_support, double *shpdx, double *shpdy, double *shpdz, double *volume, int *connectivity, double *tmass, unsigned int *bcflag, int nelem, int nnode) {
	cudaMemset(temp_support, 0, nnode * sizeof(double));
#ifdef THREED
	update_temperature_3D_kernel0<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(temperature, temp_support, shpdx, shpdy, shpdz, volume, connectivity, nelem);
	update_temperature_3D_kernel1<<<nnode / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(temperature, temp_support, tmass, bcflag, nnode);
#else
	update_temperature_kernel0<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(temperature, temp_support, shpdx, shpdz, volume, connectivity, nelem);
	update_temperature_kernel1<<<nnode / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(temperature, temp_support, tmass, bcflag, nnode);
#endif
}

extern "C++"
void launch_update_stress(double *stress, double *strain_rate, double *edvoldt, double *strain, double *plstrain, double *delta_plstrain, double *dpressure, int nelem) {
#ifdef THREED
	update_stress_3D_kernel<<<nelem / THREAD_PER_BLOCK1 + 1, THREAD_PER_BLOCK1>>>(stress, strain_rate, edvoldt, strain, plstrain, delta_plstrain, dpressure, nelem);
#else
	update_stress_kernel<<<nelem / THREAD_PER_BLOCK1 + 1, THREAD_PER_BLOCK1>>>(stress, strain_rate, edvoldt, strain, plstrain, delta_plstrain, dpressure, nelem);
#endif
}

extern "C++"
void launch_update_strain_rate(int *connectivity, double *shpdx, double *shpdy, double *shpdz, double *s, double *vel, double *dvoldt, double *edvoldt, double *dvoldt_support, double *volume, double *volume_n, int nnode, int nelem) {
    cudaMemset(dvoldt_support, 0, nnode * sizeof(double));
#ifdef THREED
	update_strain_rate_3D_kernel<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, shpdx, shpdy, shpdz, s, vel, dvoldt_support, volume, nelem);
	compute_dvoldt_3D_kernel<<<nnode / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(dvoldt, dvoldt_support, volume_n, nnode);
	compute_edvoldt_3D_kernel<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, dvoldt, edvoldt, nelem);
#else
	update_strain_rate_kernel<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, shpdx, shpdz, s, vel, dvoldt_support, volume, nelem);
	compute_dvoldt_kernel<<<nnode / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(dvoldt, dvoldt_support, volume_n, nnode);
	compute_edvoldt_kernel<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, dvoldt, edvoldt, nelem);
#endif
}

extern "C++"
void launch_NMD_stress(int *connectivity, double *dp_nd, double *dpressure, double *volume, double *volume_n, double *stress, int nelem, int nnode) {
	cudaMemset(dp_nd, 0, nnode * sizeof(double));
#ifdef THREED
	NMD_stress_3D_kernel0<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, dp_nd, dpressure, volume, nelem);
	NMD_stress_3D_kernel1<<<nnode / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(dp_nd, volume_n, nnode);
	NMD_stress_3D_kernel2<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, dp_nd, stress, dpressure, nelem);
#else
	NMD_stress_kernel0<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, dp_nd, dpressure, volume, nelem);
	NMD_stress_kernel1<<<nnode / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(dp_nd, volume_n, nnode);
	NMD_stress_kernel2<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, dp_nd, stress, dpressure, nelem);
#endif
}

extern "C++"
void launch_update_force(int *connectivity, double *shpdx, double *shpdy, double *shpdz, double *stress, double *volume, double *temperature, double *force, double *vel, double *coord, int *eindex, int *findex, int nelem, int nnode, int nsize) {
	cudaMemset(force, 0, nnode * sizeof(double) * NDIMS);
#ifdef THREED
	update_force_3D_kernel<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, shpdx, shpdy, shpdz, stress, volume, temperature, force, nelem);
	apply_stress_bcs_3D_kernel<<<nsize / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(coord, eindex, findex, connectivity, force, temperature, nsize, nelem);
	apply_damping_3D_kernel<<<nnode / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(force, vel, nnode);
#else
	update_force_kernel<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, shpdx, shpdz, stress, volume, temperature, force, nelem);
	apply_stress_bcs_kernel<<<nsize / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(coord, eindex, findex, connectivity, force, temperature, nsize, nelem);
	apply_damping_kernel<<<nnode / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(force, vel, nnode);
#endif
}

extern "C++"
void launch_update_velocity(double *mass, double *vel, double *force, int nnode) {
#ifdef THREED
	update_velocity_3D_kernel<<<nnode / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(mass, vel, force, nnode);
#else
	update_velocity_kernel<<<nnode / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(mass, vel, force, nnode);
#endif
}

extern "C++"
void launch_update_mesh(int *connectivity, double *coord, double *vel, double *volume, double *volume_n, double *shpdx, double *shpdy, double *shpdz, double *mass, double *tmass, double *temperature, double pseudo_speed, int nelem, int nnode, int bce, int bcn, int *bceindex, int *bcfindex, int *bcnindex, double *total_dx, double *total_slope, bool update_coordinate) {
    if (update_coordinate) {
    	cudaMemset(total_dx, 0, nnode * sizeof(double));
		cudaMemset(total_slope, 0, nnode * sizeof(double));
#ifdef THREED
		update_coordinate_3D_kernel<<<nnode / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(coord, vel, nnode);
		surface_processes_3D_kernel0<<<bce / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, bceindex, bcfindex, total_dx, total_slope, coord, bce, nelem);
		surface_processes_3D_kernel1<<<bcn / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(bcnindex, total_dx, total_slope, coord, bcn);
#else
		update_coordinate_kernel<<<nnode / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(coord, vel, nnode);
		surface_processes_kernel0<<<bce / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, bceindex, bcfindex, total_dx, total_slope, coord, bce, nelem);
		surface_processes_kernel1<<<bcn / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(bcnindex, total_dx, total_slope, coord, bcn);
#endif
    }
#ifdef THREED
	compute_volume_3D_kernel<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, coord, volume, shpdx, shpdy, shpdz, nelem);
#else
	compute_volume_kernel<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, coord, volume, shpdx, shpdz, nelem);
#endif
	cudaMemset(volume_n, 0, nnode * sizeof(double));
	cudaMemset(mass, 0, nnode * sizeof(double));
	cudaMemset(tmass, 0, nnode * sizeof(double));
#ifdef THREED
	compute_mass_3D_kernel<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, volume, volume_n, mass, tmass, temperature, pseudo_speed, nelem);
#else
	compute_mass_kernel<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, volume, volume_n, mass, tmass, temperature, pseudo_speed, nelem);
#endif
}

extern "C++"
void launch_rotate_stress(int *connectivity, double *strain, double *stress, double *shpdx, double *shpdy, double *shpdz, double *vel, int nelem) {
#ifdef THREED
	rotate_stress_3D_kernel<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, strain, stress, shpdx, shpdy, shpdz, vel, nelem);
#else
	rotate_stress_kernel<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, strain, stress, shpdx, shpdz, vel, nelem);
#endif
}

extern "C++"
void launch_apply_vbcs(unsigned int *bcflag, double *vel, int nnode) {
#ifdef THREED
	apply_vbcs_3D_kernel<<<nnode / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(bcflag, vel, nnode);
#else
	apply_vbcs_kernel<<<nnode / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(bcflag, vel, nnode);
#endif
}

extern "C++"
double launch_get_minh(int *connectivity, double *coord, double *volume, int nelem) {
	double *d_global_minh;
	double global_minh = 1e102;

	if (cudaMalloc((void**)&d_global_minh, sizeof(double)) != cudaSuccess) {
		return -1;
	}

	if (cudaMemcpy(d_global_minh, &global_minh, sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
		return -1;
	}

#ifdef THREED
	get_minh_3D_kernel<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, coord, volume, d_global_minh, nelem);
#else
	get_minh_kernel<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, coord, volume, d_global_minh, nelem);
#endif

	if (cudaMemcpy(&global_minh, d_global_minh, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
		return -1;
	}

	cudaFree(d_global_minh);

	return global_minh;
}

extern "C++"
int launch_get_worst_quality(int *connectivity, double *coord, double *global_minv, double *global_minq, int nelem) {
	double *d_global_minq;
	double *d_global_minv;
	*global_minv = 1e102;
	*global_minq = 1;

	if (cudaMalloc((void **)&d_global_minq, sizeof(double)) != cudaSuccess ||
		cudaMalloc((void **)&d_global_minv, sizeof(double)) != cudaSuccess) {
		return -1;
	}

	if (cudaMemcpy(d_global_minq, global_minq, sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
		cudaMemcpy(d_global_minv, global_minv, sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
		return -1;
	}

#ifdef THREED
	worst_elem_quality_3D_kernel<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, coord, d_global_minq, d_global_minv, nelem);
#else
	worst_elem_quality_kernel<<<nelem / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(connectivity, coord, d_global_minq, d_global_minv, nelem);
#endif

	if (cudaMemcpy(global_minq, d_global_minq, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(global_minv, d_global_minv, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
		return -1;
	}

	cudaFree(d_global_minv);
	cudaFree(d_global_minq);

	return 0;
}

double launch_max_bottom_distance(unsigned int *bcflag, double *coord, int nnode, double bottom) {
	double *d_global_max;
	double global_max = 0;

	if (cudaMalloc((void**)&d_global_max, sizeof(double)) != cudaSuccess) {
		return -1;
	}

	if (cudaMemcpy(d_global_max, &global_max, sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
		return -1;
	}

#ifdef THREED
	get_max_bottom_3D_kernel<<<nnode / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(bcflag, coord, d_global_max, nnode, bottom);
#else
	get_max_bottom_kernel<<<nnode / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK>>>(bcflag, coord, d_global_max, nnode, bottom);
#endif

	if (cudaMemcpy(&global_max, d_global_max, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
		return -1;
	}

	cudaFree(d_global_max);

	return global_max;
}