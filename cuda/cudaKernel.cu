#include <float.h>

#include "../constants.hpp"
#include "cudaKernel.cuh"

#include <device_functions.h>
#include <cuda_runtime_api.h>

#define FULL_MASK 0xffffffff
#define M_SQRT3   1.73205080756887729352744634151
#define SQR(x)    ((x)*(x))                        // x^2 

__constant__ double c_dt;
__constant__ double c_pls0 = 0.;
__constant__ double c_pls1 = 0.5;
__constant__ double c_friction_angle0 = 30;
__constant__ double c_dilation_angle0 = 0;
__constant__ double c_friction_angle1 = 30;
__constant__ double c_dilation_angle1 = 0;
__constant__ double c_cohesion0 = 4.4e7;
__constant__ double c_cohesion1 = 4e6;
__constant__ double c_tension_max = 1e9;
__constant__ double c_k = 3;
__constant__ double c_surface_temperature;
__constant__ double c_gravity = 10.;
__constant__ double c_damping_factor = 0.8;
__constant__ unsigned int c_vbc_x0 = 1;
__constant__ unsigned int c_vbc_x1 = 1;
__constant__ double c_vbc_val_x0 = -1e-9;
__constant__ double c_vbc_val_x1 = 1e-9;
__constant__ unsigned int c_vbc_y0 = 1;
__constant__ unsigned int c_vbc_y1 = 1;
__constant__ double c_vbc_val_y0 = 0;
__constant__ double c_vbc_val_y1 = 0;
__constant__ double c_compensation_pressure = 2.7e8;
__constant__ double c_zlength = 10000;
__constant__ double c_winker_delta_rho = 0.;
__constant__ double c_bulkm = 50e9;
__constant__ double c_shearm = 30e9;
__constant__ double c_rho = 2700;
__constant__ double c_celsius0 = 273;
__constant__ double c_alpha = 0;
__constant__ double c_cp = 1000;
__constant__ double c_surface_diffusivity = 1e-7;

__device__ int cuda_dsyevc3(double A[3][3], double w[3]);
__device__ int cuda_dsyevh3(double A[3][3], double Q[3][3], double w[3]);
__device__ int cuda_dsyevq3(double A[3][3], double Q[3][3], double w[3]);
__device__ void cuda_dsytrd3(double A[3][3], double Q[3][3], double d[3], double e[2]);

__device__ inline void MyAtomicAdd(float *address, float value){
   int oldval, newval, readback;
 
   oldval = __float_as_int(*address);
   newval = __float_as_int(__int_as_float(oldval) + value);
   while ((readback=atomicCAS((int *)address, oldval, newval)) != oldval) 
     {
      oldval = readback;
      newval = __float_as_int(__int_as_float(oldval) + value);
     }
 }

 __device__ inline void MyAtomicAdd_8(double *address, double value) {
   unsigned long long oldval, newval, readback;
 
   oldval = __double_as_longlong(*address);
   newval = __double_as_longlong(__longlong_as_double(oldval) + value);
   while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
    {
      oldval = readback;
      newval = __double_as_longlong(__longlong_as_double(oldval) + value);
    }
 }

__device__ inline void MyAtomicMin_8(double *address, double value) {
   unsigned long long oldval, newval, readback;
 
   oldval = __double_as_longlong(*address);
   newval = __double_as_longlong(fmin(__longlong_as_double(oldval), value));
   while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
    {
      oldval = readback;
      newval = __double_as_longlong(fmin(__longlong_as_double(oldval), value));
    }
}

__device__ inline void MyAtomicMax_8(double *address, double value) {
   unsigned long long oldval, newval, readback;
 
   oldval = __double_as_longlong(*address);
   newval = __double_as_longlong(fmax(__longlong_as_double(oldval), value));
   while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
    {
      oldval = readback;
      newval = __double_as_longlong(fmax(__longlong_as_double(oldval), value));
    }
}

//int set_constant_parameters(double dt, double surface_temperature, double pls0, double pls1, double friction_angle0, double friction_angle1, double dilation_angle0, double dilation_angle1, double cohesion0, double conhesion1, double tension_max) {
int set_constant_parameters(double dt, double surface_temperature) {
	if (cudaMemcpyToSymbol(c_dt, &dt, sizeof(double)) != cudaSuccess) {
        return -1;
    }

    if (cudaMemcpyToSymbol(c_surface_temperature, &surface_temperature, sizeof(double)) != cudaSuccess) {
        return -1;
    }
/*
    if (cudaMemcpyToSymbol(c_pls0, &pls0, sizeof(double)) != cudaSuccess) {
        return -1;
    }

    if (cudaMemcpyToSymbol(c_pls1, &pls1, sizeof(double)) != cudaSuccess) {
        return -1;
    }

    if (cudaMemcpyToSymbol(c_friction_angle0, &friction_angle0, sizeof(double)) != cudaSuccess) {
        return -1;
    }

    if (cudaMemcpyToSymbol(c_friction_angle1, &friction_angle1, sizeof(double)) != cudaSuccess) {
        return -1;
    }

    if (cudaMemcpyToSymbol(c_dilation_angle0, &dilation_angle0, sizeof(double)) != cudaSuccess) {
        return -1;
    }

    if (cudaMemcpyToSymbol(c_dilation_angle1, &dilation_angle1, sizeof(double)) != cudaSuccess) {
        return -1;
    }

    if (cudaMemcpyToSymbol(c_cohesion0, &cohesion0, sizeof(double)) != cudaSuccess) {
        return -1;
    }

    if (cudaMemcpyToSymbol(c_cohesion1, &cohesion1, sizeof(double)) != cudaSuccess) {
        return -1;
    }

    if (cudaMemcpyToSymbol(c_tension_max, &tension_max, sizeof(double)) != cudaSuccess) {
        return -1;
    }
*/
    return 0;
}

__global__ void update_temperature_kernel0(double *temperature, double *temp_support, double *shpdx, double *shpdz, double *volume, int *connectivity, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index0 + nelem * 2;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];

		double shpdx0 = shpdx[index0];
		double shpdx1 = shpdx[index1];
		double shpdx2 = shpdx[index2];

		double shpdz0 = shpdz[index0];
		double shpdz1 = shpdz[index1];
		double shpdz2 = shpdz[index2];

		double kv = c_k * volume[index0];

		double D00 = shpdx0 * shpdx0 + shpdz0 * shpdz0;
		double D11 = shpdx1 * shpdx1 + shpdz1 * shpdz1;
		double D22 = shpdx2 * shpdx2 + shpdz2 * shpdz2;
		double D01 = shpdx0 * shpdx1 + shpdz0 * shpdz1;
		double D02 = shpdx0 * shpdx2 + shpdz0 * shpdz2;
		double D12 = shpdx1 * shpdx2 + shpdz1 * shpdz2;

		// now we use atomic add, can be optimized in the future
		double t0 = temperature[conn0];
		double t1 = temperature[conn1];
		double t2 = temperature[conn2];

		double out0 = D00 * t0 + D01 * t1 + D02 * t2;
		double out1 = D01 * t0 + D11 * t1 + D12 * t2;
		double out2 = D02 * t0 + D12 * t1 + D22 * t2;

		out0 *= kv;
		out1 *= kv;
		out2 *= kv;

		MyAtomicAdd_8(temp_support + conn0, out0);
		MyAtomicAdd_8(temp_support + conn1, out1);
		MyAtomicAdd_8(temp_support + conn2, out2);
	}
}

__global__ void update_temperature_3D_kernel0(double *temperature, double *temp_support, double *shpdx, double *shpdy, double *shpdz, double *volume, int *connectivity, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index1 + nelem;
		int index3 = index2 + nelem;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];
		int conn3 = connectivity[index3];

		double shpdx0 = shpdx[index0];
		double shpdx1 = shpdx[index1];
		double shpdx2 = shpdx[index2];
		double shpdx3 = shpdx[index3];

		double shpdy0 = shpdy[index0];
		double shpdy1 = shpdy[index1];
		double shpdy2 = shpdy[index2];
		double shpdy3 = shpdy[index3];

		double shpdz0 = shpdz[index0];
		double shpdz1 = shpdz[index1];
		double shpdz2 = shpdz[index2];
		double shpdz3 = shpdz[index3];

		double kv = c_k * volume[index0];

		double D00 = shpdx0 * shpdx0 + shpdy0 * shpdy0 + shpdz0 * shpdz0;
		double D01 = shpdx0 * shpdx1 + shpdy0 * shpdy1 + shpdz0 * shpdz1;
		double D02 = shpdx0 * shpdx2 + shpdy0 * shpdy2 + shpdz0 * shpdz2;
		double D03 = shpdx0 * shpdx3 + shpdy0 * shpdy3 + shpdz0 * shpdz3;
		double D11 = shpdx1 * shpdx1 + shpdy1 * shpdy1 + shpdz1 * shpdz1;
		double D12 = shpdx1 * shpdx2 + shpdy1 * shpdy2 + shpdz1 * shpdz2;
		double D13 = shpdx1 * shpdx3 + shpdy1 * shpdy3 + shpdz1 * shpdz3;
		double D22 = shpdx2 * shpdx2 + shpdy2 * shpdy2 + shpdz2 * shpdz2;
		double D23 = shpdx2 * shpdx3 + shpdy2 * shpdy3 + shpdz2 * shpdz3;
		double D33 = shpdx3 * shpdx3 + shpdy3 * shpdy3 + shpdz3 * shpdz3;

		// now we use atomic add, can be optimized in the future
		double t0 = temperature[conn0];
		double t1 = temperature[conn1];
		double t2 = temperature[conn2];
		double t3 = temperature[conn3];

		double out0 = D00 * t0 + D01 * t1 + D02 * t2 + D03 * t3;
		double out1 = D01 * t0 + D11 * t1 + D12 * t2 + D13 * t3;
		double out2 = D02 * t0 + D12 * t1 + D22 * t2 + D23 * t3;
		double out3 = D03 * t0 + D13 * t1 + D23 * t3 + D33 * t3;

		out0 *= kv;
		out1 *= kv;
		out2 *= kv;
		out3 *= kv;

		MyAtomicAdd_8(temp_support + conn0, out0);
		MyAtomicAdd_8(temp_support + conn1, out1);
		MyAtomicAdd_8(temp_support + conn2, out2);
		MyAtomicAdd_8(temp_support + conn3, out3);
	}
}

__global__ void update_temperature_kernel1(double *temperature, double *temp_support, double *tmass, unsigned int *bcflag, int nnode) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nnode) {
		int flag = bcflag[index0];

		if (flag & BOUNDZ1) {
			temperature[index0] = c_surface_temperature;
		} else {
			temperature[index0] -= temp_support[index0] * c_dt / tmass[index0];
		}
	}
}

__global__ void update_temperature_3D_kernel1(double *temperature, double *temp_support, double *tmass, unsigned int *bcflag, int nnode) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nnode) {
		int flag = bcflag[index0];

		if (flag & BOUNDZ1) {
			temperature[index0] = c_surface_temperature;
		} else {
			temperature[index0] -= temp_support[index0] * c_dt / tmass[index0];
		}
	}
}

__global__ void update_strain_rate_kernel(int *connectivity, double *shpdx, double *shpdz, double *strain_rate, double *vel, double *dvoldt_support, double *volume, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index0 + nelem * 2;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];

		double vel00 = vel[conn0 * 2];
		double vel01 = vel[conn0 * 2 + 1];
		double vel10 = vel[conn1 * 2];
		double vel11 = vel[conn1 * 2 + 1];
		double vel20 = vel[conn2 * 2];
		double vel21 = vel[conn2 * 2 + 1];

		double shpdx0 = shpdx[index0];
		double shpdx1 = shpdx[index1];
		double shpdx2 = shpdx[index2];

		double shpdz0 = shpdz[index0];
		double shpdz1 = shpdz[index1];
		double shpdz2 = shpdz[index2];

		double s0 = vel00 * shpdx0 + vel10 * shpdx1 + vel20 * shpdx2;
		double s1 = vel01 * shpdz0 + vel11 * shpdz1 + vel21 * shpdz2;
		double s2 = vel00 * shpdz0 + vel10 * shpdz1 + vel20 * shpdz2;

		s2 += vel01 * shpdx0 + vel11 * shpdx1 + vel21 * shpdx2;
		s2 *= 0.5;

		strain_rate[index0] = s0;
		strain_rate[index1] = s1;
		strain_rate[index2] = s2;

		double dj = s0 + s1;

		dj *= volume[index0];

		MyAtomicAdd_8(dvoldt_support + conn0, dj);
		MyAtomicAdd_8(dvoldt_support + conn1, dj);
		MyAtomicAdd_8(dvoldt_support + conn2, dj);
	}
}

__global__ void update_strain_rate_3D_kernel(int *connectivity, double *shpdx, double *shpdy, double *shpdz, double *strain_rate, double *vel, double *dvoldt_support, double *volume, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index1 + nelem;
		int index3 = index2 + nelem;
		int index4 = index3 + nelem;
		int index5 = index4 + nelem;

		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];
		int conn3 = connectivity[index3];

		double vel00 = vel[conn0 * 3];
		double vel01 = vel[conn0 * 3 + 1];
		double vel02 = vel[conn0 * 3 + 2];

		double vel10 = vel[conn1 * 3];
		double vel11 = vel[conn1 * 3 + 1];
		double vel12 = vel[conn1 * 3 + 2];

		double vel20 = vel[conn2 * 3];
		double vel21 = vel[conn2 * 3 + 1];
		double vel22 = vel[conn2 * 3 + 2];

		double vel30 = vel[conn3 * 3];
		double vel31 = vel[conn3 * 3 + 1];
		double vel32 = vel[conn3 * 3 + 2];

		double shpdx0 = shpdx[index0];
		double shpdx1 = shpdx[index1];
		double shpdx2 = shpdx[index2];
		double shpdx3 = shpdx[index3];

		double shpdy0 = shpdy[index0];
		double shpdy1 = shpdy[index1];
		double shpdy2 = shpdy[index2];
		double shpdy3 = shpdy[index3];

		double shpdz0 = shpdz[index0];
		double shpdz1 = shpdz[index1];
		double shpdz2 = shpdz[index2];
		double shpdz3 = shpdz[index3];

		double s0 = vel00 * shpdx0 + vel10 * shpdx1 + vel20 * shpdx2 + vel30 * shpdx3;
		double s1 = vel01 * shpdy0 + vel11 * shpdy1 + vel21 * shpdy2 + vel31 * shpdy3;
		double s2 = vel02 * shpdz0 + vel12 * shpdz1 + vel22 * shpdz2 + vel32 * shpdz3;
		double s3 = vel00 * shpdy0 + vel10 * shpdy1 + vel20 * shpdy2 + vel30 * shpdy3;
		s3 += vel01 * shpdx0 + vel11 * shpdx1 + vel21 * shpdx2 + vel31 * shpdx3;
		s3 *= 0.5;
		double s4 = vel00 * shpdz0 + vel10 * shpdz1 + vel20 * shpdz2 + vel30 * shpdz3;
		s4 += vel02 * shpdx0 + vel12 * shpdx1 + vel22 * shpdx2 + vel32 * shpdx3;
		s4 *= 0.5;
		double s5 = vel01 * shpdz0 + vel11 * shpdz1 + vel21 * shpdz2 + vel31 * shpdz3;
		s5 += vel02 * shpdy0 + vel12 * shpdy1 + vel22 * shpdy2 + vel32 * shpdy3;
		s5 *= 0.5;

		strain_rate[index0] = s0;
		strain_rate[index1] = s1;
		strain_rate[index2] = s2;
		strain_rate[index3] = s3;
		strain_rate[index4] = s4;
		strain_rate[index5] = s5;

		/* dvoldt is the volumetric strain rate, weighted by the element volume,
	     * lumped onto the nodes.
	     */
		double dj = s0 + s1 + s2;

		dj *= volume[index0];

		MyAtomicAdd_8(dvoldt_support + conn0, dj);
		MyAtomicAdd_8(dvoldt_support + conn1, dj);
		MyAtomicAdd_8(dvoldt_support + conn2, dj);
		MyAtomicAdd_8(dvoldt_support + conn3, dj);
	}
}

__global__ void compute_dvoldt_kernel(double *dvoldt, double *dvoldt_support, double *volume_n, int nnode) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nnode) {
		dvoldt[index0] = dvoldt_support[index0] / volume_n[index0];
	}
}

__global__ void compute_dvoldt_3D_kernel(double *dvoldt, double *dvoldt_support, double *volume_n, int nnode) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nnode) {
		dvoldt[index0] = dvoldt_support[index0] / volume_n[index0];
	}
}

__global__ void compute_edvoldt_kernel(int *connectivity, double *dvoldt, double *edvoldt, int nelem) {
    /* edvoldt is the averaged (i.e. smoothed) dvoldt on the element.
     * It is used in update_stress() to prevent mesh locking.
     */
    int index0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (index0 < nelem) {
    	int index1 = index0 + nelem;
		int index2 = index0 + nelem * 2;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];

		double dj = dvoldt[conn0];
		dj += dvoldt[conn1];
		dj += dvoldt[conn2];

		edvoldt[index0] = dj / NODES_PER_ELEM;
    }
}

__global__ void compute_edvoldt_3D_kernel(int *connectivity, double *dvoldt, double *edvoldt, int nelem) {
    /* edvoldt is the averaged (i.e. smoothed) dvoldt on the element.
     * It is used in update_stress() to prevent mesh locking.
     */
    int index0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (index0 < nelem) {
    	int index1 = index0 + nelem;
		int index2 = index1 + nelem;
		int index3 = index2 + nelem;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];
		int conn3 = connectivity[index3];

		double dj = dvoldt[conn0];
		dj += dvoldt[conn1];
		dj += dvoldt[conn2];
		dj += dvoldt[conn3];

		edvoldt[index0] = dj / NODES_PER_ELEM;
    }
}

__global__ void update_stress_kernel(double *stress, double *strain_rate, double *edvoldt, double *strain, double *plstrain, double *delta_plstrain, double *dpressure, int nelem) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < nelem) {
		int index1 = index + nelem;
		int index2 = index + nelem * 2;

		double sr_xx = strain_rate[index];
		double sr_yy = strain_rate[index1];
		double sr_xy = strain_rate[index2];

		double e_edvoldt = edvoldt[index];

		// anti-mesh locking correction on strain rate
	    double div = sr_xx + sr_yy;
	    div = (e_edvoldt - div) / 2;

	    sr_xx += div;
	    sr_yy += div;

	    strain_rate[index]  = sr_xx;
		strain_rate[index1] = sr_yy;

	    //strain increment
	    double ds_xx = sr_xx * c_dt;
	    double ds_yy = sr_yy * c_dt;
	    double ds_xy = sr_xy * c_dt;

	    // update strain with strain rate
		strain[index]  += ds_xx;
		strain[index1] += ds_yy;
		strain[index2] += ds_xy;

        double e_plstrain = plstrain[index];

        double cohesion, phi, psi, hardn;
	    
        if (e_plstrain < c_pls0) {
            // no weakening yet
            cohesion = c_cohesion0;
            phi = c_friction_angle0;
            psi = c_dilation_angle0;
            hardn = 0;
        } else if (e_plstrain < c_pls1) {
            // linear weakening
            double p = (e_plstrain - c_pls0) / (c_pls1 - c_pls0);
            cohesion = (c_cohesion0 + p * (c_cohesion1 - c_cohesion0));
            phi = (c_friction_angle0 + p * (c_friction_angle1 - c_friction_angle0));
            psi = (c_dilation_angle0 + p * (c_dilation_angle1 - c_dilation_angle0));
            hardn = (c_cohesion1 - c_cohesion0) / (c_pls1 - c_pls0);
        }
        else {
            // saturated weakening
            cohesion = c_cohesion1;
            phi = c_friction_angle1;
            psi = c_dilation_angle1;
            hardn = 0;
        }

	    // derived variables
	    double sphi = sin(phi * DEG2RAD);
	    double spsi = sin(psi * DEG2RAD);

	    double anphi = (1 + sphi) / (1 - sphi);
	    double anpsi = (1 + spsi) / (1 - spsi);
	    double amc = 2 * cohesion * sqrt(anphi);

	    double ten_max = (phi == 0) ? c_tension_max : fmin(c_tension_max, cohesion / tan(phi * DEG2RAD));

	    /* increment the stress s according to the incremental strain de */
	    double lambda = c_bulkm - 2. / 3. * c_shearm;
	    double dev = ds_xx + ds_yy;

	    double st_xx = stress[index];
	    double st_yy = stress[index1];
	    double st_xy = stress[index2];

	    double old_s = st_xx + st_yy;

	    st_xx += 2 * c_shearm * ds_xx + lambda * dev;
	    st_yy += 2 * c_shearm * ds_yy + lambda * dev;
	    st_xy += 2 * c_shearm * ds_xy;

	    //
	    // transform to principal stress coordinate system
	    //
	    // eigenvalues (principal stresses)
	    double p0, p1;;
	    // In 2D, we only construct the eigenvectors from
	    // cos(2*theta) and sin(2*theta) of Mohr circle
	    double cos2t, sin2t;

	    // center and radius of Mohr circle
	    double s0 = 0.5 * (st_xx + st_yy);
	    double rad = sqrt(0.25 * (st_xx - st_yy) * (st_xx - st_yy) + st_xy * st_xy);

	    // principal stresses in the X-Z plane
	    p0 = s0 - rad;
	    p1 = s0 + rad;

	    if (rad > 1e-15) {
	    	cos2t = 0.5 * (st_yy - st_xx) / rad;
	    	sin2t = -st_xy / rad;
	    } else {
	    	cos2t = 1;
	    	sin2t = 0;
	    }

	    double fs = p0 - p1 * anphi + amc;
	    double ft = p1 - ten_max;

	    double depls = 0;

	    if (fs <= 0 || ft >= 0) {
	    	// yield, shear or tensile?
	    	double pa = sqrt(1 + anphi * anphi) + anphi;
	    	double ps = ten_max * anphi - amc;
		    double h  = p1 - ten_max + pa * (p0 - ps);
    		double a1 = c_bulkm + 4. / 3. * c_shearm;
    		double a2 = lambda;

    		double alam;
		    if (h < 0) {
		        // shear failure
		        alam = fs / (a1 - a2 * anpsi + a1 * anphi * anpsi - a2 * anphi + 2 * sqrt(anphi) * hardn);
		        p0 -= alam * (a1 - a2 * anpsi);
		        p1 -= alam * (a2 - a1 * anpsi);

		        // 2nd invariant of plastic strain
		        /* // plastic strain in the principle directions
		         * double depls1 = alam;
		         * double depls2 = -alam * anpsi;
		         * double deplsm = (depls1 + depls2) / 2;
		         * depls = std::sqrt(((depls1-deplsm)*(depls1-deplsm) +
		         *                    (depls2-deplsm)*(depls2-deplsm) +
		         *                    deplsm*deplsm) / 2);
		         */
		        // the equations above can be reduce to:
		        depls = fabs(alam) * sqrt((3 + 2 * anpsi + 3 * anpsi * anpsi) / 8);
		    } else {
		        // tensile failure
		        alam = ft / a1;
		        p0 -= alam * a2;
		        p1 -= alam * a1;

		        // 2nd invariant of plastic strain
		        /* double depls1 = 0;
		         * double depls3 = alam;
		         * double deplsm = (depls1 + depls3) / 2;
		         * depls = std::sqrt(((depls1-deplsm)*(depls1-deplsm) +
		         *                    (depls2-deplsm)*(depls2-deplsm) +
		         *                    deplsm*deplsm) / 2);
		         */
		        depls = fabs(alam) * sqrt(3. / 8.);
		    }

		    // rotate the principal stresses back to global axes
	        double dc2 = (p0 - p1) * cos2t;
	        double dss = p0 + p1;
	        st_xx = 0.5 * (dss + dc2);
	        st_yy = 0.5 * (dss - dc2);
	        st_xy = 0.5 * (p0 - p1) * sin2t;

	        plstrain[index] += depls;
	    }

	    delta_plstrain[index] = depls;

		stress[index]  = st_xx;
		stress[index1] = st_yy;
		stress[index2] = st_xy;

		dpressure[index] = st_xx + st_yy - old_s;
	}
}

__device__ int cuda_dsyevc3(double A[3][3], double w[3]) {
  double m, c1, c0;
  
  double de = A[0][1] * A[1][2];
  double dd = SQR(A[0][1]);
  double ee = SQR(A[1][2]);
  double ff = SQR(A[0][2]);
  m  = A[0][0] + A[1][1] + A[2][2];
  c1 = (A[0][0]*A[1][1] + A[0][0]*A[2][2] + A[1][1]*A[2][2]) - (dd + ee + ff);
  c0 = A[2][2]*dd + A[0][0]*ee + A[1][1]*ff - A[0][0]*A[1][1]*A[2][2] - 2.0 * A[0][2]*de;

  double p, sqrt_p, q, c, s, phi;
  p = SQR(m) - 3.0*c1;
  q = m*(p - (3.0/2.0)*c1) - (27.0/2.0)*c0;
  sqrt_p = sqrt(fabs(p));

  phi = 27.0 * ( 0.25*SQR(c1)*(p - c1) + c0*(q + 27.0/4.0*c0));
  phi = (1.0/3.0) * atan2(sqrt(fabs(phi)), q);
  
  c = sqrt_p*cos(phi);
  s = (1.0/M_SQRT3)*sqrt_p*sin(phi);

  w[1]  = (1.0/3.0)*(m - c);
  w[2]  = w[1] + s;
  w[0]  = w[1] + c;
  w[1] -= s;

  return 0;
}

__device__ int cuda_dsyevh3(double A[3][3], double Q[3][3], double w[3]) {
	double norm;          // Squared norm or inverse norm of current eigenvector
	double error;         // Estimated maximum roundoff error
	double t, u;          // Intermediate storage
	int j;                // Loop counter

	// Calculate eigenvalues
	cuda_dsyevc3(A, w);
  
	t = fabs(w[0]);
	if ((u=fabs(w[1])) > t) {
		t = u;
	}

	if ((u=fabs(w[2])) > t) {
		t = u;
	}

	if (t < 1.0) {
		u = t;
	} else {
		u = SQR(t);
	}

	error = 256.0 * DBL_EPSILON * SQR(u);

	Q[0][1] = A[0][1]*A[1][2] - A[0][2]*A[1][1];
	Q[1][1] = A[0][2]*A[0][1] - A[1][2]*A[0][0];
	Q[2][1] = SQR(A[0][1]);

	// Calculate first eigenvector by the formula
	Q[0][0] = Q[0][1] + A[0][2]*w[0];
	Q[1][0] = Q[1][1] + A[1][2]*w[0];
	Q[2][0] = (A[0][0] - w[0]) * (A[1][1] - w[0]) - Q[2][1];
	norm    = SQR(Q[0][0]) + SQR(Q[1][0]) + SQR(Q[2][0]);

  	if (norm <= error) {
    	return cuda_dsyevq3(A, Q, w);
  	} else {
    	norm = sqrt(1.0 / norm);
    	for (j=0; j < 3; j++) {
	    	Q[j][0] = Q[j][0] * norm;
	    }
    }
  
  	// Calculate second eigenvector by the formula
  	Q[0][1]  = Q[0][1] + A[0][2]*w[1];
  	Q[1][1]  = Q[1][1] + A[1][2]*w[1];
  	Q[2][1]  = (A[0][0] - w[1]) * (A[1][1] - w[1]) - Q[2][1];
  	norm     = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
  	if (norm <= error) {
    	return cuda_dsyevq3(A, Q, w);
  	} else {
    	norm = sqrt(1.0 / norm);
    	for (j=0; j < 3; j++) {
      		Q[j][1] = Q[j][1] * norm;
    	}
  	}
  
  	// Calculate third eigenvector according to
  	Q[0][2] = Q[1][0]*Q[2][1] - Q[2][0]*Q[1][1];
  	Q[1][2] = Q[2][0]*Q[0][1] - Q[0][0]*Q[2][1];
  	Q[2][2] = Q[0][0]*Q[1][1] - Q[1][0]*Q[0][1];

  	return 0;
}

__device__ int cuda_dsyevq3(double A[3][3], double Q[3][3], double w[3]) {
  	double e[3];                   // The third element is used only as temporary workspace
  	double g, r, p, f, b, s, c, t; // Intermediate storage
  	int nIter;
  	int m;

  	// Transform A to real tridiagonal form by the Householder method
  	cuda_dsytrd3(A, Q, w, e);
  
  	// Calculate eigensystem of the remaining real symmetric tridiagonal matrix
  	// with the QL method
  	//
  	// Loop over all off-diagonal elements
  	for (int l=0; l < 2; l++) {
    	nIter = 0;
    	while (1) {
      		// Check for convergence and exit iteration loop if off-diagonal
      		// element e(l) is zero
      		for (m=l; m <= 1; m++) {
        		g = fabs(w[m])+fabs(w[m+1]);
        		if (fabs(e[m]) + g == g) {
          			break;
        		}
      		}

      		if (m == l) {
        		break;
      		}
      
      		if (nIter++ >= 30) {
        		return -1;
      		}

      		// Calculate g = d_m - k
      		g = (w[l+1] - w[l]) / (e[l] + e[l]);
      		r = sqrt(SQR(g) + 1.0);
      		if (g > 0) {
        		g = w[m] - w[l] + e[l]/(g + r);
      		} else {
        		g = w[m] - w[l] + e[l]/(g - r);
      		}

      		s = c = 1.0;
      		p = 0.0;
      		for (int i=m-1; i >= l; i--) {
        		f = s * e[i];
        		b = c * e[i];
        		if (fabs(f) > fabs(g)) {
          			c      = g / f;
          			r      = sqrt(SQR(c) + 1.0);
          			e[i+1] = f * r;
          			c     *= (s = 1.0/r);
        		} else {
		          	s      = f / g;
		          	r      = sqrt(SQR(s) + 1.0);
		          	e[i+1] = g * r;
		          	s     *= (c = 1.0/r);
		        }
        
		        g = w[i+1] - p;
		        r = (w[i] - g)*s + 2.0*c*b;
		        p = s * r;
		        w[i+1] = g + p;
		        g = c*r - b;

		        // Form eigenvectors
		        for (int k=0; k < 3; k++) {
		          	t = Q[k][i+1];
		          	Q[k][i+1] = s*Q[k][i] + c*t;
		          	Q[k][i]   = c*Q[k][i] - s*t;
		        }
      		}

	        w[l] -= p;
	        e[l]  = g;
	        e[m]  = 0.0;
    	}
  	}

  	return 0;
}

__device__ void cuda_dsytrd3(double A[3][3], double Q[3][3], double d[3], double e[2]) {
 	double u[3], q[3];
	double omega, f;
  	double K, h, g;
  
	// Initialize Q to the identitity matrix
	for (int i=0; i < 3; i++) {
		Q[i][i] = 1.0;
		for (int j=0; j < i; j++) {
	  		Q[i][j] = Q[j][i] = 0.0;
		}
	}

  	// Bring first row and column to the desired form 
  	h = SQR(A[0][1]) + SQR(A[0][2]);
  	if (A[0][1] > 0) {
    	g = -sqrt(h);
  	} else {
    	g = sqrt(h);
  	}

	e[0] = g;
	f    = g * A[0][1];
	u[1] = A[0][1] - g;
  	u[2] = A[0][2];
  
  	omega = h - f;
  	if (omega > 0.0) {
	    omega = 1.0 / omega;
	    K     = 0.0;
	    for (int i=1; i < 3; i++) {
      		f    = A[1][i] * u[1] + A[i][2] * u[2];
      		q[i] = omega * f;                  // p
      		K   += u[i] * f;                   // u* A u
    	}
    	K *= 0.5 * SQR(omega);

    	for (int i=1; i < 3; i++)
    	  q[i] = q[i] - K * u[i];
    
	    d[0] = A[0][0];
	    d[1] = A[1][1] - 2.0*q[1]*u[1];
	    d[2] = A[2][2] - 2.0*q[2]*u[2];
    
	    // Store inverse Householder transformation in Q
	    for (int j=1; j < 3; j++) {
	      	f = omega * u[j];
	      	for (int i=1; i < 3; i++) {
	        	Q[i][j] = Q[i][j] - f*u[i];
	      	}
	    }

	    // Calculate updated A[1][2] and store it in e[1]
	    e[1] = A[1][2] - q[1]*u[2] - u[1]*q[2];
  	} else {
    	for (int i=0; i < 3; i++) {
      		d[i] = A[i][i];
    	}
    	e[1] = A[1][2];
  	}
}

__global__ void update_stress_3D_kernel(double *stress, double *strain_rate, double *edvoldt, double *strain, double *plstrain, double *delta_plstrain, double *dpressure, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index1 + nelem;
		int index3 = index2 + nelem;
		int index4 = index3 + nelem;
		int index5 = index4 + nelem;

		//tensor
	    double a[3][3];

		a[0][0] = strain_rate[index0];
		a[1][1] = strain_rate[index1];
		a[2][2] = strain_rate[index2];
		a[0][1] = strain_rate[index3];
		a[0][2] = strain_rate[index4];
		a[1][2] = strain_rate[index5];

		// anti-mesh locking correction on strain rate
	    double div = a[0][0] + a[1][1] + a[2][2];
	    div = (edvoldt[index0] - div) / NDIMS;

	    a[0][0] += div;
	    a[1][1] += div;
	    a[2][2] += div;

	    strain_rate[index0] = a[0][0];
		strain_rate[index1] = a[1][1];
		strain_rate[index2] = a[2][2];

	    //strain increment
	    double ds_xx = a[0][0] * c_dt;
	    double ds_yy = a[1][1] * c_dt;
	    double ds_zz = a[2][2] * c_dt;
	    double ds_xy = a[0][1] * c_dt;
	    double ds_xz = a[0][2] * c_dt;
	    double ds_yz = a[1][2] * c_dt;

	    // update strain with strain rate
		strain[index0] += ds_xx;
		strain[index1] += ds_yy;
		strain[index2] += ds_zz;
		strain[index3] += ds_xy;
		strain[index4] += ds_xz;
		strain[index5] += ds_yz;

        double e_plstrain = plstrain[index0];

        double cohesion, phi, psi, hardn;
	    
        if (e_plstrain < c_pls0) {
            // no weakening yet
            cohesion = c_cohesion0;
            phi = c_friction_angle0;
            psi = c_dilation_angle0;
            hardn = 0;
        } else if (e_plstrain < c_pls1) {
            // linear weakening
            double p = (e_plstrain - c_pls0) / (c_pls1 - c_pls0);
            cohesion = (c_cohesion0 + p * (c_cohesion1 - c_cohesion0));
            phi = (c_friction_angle0 + p * (c_friction_angle1 - c_friction_angle0));
            psi = (c_dilation_angle0 + p * (c_dilation_angle1 - c_dilation_angle0));
            hardn = (c_cohesion1 - c_cohesion0) / (c_pls1 - c_pls0);
        } else {
            // saturated weakening
            cohesion = c_cohesion1;
            phi = c_friction_angle1;
            psi = c_dilation_angle1;
            hardn = 0;
        }

	    // derived variables
	    double sphi = sin(phi * DEG2RAD);
	    double spsi = sin(psi * DEG2RAD);

	    double anphi = (1 + sphi) / (1 - sphi);
	    double anpsi = (1 + spsi) / (1 - spsi);
	    double amc = 2 * cohesion * sqrt(anphi);

	    double ten_max = (phi == 0) ? c_tension_max : fmin(c_tension_max, cohesion / tan(phi * DEG2RAD));

	    double lambda = c_bulkm - 2. / 3. * c_shearm;
	    double dev = ds_xx + ds_yy + ds_zz;

	    a[0][0] = stress[index0];
	    a[1][1] = stress[index1];
	    a[2][2] = stress[index2];
	    a[0][1] = stress[index3];
	    a[0][2] = stress[index4];
	    a[1][2] = stress[index5];

	    double old_s = a[0][0] + a[1][1] + a[2][2];

	    a[0][0] += 2 * c_shearm * ds_xx + lambda * dev;
	    a[1][1] += 2 * c_shearm * ds_yy + lambda * dev;
	    a[2][2] += 2 * c_shearm * ds_zz + lambda * dev;
	    a[0][1] += 2 * c_shearm * ds_xy;
	    a[0][2] += 2 * c_shearm * ds_xz;
	    a[1][2] += 2 * c_shearm * ds_yz;

	    // eigenvalues and eigenvectors
	    double p[3], v[3][3];

	    cuda_dsyevh3(a, v, p);

	    // reorder p and v
	    double tmp, b[3];
	    if (p[0] > p[1]) {
	        tmp = p[0];
	        p[0] = p[1];
	        p[1] = tmp;
	        for (int i=0; i<3; ++i)
	            b[i] = v[i][0];
	        for (int i=0; i<3; ++i)
	            v[i][0] = v[i][1];
	        for (int i=0; i<3; ++i)
	            v[i][1] = b[i];
	    }
	    if (p[1] > p[2]) {
	        tmp = p[1];
	        p[1] = p[2];
	        p[2] = tmp;
	        for (int i=0; i<3; ++i)
	            b[i] = v[i][1];
	        for (int i=0; i<3; ++i)
	            v[i][1] = v[i][2];
	        for (int i=0; i<3; ++i)
	            v[i][2] = b[i];
	    }
	    if (p[0] > p[1]) {
	        tmp = p[0];
	        p[0] = p[1];
	        p[1] = tmp;
	        for (int i=0; i<3; ++i)
	            b[i] = v[i][0];
	        for (int i=0; i<3; ++i)
	            v[i][0] = v[i][1];
	        for (int i=0; i<3; ++i)
	            v[i][1] = b[i];
	    }

	    double fs = p[0] - p[2] * anphi + amc;
	    double ft = p[2] - ten_max;

	    double depls = 0;

	    if (fs <= 0 || ft >= 0) {
	    	// yield, shear or tensile?
	    	double pa = sqrt(1 + anphi * anphi) + anphi;
	    	double ps = ten_max * anphi - amc;
		    double h  = p[2] - ten_max + pa * (p[0] - ps);
    		double a1 = c_bulkm + 4. / 3. * c_shearm;
    		double a2 = lambda;

    		double alam;
		    if (h < 0) {
		        // shear failure
		        alam = fs / (a1 - a2 * anpsi + a1 * anphi * anpsi - a2 * anphi + 2 * sqrt(anphi) * hardn);
		        p[0] -= alam * (a1 - a2 * anpsi);
		        p[1] -= alam * (a2 - a2 * anpsi);
		        p[2] -= alam * (a2 - a1 * anpsi);

		        depls = fabs(alam) * sqrt((7 + 4 * anpsi + 7 * anpsi * anpsi) / 18);
		    } else {
		        // tensile failure
		        alam = ft / a1;
		        p[0] -= alam * a2;
		        p[1] -= alam * a2;
		        p[2] -= alam * a1;

		        // 2nd invariant of plastic strain
		        depls = fabs(alam) * sqrt(7. / 18.);
		    }

		    // rotate the principal stresses back to global axes
		    a[0][0] = 0;
		    a[0][1] = 0;
	        a[0][2] = 0;
	        a[1][0] = 0;
	        a[1][1] = 0;
	        a[1][2] = 0;
	        a[2][0] = 0;
	        a[2][1] = 0;
	        a[2][2] = 0;
	        
	        for(int m=0; m<3; m++) {
	            for(int n=m; n<3; n++) {
	                for(int k=0; k<3; k++) {
	                    a[m][n] += v[m][k] * v[n][k] * p[k];
	                }
	            }
	        }

	        plstrain[index0] = e_plstrain + depls;
	    }

	    delta_plstrain[index0] = depls;

		stress[index0] = a[0][0];
		stress[index1] = a[1][1];
		stress[index2] = a[2][2];
		stress[index3] = a[0][1];
		stress[index4] = a[0][2];
		stress[index5] = a[1][2];

		dpressure[index0] = a[0][0] + a[1][1] + a[2][2] - old_s;
	}
}

__global__ void NMD_stress_kernel0(int *connectivity, double *dp_nd, double *dpressure, double *volume, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index0 + nelem * 2;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];

		double dp = dpressure[index0] * volume[index0];

		MyAtomicAdd_8(dp_nd + conn0, dp);
		MyAtomicAdd_8(dp_nd + conn1, dp);
		MyAtomicAdd_8(dp_nd + conn2, dp);
	}
}

__global__ void NMD_stress_3D_kernel0(int *connectivity, double *dp_nd, double *dpressure, double *volume, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index1 + nelem;
		int index3 = index2 + nelem;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];
		int conn3 = connectivity[index3];

		double dp = dpressure[index0] * volume[index0];

		MyAtomicAdd_8(dp_nd + conn0, dp);
		MyAtomicAdd_8(dp_nd + conn1, dp);
		MyAtomicAdd_8(dp_nd + conn2, dp);
		MyAtomicAdd_8(dp_nd + conn3, dp);
	}
}

__global__ void NMD_stress_kernel1(double *dp_nd, double *volume_n, int nnode) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nnode) {
		dp_nd[index0] /= volume_n[index0];
	}
}

__global__ void NMD_stress_3D_kernel1(double *dp_nd, double *volume_n, int nnode) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nnode) {
		dp_nd[index0] /= volume_n[index0];
	}
}

__global__ void NMD_stress_kernel2(int *connectivity, double *dp_nd, double *stress, double *dpressure, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index0 + nelem * 2;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];

		double dp = dp_nd[conn0];
		dp += dp_nd[conn1];
		dp += dp_nd[conn2];

		dp /= NODES_PER_ELEM;
		dp -= dpressure[index0];
		dp /= NDIMS;

		stress[index0] += dp;
		stress[index1] += dp;
	}
}

__global__ void NMD_stress_3D_kernel2(int *connectivity, double *dp_nd, double *stress, double *dpressure, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index1 + nelem;
		int index3 = index2 + nelem;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];
		int conn3 = connectivity[index3];

		double dp = dp_nd[conn0];
		dp += dp_nd[conn1];
		dp += dp_nd[conn2];
		dp += dp_nd[conn3];

		dp /= NODES_PER_ELEM;
		dp -= dpressure[index0];
		dp /= NDIMS;

		stress[index0] += dp;
		stress[index1] += dp;
		stress[index2] += dp;
	}
}

__global__ void update_force_kernel(int *connectivity, double *shpdx, double *shpdz, double *stress, double *volume, double *temperature, double *force, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index0 + nelem * 2;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];

		double shpdx0 = shpdx[index0];
		double shpdx1 = shpdx[index1];
		double shpdx2 = shpdx[index2];

		double shpdz0 = shpdz[index0];
		double shpdz1 = shpdz[index1];
		double shpdz2 = shpdz[index2];

		double s0 = stress[index0];
		double s1 = stress[index1];
		double s2 = stress[index2];

		double vol = volume[index0];

		double buoy = 0;

		// average temperature of this element
		/*
	    double T = 0;
	    T += temperature[conn0];
	    T += temperature[conn1];
	    T += temperature[conn2];
	    T /= NODES_PER_ELEM;
	    T -= c_celsius0;

	    double rho = c_rho * (1 - c_alpha * T);
	    */
	    double rho = c_rho;

		buoy = rho * c_gravity / NODES_PER_ELEM;

		double f00 = -(s0 * shpdx0 + s2 * shpdz0) * vol;
		double f10 = -(s0 * shpdx1 + s2 * shpdz1) * vol;
		double f20 = -(s0 * shpdx2 + s2 * shpdz2) * vol;
		double f01 = -(s2 * shpdx0 + s1 * shpdz0 + buoy) * vol;
		double f11 = -(s2 * shpdx1 + s1 * shpdz1 + buoy) * vol;
		double f21 = -(s2 * shpdx2 + s1 * shpdz2 + buoy) * vol;

		MyAtomicAdd_8(force + conn0 * 2, f00);
		MyAtomicAdd_8(force + conn0 * 2 + 1, f01);
		MyAtomicAdd_8(force + conn1 * 2, f10);
		MyAtomicAdd_8(force + conn1 * 2 + 1, f11);
		MyAtomicAdd_8(force + conn2 * 2, f20);
		MyAtomicAdd_8(force + conn2 * 2 + 1, f21);
	}
}

__global__ void update_force_3D_kernel(int *connectivity, double *shpdx, double *shpdy, double *shpdz, double *stress, double *volume, double *temperature, double *force, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index1 + nelem;
		int index3 = index2 + nelem;
		int index4 = index3 + nelem;
		int index5 = index4 + nelem;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];
		int conn3 = connectivity[index3];

		double shpdx0 = shpdx[index0];
		double shpdx1 = shpdx[index1];
		double shpdx2 = shpdx[index2];
		double shpdx3 = shpdx[index3];

		double shpdy0 = shpdy[index0];
		double shpdy1 = shpdy[index1];
		double shpdy2 = shpdy[index2];
		double shpdy3 = shpdy[index3];

		double shpdz0 = shpdz[index0];
		double shpdz1 = shpdz[index1];
		double shpdz2 = shpdz[index2];
		double shpdz3 = shpdz[index3];

		double s0 = stress[index0];
		double s1 = stress[index1];
		double s2 = stress[index2];
		double s3 = stress[index3];
		double s4 = stress[index4];
		double s5 = stress[index5];

		double vol = volume[index0];

		double buoy = 0;

		// average temperature of this element
		/*
	    double T = 0;
	    T += temperature[conn0];
	    T += temperature[conn1];
	    T += temperature[conn2];
	    T /= NODES_PER_ELEM;
	    T -= c_celsius0;

	    double rho = c_rho * (1 - c_alpha * T);
	    */
	    double rho = c_rho;

		buoy = rho * c_gravity / NODES_PER_ELEM;

		double f00 = -(s0 * shpdx0 + s3 * shpdy0 + s4 * shpdz0) * vol;
		double f10 = -(s0 * shpdx1 + s3 * shpdy1 + s4 * shpdz1) * vol;
		double f20 = -(s0 * shpdx2 + s3 * shpdy2 + s4 * shpdz2) * vol;
		double f30 = -(s0 * shpdx3 + s3 * shpdy3 + s4 * shpdz3) * vol;

		double f01 = -(s3 * shpdx0 + s1 * shpdy0 + s5 * shpdz0) * vol;
		double f11 = -(s3 * shpdx1 + s1 * shpdy1 + s5 * shpdz1) * vol;
		double f21 = -(s3 * shpdx2 + s1 * shpdy2 + s5 * shpdz2) * vol;
		double f31 = -(s3 * shpdx3 + s1 * shpdy3 + s5 * shpdz3) * vol;

		double f02 = -(s4 * shpdx0 + s5 * shpdy0 + s2 * shpdz0 + buoy) * vol;
		double f12 = -(s4 * shpdx1 + s5 * shpdy1 + s2 * shpdz1 + buoy) * vol;
		double f22 = -(s4 * shpdx2 + s5 * shpdy2 + s2 * shpdz2 + buoy) * vol;
		double f32 = -(s4 * shpdx3 + s5 * shpdy3 + s2 * shpdz3 + buoy) * vol;

		MyAtomicAdd_8(force + conn0 * 3, f00);
		MyAtomicAdd_8(force + conn0 * 3 + 1, f01);
		MyAtomicAdd_8(force + conn0 * 3 + 2, f02);
		MyAtomicAdd_8(force + conn1 * 3, f10);
		MyAtomicAdd_8(force + conn1 * 3 + 1, f11);
		MyAtomicAdd_8(force + conn1 * 3 + 2, f12);
		MyAtomicAdd_8(force + conn2 * 3, f20);
		MyAtomicAdd_8(force + conn2 * 3 + 1, f21);
		MyAtomicAdd_8(force + conn2 * 3 + 2, f22);
		MyAtomicAdd_8(force + conn3 * 3, f30);
		MyAtomicAdd_8(force + conn3 * 3 + 1, f31);
		MyAtomicAdd_8(force + conn3 * 3 + 2, f32);
	}
}

__global__ void surface_processes_kernel0(int *connectivity, int *eindex, int *findex, double *total_dx, double *total_slope, double *coord, int nsize, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nsize) {
		int e = eindex[index0];
		int f = findex[index0];

		int conn0 = connectivity[e];
		int conn1 = connectivity[e + nelem];
		int conn2 = connectivity[e + nelem * 2];

		int n0, n1;

		if (f == 0) {
			n0 = conn1;
			n1 = conn2;
		} else if (f == 1) {
			n0 = conn2;
			n1 = conn0;
		} else {
			n0 = conn0;
			n1 = conn1;
		}

		double x0 = coord[n0 * 2];
		double z0 = coord[n0 * 2 + 1];
		double x1 = coord[n1 * 2];
		double z1 = coord[n1 * 2 + 1];

		double dx = fabs(x1 - x0);
		MyAtomicAdd_8(total_dx + n0, dx);
		MyAtomicAdd_8(total_dx + n1, dx);

        double slope = -(z1 - z0) / dx;
        MyAtomicAdd_8(total_slope + n0, slope);
        slope *= -1;
        MyAtomicAdd_8(total_slope + n1, slope);
	}
}

__global__ void surface_processes_3D_kernel0(int *connectivity, int *eindex, int *findex, double *total_dx, double *total_slope, double *coord, int nsize, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nsize) {
		int e = eindex[index0];
		int f = findex[index0];

		int conn0 = connectivity[e];
		int conn1 = connectivity[e + nelem];
		int conn2 = connectivity[e + nelem * 2];
		int conn3 = connectivity[e + nelem * 3];

		int n0, n1, n2;

		if (f == 0) {
			n0 = conn1;
			n1 = conn2;
			n2 = conn3;
		} else if (f == 1) {
			n0 = conn0;
			n1 = conn3;
			n2 = conn2;
		} else if (f == 2) {
			n0 = conn0;
			n1 = conn1;
			n2 = conn3;
		} else {
			n0 = conn0;
			n1 = conn2;
			n2 = conn1;
		}

		double normal[3];

        double x01, y01, z01, x02, y02, z02;
        x01 = coord[n1 * 3] - coord[n0 * 3];
        y01 = coord[n1 * 3 + 1] - coord[n0 * 3 + 1];
        z01 = coord[n1 * 3 + 2] - coord[n0 * 3 + 2];
        x02 = coord[n2 * 3] - coord[n0 * 3];
        y02 = coord[n2 * 3 + 1] - coord[n0 * 3 + 1];
        z02 = coord[n2 * 3 + 2] - coord[n0 * 3 + 2];

        normal[0] = y01 * z02 - z01 * y02;
        normal[1] = z01 * x02 - x01 * z02;
        normal[2] = x01 * y02 - y01 * x02;

        double projected_area = 0.5 * normal[2];

		MyAtomicAdd_8(total_dx + n0, projected_area);
		MyAtomicAdd_8(total_dx + n1, projected_area);
		MyAtomicAdd_8(total_dx + n2, projected_area);

		double shp2dx[NODES_PER_FACET], shp2dy[NODES_PER_FACET];
        double iv = 0.5 / projected_area;
        shp2dx[0] = iv * (coord[n1 * 3 + 1] - coord[n2 * 3 + 1]);
        shp2dx[1] = iv * (coord[n2 * 3 + 1] - coord[n0 * 3 + 1]);
        shp2dx[2] = iv * (coord[n0 * 3 + 1] - coord[n1 * 3 + 1]);
        shp2dy[0] = iv * (coord[n2 * 3] - coord[n1 * 3]);
        shp2dy[1] = iv * (coord[n0 * 3] - coord[n2 * 3]);
        shp2dy[2] = iv * (coord[n1 * 3] - coord[n0 * 3]);

        double D[NODES_PER_FACET][NODES_PER_FACET];
        for (int j=0; j<NODES_PER_FACET; j++) {
            for (int k=0; k<NODES_PER_FACET; k++) {
                D[j][k] = (shp2dx[j] * shp2dx[k] +
                           shp2dy[j] * shp2dy[k]);
            }
        }

        const int n[3] = {n0, n1, n2};
        for (int j=0; j<NODES_PER_FACET; j++) {
            double slope = 0;
            for (int k=0; k<NODES_PER_FACET; k++)
                slope += D[j][k] * coord[n[k] * 3 + 2];

                MyAtomicAdd_8(total_slope + n[j], slope * projected_area);
        }
	}
}

__global__ void surface_processes_kernel1(int *nindex, double *total_dx, double *total_slope, double *coord, int nsize) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nsize) {
		int n = nindex[index0];
		double dh = c_surface_diffusivity * c_dt * total_slope[n] / total_dx[n];
		coord[n * 2 + 1] -= dh;
	}
}

__global__ void surface_processes_3D_kernel1(int *nindex, double *total_dx, double *total_slope, double *coord, int nsize) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nsize) {
		int n = nindex[index0];
		double dh = c_surface_diffusivity * c_dt * total_slope[n] / total_dx[n];
		coord[n * 3 + 2] -= dh;
	}
}

__global__ void update_velocity_kernel(double *mass, double *vel, double *force, int nnode) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nnode) {
		double f0 = force[index0 * 2];
		double f1 = force[index0 * 2 + 1];

		double m = mass[index0];

		vel[index0 * 2] += c_dt * f0 / m;
		vel[index0 * 2 + 1] += c_dt * f1 / m;
	}
}

__global__ void update_velocity_3D_kernel(double *mass, double *vel, double *force, int nnode) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nnode) {
		double f0 = force[index0 * 3];
		double f1 = force[index0 * 3 + 1];
		double f2 = force[index0 * 3 + 2];

		double m = mass[index0];

		vel[index0 * 3] += c_dt * f0 / m;
		vel[index0 * 3 + 1] += c_dt * f1 / m;
		vel[index0 * 3 + 2] += c_dt * f2 / m;
	}
}

__global__ void update_coordinate_kernel(double *coord, double *vel, int nnode) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nnode) {
		coord[index0 * 2]     += vel[index0 * 2] * c_dt;
		coord[index0 * 2 + 1] += vel[index0 * 2 + 1] * c_dt;
	}
}

__global__ void update_coordinate_3D_kernel(double *coord, double *vel, int nnode) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nnode) {
		coord[index0 * 3]     += vel[index0 * 3] * c_dt;
		coord[index0 * 3 + 1] += vel[index0 * 3 + 1] * c_dt;
		coord[index0 * 3 + 2] += vel[index0 * 3 + 2] * c_dt;
	}
}

__global__ void compute_volume_kernel(int *connectivity, double *coord, double *volume, double *shpdx, double *shpdz, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index0 + nelem * 2;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];

		double x0 = coord[conn0 * 2];
		double z0 = coord[conn0 * 2 + 1];
		double x1 = coord[conn1 * 2];
		double z1 = coord[conn1 * 2 + 1];
		double x2 = coord[conn2 * 2];
		double z2 = coord[conn2 * 2 + 1];

		double ab0 = x1 - x0;
		double ab1 = z1 - z0;
		double ac0 = x2 - x0;
		double ac1 = z2 - z0;

		double area = fabs(ab0 * ac1 - ab1 * ac0) / 2;

		volume[index0] = area;

		double iv = 1 / (2 * area);

        shpdx[index0] = iv * (z1 - z2);
        shpdx[index1] = iv * (z2 - z0);
        shpdx[index2] = iv * (z0 - z1);

        shpdz[index0] = iv * (x2 - x1);
        shpdz[index1] = iv * (x0 - x2);
        shpdz[index2] = iv * (x1 - x0);
	}
}

__global__ void compute_volume_3D_kernel(int *connectivity, double *coord, double *volume, double *shpdx, double *shpdy, double *shpdz, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index1 + nelem;
		int index3 = index2 + nelem;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];
		int conn3 = connectivity[index3];

		double x0 = coord[conn0 * 3];
		double y0 = coord[conn0 * 3 + 1];
		double z0 = coord[conn0 * 3 + 2];
		double x1 = coord[conn1 * 3];
		double y1 = coord[conn1 * 3 + 1];
		double z1 = coord[conn1 * 3 + 2];
		double x2 = coord[conn2 * 3];
		double y2 = coord[conn2 * 3 + 1];
		double z2 = coord[conn2 * 3 + 2];
		double x3 = coord[conn3 * 3];
		double y3 = coord[conn3 * 3 + 1];
		double z3 = coord[conn3 * 3 + 2];

		double x01 = x0 - x1;
		double x02 = x0 - x2;
		double x03 = x0 - x3;
		double x12 = x1 - x2;
		double x13 = x1 - x3;
		double x23 = x2 - x3;

		double y01 = y0 - y1;
		double y02 = y0 - y2;
		double y03 = y0 - y3;
		double y12 = y1 - y2;
		double y13 = y1 - y3;
		double y23 = y2 - y3;

		double z01 = z0 - z1;
		double z02 = z0 - z2;
		double z03 = z0 - z3;
		double z12 = z1 - z2;
		double z13 = z1 - z3;
		double z23 = z2 - z3;

	    double vol = (x01*(y23*z12 - y12*z23) + x12*(y01*z23 - y23*z01) + x23*(y12*z01 - y01*z12)) / 6;

		volume[index0] = vol;

		double iv = 1 / (6 * vol);

		shpdx[index0] = iv * (y13*z12 - y12*z13);
        shpdx[index1] = iv * (y02*z23 - y23*z02);
        shpdx[index2] = iv * (y13*z03 - y03*z13);
        shpdx[index3] = iv * (y01*z02 - y02*z01);

        shpdy[index0] = iv * (z13*x12 - z12*x13);
        shpdy[index1] = iv * (z02*x23 - z23*x02);
        shpdy[index2] = iv * (z13*x03 - z03*x13);
        shpdy[index3] = iv * (z01*x02 - z02*x01);

        shpdz[index0] = iv * (x13*y12 - x12*y13);
        shpdz[index1] = iv * (x02*y23 - x23*y02);
        shpdz[index2] = iv * (x13*y03 - x03*y13);
        shpdz[index3] = iv * (x01*y02 - x02*y01);
	}
}

__global__ void compute_mass_kernel(int *connectivity, double *volume, double *volume_n, double *mass, double *tmass, double *temperature, double pseudo_speed, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index0 + nelem * 2;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];

		double vol = volume[index0];
		double rho = c_bulkm / pseudo_speed / pseudo_speed;
		double m = rho * vol / NODES_PER_ELEM;

		// average temperature of this element
		/*
	    double T = 0;
	    T += temperature[conn0];
	    T += temperature[conn1];
	    T += temperature[conn2];
	    T /= NODES_PER_ELEM;
	    T -= c_celsius0;

	    double rho = c_rho * (1 - c_alpha * T);
	    */
	    rho = c_rho;

		double tm = rho * c_cp * vol / NODES_PER_ELEM;

		MyAtomicAdd_8(volume_n + conn0, vol);
		MyAtomicAdd_8(mass + conn0, m);
		MyAtomicAdd_8(tmass + conn0, tm);

		MyAtomicAdd_8(volume_n + conn1, vol);
		MyAtomicAdd_8(mass + conn1, m);
		MyAtomicAdd_8(tmass + conn1, tm);

		MyAtomicAdd_8(volume_n + conn2, vol);
		MyAtomicAdd_8(mass + conn2, m);
		MyAtomicAdd_8(tmass + conn2, tm);
	}
}

__global__ void compute_mass_3D_kernel(int *connectivity, double *volume, double *volume_n, double *mass, double *tmass, double *temperature, double pseudo_speed, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index1 + nelem;
		int index3 = index2 + nelem;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];
		int conn3 = connectivity[index3];

		double vol = volume[index0];
		double rho = c_bulkm / pseudo_speed / pseudo_speed;
		double m = rho * vol / NODES_PER_ELEM;

		// average temperature of this element
		/*
	    double T = 0;
	    T += temperature[conn0];
	    T += temperature[conn1];
	    T += temperature[conn2];
	    T /= NODES_PER_ELEM;
	    T -= c_celsius0;

	    double rho = c_rho * (1 - c_alpha * T);
	    */
	    rho = c_rho;

		double tm = rho * c_cp * vol / NODES_PER_ELEM;

		MyAtomicAdd_8(volume_n + conn0, vol);
		MyAtomicAdd_8(mass + conn0, m);
		MyAtomicAdd_8(tmass + conn0, tm);

		MyAtomicAdd_8(volume_n + conn1, vol);
		MyAtomicAdd_8(mass + conn1, m);
		MyAtomicAdd_8(tmass + conn1, tm);

		MyAtomicAdd_8(volume_n + conn2, vol);
		MyAtomicAdd_8(mass + conn2, m);
		MyAtomicAdd_8(tmass + conn2, tm);

		MyAtomicAdd_8(volume_n + conn3, vol);
		MyAtomicAdd_8(mass + conn3, m);
		MyAtomicAdd_8(tmass + conn3, tm);
	}
}

__global__ void rotate_stress_kernel(int *connectivity, double *strain, double *stress, double *shpdx, double *shpdz, double *vel, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index0 + nelem * 2;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];

		double shpdx0 = shpdx[index0];
		double shpdx1 = shpdx[index1];
		double shpdx2 = shpdx[index2];

		double shpdz0 = shpdz[index0];
		double shpdz1 = shpdz[index1];
		double shpdz2 = shpdz[index2];

		double v00 = vel[conn0 * 2];
		double v01 = vel[conn0 * 2 + 1];
		double v10 = vel[conn1 * 2];
		double v11 = vel[conn1 * 2 + 1];
		double v20 = vel[conn2 * 2];
		double v21 = vel[conn2 * 2 + 1];

		double w2 = v01 * shpdx0 + v11 * shpdx1 + v21 * shpdx2;
		w2 -= v00 * shpdz0 + v10 * shpdz1 + v20 * shpdz2;
		w2 /= 2;

		double strain0 = strain[index0];
		double strain1 = strain[index1];
		double strain2 = strain[index2];

		double sinc0 = -2. * strain2 * w2;
		double sinc1 =  2. * strain2 * w2;
		double sinc2 = (strain0 - strain1) * w2;

		strain[index0] += c_dt * sinc0;
		strain[index1] += c_dt * sinc1;
		strain[index2] += c_dt * sinc2;

		double stress0 = stress[index0];
		double stress1 = stress[index1];
		double stress2 = stress[index2];

		sinc0 = -2. * stress2 * w2;
		sinc1 =  2. * stress2 * w2;
		sinc2 = (stress0 - stress1) * w2;

		stress[index0] += c_dt * sinc0;
		stress[index1] += c_dt * sinc1;
		stress[index2] += c_dt * sinc2;
	}
}

__global__ void rotate_stress_3D_kernel(int *connectivity, double *strain, double *stress, double *shpdx, double *shpdy, double *shpdz, double *vel, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index1 + nelem;
		int index3 = index2 + nelem;
		int index4 = index3 + nelem;
		int index5 = index4 + nelem;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];
		int conn3 = connectivity[index3];

		double shpdx0 = shpdx[index0];
		double shpdx1 = shpdx[index1];
		double shpdx2 = shpdx[index2];
		double shpdx3 = shpdx[index3];

		double shpdy0 = shpdy[index0];
		double shpdy1 = shpdy[index1];
		double shpdy2 = shpdy[index2];
		double shpdy3 = shpdy[index3];

		double shpdz0 = shpdz[index0];
		double shpdz1 = shpdz[index1];
		double shpdz2 = shpdz[index2];
		double shpdz3 = shpdz[index3];

		double v00 = vel[conn0 * 3];
		double v01 = vel[conn0 * 3 + 1];
		double v02 = vel[conn0 * 3 + 2];

		double v10 = vel[conn1 * 3];
		double v11 = vel[conn1 * 3 + 1];
		double v12 = vel[conn1 * 3 + 2];

		double v20 = vel[conn2 * 3];
		double v21 = vel[conn2 * 3 + 1];
		double v22 = vel[conn2 * 3 + 2];

		double v30 = vel[conn3 * 3];
		double v31 = vel[conn3 * 3 + 1];
		double v32 = vel[conn3 * 3 + 2];

		double w3 = v00 * shpdy0 + v10 * shpdy1 + v20 * shpdy2 + v30 * shpdy3;
		w3 -= v01 * shpdx0 + v11 * shpdx1 + v21 * shpdx2 + v31 * shpdx3;
		w3 /= 2;

		double w4 = v00 * shpdz0 + v10 * shpdz1 + v20 * shpdz2 + v30 * shpdz3;
		w4 -= v02 * shpdx0 + v12 * shpdx1 + v22 * shpdx2 + v32 * shpdx3;
		w4 /= 2;

		double w5 = v01 * shpdz0 + v11 * shpdz1 + v21 * shpdz2 + v31 * shpdz3;
		w5 -= v02 * shpdy0 + v12 * shpdy1 + v22 * shpdy2 + v32 * shpdy3;
		w5 /= 2;

		double strain0 = strain[index0];
		double strain1 = strain[index1];
		double strain2 = strain[index2];
		double strain3 = strain[index3];
		double strain4 = strain[index4];
		double strain5 = strain[index5];

		double sinc0 = -2. * strain3 * w3 - 2. * strain4 * w4;
		double sinc1 =  2. * strain3 * w3 - 2. * strain5 * w5;
		double sinc2 =  2. * strain4 * w4 + 2. * strain5 * w5;
		double sinc3 = strain0 * w3 - strain1 * w3 - strain4 * w5 - strain5 * w4;
		double sinc4 = strain0 * w4 - strain2 * w4 + strain3 * w5 - strain5 * w3;
        double sinc5 = strain1 * w5 - strain2 * w5 + strain3 * w4 + strain4 * w3;

		strain[index0] = strain0 + c_dt * sinc0;
		strain[index1] = strain1 + c_dt * sinc1;
		strain[index2] = strain2 + c_dt * sinc2;
		strain[index3] = strain3 + c_dt * sinc3;
		strain[index4] = strain4 + c_dt * sinc4;
		strain[index5] = strain5 + c_dt * sinc5;

		strain0 = stress[index0];
		strain1 = stress[index1];
		strain2 = stress[index2];
		strain3 = stress[index3];
		strain4 = stress[index4];
		strain5 = stress[index5];

		sinc0 = -2. * strain3 * w3 - 2. * strain4 * w4;
		sinc1 =  2. * strain3 * w3 - 2. * strain5 * w5;
		sinc2 =  2. * strain4 * w4 + 2. * strain5 * w5;
		sinc3 = strain0 * w3 - strain1 * w3 - strain4 * w5 - strain5 * w4;
		sinc4 = strain0 * w4 - strain2 * w4 + strain3 * w5 - strain5 * w3;
        sinc5 = strain1 * w5 - strain2 * w5 + strain3 * w4 + strain4 * w3;

		stress[index0] = strain0 + c_dt * sinc0;
		stress[index1] = strain1 + c_dt * sinc1;
		stress[index2] = strain2 + c_dt * sinc2;
		stress[index3] = strain3 + c_dt * sinc3;
		stress[index4] = strain4 + c_dt * sinc4;
		stress[index5] = strain5 + c_dt * sinc5;
	}
}

__global__ void apply_damping_kernel(double *force, double *vel, int nnode) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nnode) {
		double v0 = vel[index0 * 2];
		double v1 = vel[index0 * 2 + 1];

		if (fabs(v0) > 1e-13) {
			force[index0 * 2] -= c_damping_factor * copysign(force[index0 * 2], v0);
		}

		if (fabs(v1) > 1e-13) {
			force[index0 * 2 + 1] -= c_damping_factor * copysign(force[index0 * 2 + 1], v1);
		}
	}
}

__global__ void apply_damping_3D_kernel(double *force, double *vel, int nnode) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nnode) {
		double v0 = vel[index0 * 3];
		double v1 = vel[index0 * 3 + 1];
		double v2 = vel[index0 * 3 + 2];

		if (fabs(v0) > 1e-13) {
			force[index0 * 3] -= c_damping_factor * copysign(force[index0 * 3], v0);
		}

		if (fabs(v1) > 1e-13) {
			force[index0 * 3 + 1] -= c_damping_factor * copysign(force[index0 * 3 + 1], v1);
		}

		if (fabs(v2) > 1e-13) {
			force[index0 * 3 + 2] -= c_damping_factor * copysign(force[index0 * 3 + 2], v2);
		}
	}
}

__global__ void apply_vbcs_kernel(unsigned int *bcflag, double *vel, int nnode) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nnode) {
		unsigned int flag = bcflag[index0];

		// is this a boundary node?
		if (flag & BOUND_ANY) {
			//
	        // X
	        //
	        if (flag & BOUNDX0) {
	            switch (c_vbc_x0) {
	            case 0:
	                break;
	            case 1:
	                vel[index0 * 2] = c_vbc_val_x0;
	                break;
	            case 2:
	                vel[index0 * 2 + 1] = 0;
	                break;
	            case 3:
	                vel[index0 * 2] = c_vbc_val_x0;
	                vel[index0 * 2 + 1] = 0;
	                break;
	            }
	        }
	        if (flag & BOUNDX1) {
	            switch (c_vbc_x1) {
	            case 0:
	                break;
	            case 1:
	                vel[index0 * 2] = c_vbc_val_x1;
	                break;
	            case 2:
	                vel[index0 * 2 + 1] = 0;
	                break;
	            case 3:
	                vel[index0 * 2] = c_vbc_val_x1;
	                vel[index0 * 2 + 1] = 0;
	                break;
	            }
	        }
		}
	}
}

__global__ void apply_vbcs_3D_kernel(unsigned int *bcflag, double *vel, int nnode) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nnode) {
		unsigned int flag = bcflag[index0];

		// is this a boundary node?
		if (flag & BOUND_ANY) {
			//
	        // X
	        //
	        if (flag & BOUNDX0) {
	            switch (c_vbc_x0) {
	            case 0:
	                break;
	            case 1:
	                vel[index0 * 3] = c_vbc_val_x0;
	                break;
	            case 2:
	                vel[index0 * 3 + 1] = 0;
	                vel[index0 * 3 + 2] = 0;
	                break;
	            case 3:
	                vel[index0 * 3] = c_vbc_val_x0;
	                vel[index0 * 3 + 1] = 0;
	                vel[index0 * 3 + 2] = 0;
	                break;
	            case 4:
	                vel[index0 * 3 + 1] = c_vbc_val_x0;
	                vel[index0 * 3 + 2] = 0;
	                break;
	            case 5:
	                vel[index0 * 3] = 0;
	                vel[index0 * 3 + 1] = c_vbc_val_x0;
	                vel[index0 * 3 + 2] = 0;
	                break;
	            case 7:
	                vel[index0 * 3] = c_vbc_val_x0;
	                vel[index0 * 3 + 1] = 0;
	                break;
	            }
	        }
	        if (flag & BOUNDX1) {
	            switch (c_vbc_x1) {
	            case 0:
	                break;
	            case 1:
	                vel[index0 * 3] = c_vbc_val_x1;
	                break;
	            case 2:
	                vel[index0 * 3 + 1] = 0;
	                vel[index0 * 3 + 2] = 0;
	                break;
	            case 3:
	                vel[index0 * 3] = c_vbc_val_x1;
	                vel[index0 * 3 + 1] = 0;
	                vel[index0 * 3 + 2] = 0;
	                break;
	            case 4:
	                vel[index0 * 3 + 1] = c_vbc_val_x1;
	                vel[index0 * 3 + 2] = 0;
	                break;
	            case 5:
	                vel[index0 * 3] = 0;
	                vel[index0 * 3 + 1] = c_vbc_val_x1;
	                vel[index0 * 3 + 2] = 0;
	                break;
	            case 7:
	                vel[index0 * 3] = c_vbc_val_x1;
	                vel[index0 * 3 + 1] = 0;
	                break;
	            }
	        }
		}

		//
        // Y
        //
        if (flag & BOUNDY0) {
            switch (c_vbc_y0) {
            case 0:
                break;
            case 1:
                vel[index0 * 3 + 1] = c_vbc_val_y0;
                break;
            case 2:
                vel[index0 * 3] = 0;
                vel[index0 * 3 + 2] = 0;
                break;
            case 3:
                vel[index0 * 3] = 0;
                vel[index0 * 3 + 1] = c_vbc_val_y0;
                vel[index0 * 3 + 2] = 0;
                break;
            case 4:
                vel[index0 * 3] = c_vbc_val_y0;
                vel[index0 * 3 + 2] = 0;
                break;
            case 5:
                vel[index0 * 3] = c_vbc_val_y0;
                vel[index0 * 3 + 1] = 0;
                vel[index0 * 3 + 2] = 0;
                break;
            case 7:
                vel[index0 * 3] = 0;
                vel[index0 * 3 + 1] = c_vbc_val_y0;
                break;
            }
        }
        if (flag & BOUNDY1) {
            switch (c_vbc_y1) {
            case 0:
                break;
            case 1:
                vel[index0 * 3 + 1] = c_vbc_val_y1;
                break;
            case 2:
                vel[index0 * 3] = 0;
                vel[index0 * 3 + 2] = 0;
                break;
            case 3:
                vel[index0 * 3] = c_vbc_val_y1;
                vel[index0 * 3 + 1] = 0;
                vel[index0 * 3 + 2] = 0;
                break;
            case 4:
                vel[index0 * 3] = c_vbc_val_y1;
                vel[index0 * 3 + 2] = 0;
                break;
            case 5:
                vel[index0 * 3] = c_vbc_val_y1;
                vel[index0 * 3 + 1] = 0;
                vel[index0 * 3 + 2] = 0;
                break;
            case 7:
                vel[index0 * 3] = 0;
                vel[index0 * 3 + 1] = c_vbc_val_y1;
                break;
            }
        }
	}
}

__global__ void apply_stress_bcs_kernel(double *coord, int *eindex, int *findex, int *connectivity, double *force, double *temperature, int nsize, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nsize) {
		int e = eindex[index0];
		int f = findex[index0];

		int conn0 = connectivity[e];
		int conn1 = connectivity[e + nelem];
		int conn2 = connectivity[e + nelem * 2];

		int n0, n1;

		if (f == 0) {
			n0 = conn1;
			n1 = conn2;
		} else if (f == 1) {
			n0 = conn2;
			n1 = conn0;
		} else {
			n0 = conn0;
			n1 = conn1;
		}

		double x0 = coord[n0 * 2];
		double z0 = coord[n0 * 2 + 1];
		double x1 = coord[n1 * 2];
		double z1 = coord[n1 * 2 + 1];

		double norm0 = z1 - z0;
		double norm1 = x0 - x1;

		double zcenter = (z0 + z1) / NODES_PER_FACET;

		// average temperature of this element
		/*
	    double T = 0;
	    T += temperature[conn0];
	    T += temperature[conn1];
	    T += temperature[conn2];
	    T /= NODES_PER_ELEM;
	    T -= c_celsius0;

	    double rho = c_rho * (1 - c_alpha * T);
	    */
	    double rho = c_rho;

		double p = c_compensation_pressure - rho * c_gravity * (zcenter + c_zlength);
		p /= -NODES_PER_FACET;

		MyAtomicAdd_8(force + n0 * 2, p * norm0);
		MyAtomicAdd_8(force + n0 * 2 + 1, p * norm1);
		MyAtomicAdd_8(force + n1 * 2, p * norm0);
		MyAtomicAdd_8(force + n1 * 2 + 1, p * norm1);
	}
}

__global__ void apply_stress_bcs_3D_kernel(double *coord, int *eindex, int *findex, int *connectivity, double *force, double *temperature, int nsize, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	if (index0 < nsize) {
		int e = eindex[index0];
		int f = findex[index0];

		int conn0 = connectivity[e];
		int conn1 = connectivity[e + nelem];
		int conn2 = connectivity[e + nelem * 2];
		int conn3 = connectivity[e + nelem * 3];

		int n0, n1, n2;

		if (f == 0) {
			n0 = conn1;
			n1 = conn2;
			n2 = conn3;
		} else if (f == 1) {
			n0 = conn0;
			n1 = conn3;
			n2 = conn2;
		} else if (f == 2) {
			n0 = conn0;
			n1 = conn1;
			n2 = conn3;
		} else {
			n0 = conn0;
			n1 = conn2;
			n2 = conn1;
		}

		double normal[3];

        double z0 = coord[n0 * 3 + 2];
        double z1 = coord[n1 * 3 + 2];
        double z2 = coord[n2 * 3 + 2];

        double x01 = coord[n1 * 3] - coord[n0 * 3];
        double y01 = coord[n1 * 3 + 1] - coord[n0 * 3 + 1];
        double z01 = z1 - z0;
        double x02 = coord[n2 * 3] - coord[n0 * 3];
        double y02 = coord[n2 * 3 + 1] - coord[n0 * 3 + 1];
        double z02 = z2 - z0;

        normal[0] = (y01 * z02 - z01 * y02) / 2;
        normal[1] = (z01 * x02 - x01 * z02) / 2;
        normal[2] = (x01 * y02 - y01 * x02) / 2;

		double zcenter = (z0 + z1 + z2) / NODES_PER_FACET;

		// average temperature of this element
		/*
	    double T = 0;
	    T += temperature[conn0];
	    T += temperature[conn1];
	    T += temperature[conn2];
	    T /= NODES_PER_ELEM;
	    T -= c_celsius0;

	    double rho = c_rho * (1 - c_alpha * T);
	    */
	    double rho = c_rho;

		double p = c_compensation_pressure - rho * c_gravity * (zcenter + c_zlength);
		p /= -NODES_PER_FACET;

		double p0 = p * normal[0];
		double p1 = p * normal[1];
		double p2 = p * normal[2];

		MyAtomicAdd_8(force + n0 * 3, p0);
		MyAtomicAdd_8(force + n0 * 3 + 1, p1);
		MyAtomicAdd_8(force + n0 * 3 + 2, p2);
		MyAtomicAdd_8(force + n1 * 3, p0);
		MyAtomicAdd_8(force + n1 * 3 + 1, p1);
		MyAtomicAdd_8(force + n1 * 3 + 2, p2);
		MyAtomicAdd_8(force + n2 * 3, p0);
		MyAtomicAdd_8(force + n2 * 3 + 1, p1);
		MyAtomicAdd_8(force + n2 * 3 + 2, p2);
	}
}

__global__ void get_minh_kernel(int *connectivity, double *coord, double *volume, double *global_minh, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	double minh = 1e100;
	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index0 + nelem * 2;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];

		double x0 = coord[conn0 * 2];
		double z0 = coord[conn0 * 2 + 1];
		double x1 = coord[conn1 * 2];
		double z1 = coord[conn1 * 2 + 1];
		double x2 = coord[conn2 * 2];
		double z2 = coord[conn2 * 2 + 1];

		double s01 = (x1 - x0) * (x1 - x0) + (z1 - z0) * (z1 - z0);
		double s02 = (x2 - x0) * (x2 - x0) + (z2 - z0) * (z2 - z0);
		double s12 = (x2 - x1) * (x2 - x1) + (z2 - z1) * (z2 - z1);

		minh = fmax(s01, s02);
		minh = fmax(minh, s12);
		minh = sqrt(minh);
		minh = 2 * volume[index0] / minh;
	}

	for (int i = 16; i; i >>= 1) {
		minh = fmin(__shfl_down_sync(FULL_MASK, minh, i), minh);
	}

	__shared__ double blockminh;
	blockminh = 1e101;
	__syncthreads();

	if (threadIdx.x % 32 == 0) {
		MyAtomicMin_8(&blockminh, minh);
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		MyAtomicMin_8(global_minh, blockminh);
	}
}

__device__ double triangle_area_3D(const double *a, const double *b, const double *c) {
    double ab0, ab1, ac0, ac1;

    // ab: vector from a to b
    ab0 = b[0] - a[0];
    ab1 = b[1] - a[1];
    // ac: vector from a to c
    ac0 = c[0] - a[0];
    ac1 = c[1] - a[1];

    double ab2, ac2;
    ab2 = b[2] - a[2];
    ac2 = c[2] - a[2];

    // vector components of ab x ac
    double d0, d1, d2;
    d0 = ab1*ac2 - ab2*ac1;
    d1 = ab2*ac0 - ab0*ac2;
    d2 = ab0*ac1 - ab1*ac0;

    return sqrt(d0*d0 + d1*d1 + d2*d2) / 2;
}

__global__ void get_minh_3D_kernel(int *connectivity, double *coord, double *volume, double *global_minh, int nelem) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	double minh = 1e100;
	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index1 + nelem;
		int index3 = index2 + nelem;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];
		int conn3 = connectivity[index3];

		double *a, *b, *c, *d;
		a = coord + conn0 * 3;
		b = coord + conn1 * 3;
		c = coord + conn2 * 3;
		d = coord + conn3 * 3;

		double area = triangle_area_3D(a, b, c);
		area = fmax(area, triangle_area_3D(a, b, d));
		area = fmax(area, triangle_area_3D(c, d, a));
		area = fmax(area, triangle_area_3D(c, d, b));

		minh = 3 * volume[index0] / area;
	}

	for (int i = 16; i; i >>= 1) {
		minh = fmin(__shfl_down_sync(FULL_MASK, minh, i), minh);
	}

	__shared__ double blockminh;
	blockminh = 1e101;
	__syncthreads();

	if (threadIdx.x % 32 == 0) {
		MyAtomicMin_8(&blockminh, minh);
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		MyAtomicMin_8(global_minh, blockminh);
	}
}

__global__ void worst_elem_quality_kernel(int *connectivity, double *coord, double *q, double *global_minv, int nelem) {
    int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	double quality = 1;
	double minv = 1e100;
	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index0 + nelem * 2;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];

		double x0 = coord[conn0 * 2];
		double z0 = coord[conn0 * 2 + 1];
		double x1 = coord[conn1 * 2];
		double z1 = coord[conn1 * 2 + 1];
		double x2 = coord[conn2 * 2];
		double z2 = coord[conn2 * 2 + 1];

		double normalization_factor = 4. * M_SQRT3;

		double ab0 = x1 - x0;
		double ab1 = z1 - z0;
		double ac0 = x2 - x0;
		double ac1 = z2 - z0;
		double bc0 = x2 - x1;
		double bc1 = z2 - z1;

		double dist2_sum = ab0 * ab0 + ab1 * ab1;
		dist2_sum += ac0 * ac0 + ac1 * ac1;
		dist2_sum += bc0 * bc0 + bc1 * bc1;

		minv = fabs(ab0 * ac1 - ab1 * ac0) / 2;

		quality = normalization_factor * minv / dist2_sum;
	}

	for (int i = 16; i; i >>= 1) {
		quality = fmin(__shfl_down_sync(FULL_MASK, quality, i), quality);
		minv = fmin(__shfl_down_sync(FULL_MASK, minv, i), minv);
	}

	__shared__ double blockminq;
	__shared__ double blockminv;
	blockminq = 1;
	blockminv = 1e101;
	__syncthreads();

	if (threadIdx.x % 32 == 0) {
		MyAtomicMin_8(&blockminq, quality);
		MyAtomicMin_8(&blockminv, minv);
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		MyAtomicMin_8(q, blockminq);
		MyAtomicMin_8(global_minv, blockminv);
	}
}

__global__ void worst_elem_quality_3D_kernel(int *connectivity, double *coord, double *q, double *global_minv, int nelem) {
    int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	double quality = 1;
	double minv = 1e100;
	if (index0 < nelem) {
		int index1 = index0 + nelem;
		int index2 = index1 + nelem;
		int index3 = index2 + nelem;
		int conn0 = connectivity[index0];
		int conn1 = connectivity[index1];
		int conn2 = connectivity[index2];
		int conn3 = connectivity[index3];

		double *a, *b, *c, *d;
		a = coord + conn0 * 3;
		b = coord + conn1 * 3;
		c = coord + conn2 * 3;
		d = coord + conn3 * 3;

		double area_sum = triangle_area_3D(a, b, c);
		area_sum += triangle_area_3D(a, b, d);
		area_sum += triangle_area_3D(c, d, a);
		area_sum += triangle_area_3D(c, d, b);

		double normalization_factor = 216. * M_SQRT3;

		double x01 = a[0] - b[0];
	    double x12 = b[0] - c[0];
	    double x23 = c[0] - d[0];

	    double y01 = a[1] - b[1];
	    double y12 = b[1] - c[1];
	    double y23 = c[1] - d[1];

	    double z01 = a[2] - b[2];
	    double z12 = b[2] - c[2];
	    double z23 = c[2] - d[2];

	    minv = (x01*(y23*z12 - y12*z23) +
	            x12*(y01*z23 - y23*z01) +
	            x23*(y12*z01 - y01*z12)) / 6;

		quality = normalization_factor * minv * minv / area_sum / area_sum / area_sum;
	}

	for (int i = 16; i; i >>= 1) {
		quality = fmin(__shfl_down_sync(FULL_MASK, quality, i), quality);
		minv = fmin(__shfl_down_sync(FULL_MASK, minv, i), minv);
	}

	__shared__ double blockminq;
	__shared__ double blockminv;
	blockminq = 1;
	blockminv = 1e101;
	__syncthreads();

	if (threadIdx.x % 32 == 0) {
		MyAtomicMin_8(&blockminq, quality);
		MyAtomicMin_8(&blockminv, minv);
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		MyAtomicMin_8(q, blockminq);
		MyAtomicMin_8(global_minv, blockminv);
	}
}

__global__ void get_max_bottom_kernel(unsigned int *bcflag, double *coord, double *global_maxd, int nnode, double bottom) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	double maxd = 0;
	if (index0 < nnode) {
		if (bcflag[index0] & BOUNDZ0) {
			maxd = fabs(coord[index0 * 2 + 1] - bottom);
		}
	}

	for (int i = 16; i; i >>= 1) {
		maxd = fmax(__shfl_down_sync(FULL_MASK, maxd, i), maxd);
	}

	__shared__ double blockmaxd;
	blockmaxd = 0;
	__syncthreads();

	if (threadIdx.x % 32 == 0) {
		MyAtomicMax_8(&blockmaxd, maxd);
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		MyAtomicMax_8(global_maxd, blockmaxd);
	}
}

__global__ void get_max_bottom_3D_kernel(unsigned int *bcflag, double *coord, double *global_maxd, int nnode, double bottom) {
	int index0 = blockIdx.x * blockDim.x + threadIdx.x;

	double maxd = 0;
	if (index0 < nnode) {
		if (bcflag[index0] & BOUNDZ0) {
			maxd = fabs(coord[index0 * 3 + 2] - bottom);
		}
	}

	for (int i = 16; i; i >>= 1) {
		maxd = fmax(__shfl_down_sync(FULL_MASK, maxd, i), maxd);
	}

	__shared__ double blockmaxd;
	blockmaxd = 0;
	__syncthreads();

	if (threadIdx.x % 32 == 0) {
		MyAtomicMax_8(&blockmaxd, maxd);
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		MyAtomicMax_8(global_maxd, blockmaxd);
	}
}
