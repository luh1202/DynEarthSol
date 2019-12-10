#include <iostream>
#include "constants.hpp"
#include "parameters.hpp"
#include "bc.hpp"
#include "matprops.hpp"
#include "utils.hpp"
#include "fields_cuda.hpp"
#include "cuda.hpp"

void allocate_gpu_variable(const Param &param, Variables& var) {
    int nmax = 400000;
    int emax = 2000000;
    int bsmax = 100000;

    if (launch_cudaMalloc((void**)&var.d_connectivity,   sizeof(int) * emax * NODES_PER_ELEM)    != 0 ||
        launch_cudaMalloc((void**)&var.d_bcflag,         sizeof(unsigned int) * nmax)            != 0 ||
        launch_cudaMalloc((void**)&var.d_shpdx,          sizeof(double) * emax * NODES_PER_ELEM) != 0 ||
        launch_cudaMalloc((void**)&var.d_shpdz,          sizeof(double) * emax * NODES_PER_ELEM) != 0 ||
        launch_cudaMalloc((void**)&var.d_strain_rate,    sizeof(double) * emax * NSTR)           != 0 ||
        launch_cudaMalloc((void**)&var.d_temperature,    sizeof(double) * nmax)                  != 0 ||
        launch_cudaMalloc((void**)&var.d_support,        sizeof(double) * nmax)                  != 0 ||
        launch_cudaMalloc((void**)&var.d_vel,            sizeof(double) * nmax * NDIMS)          != 0 ||
        launch_cudaMalloc((void**)&var.d_volume,         sizeof(double) * emax)                  != 0 ||
        launch_cudaMalloc((void**)&var.d_tmass,          sizeof(double) * nmax)                  != 0 ||
        launch_cudaMalloc((void**)&var.d_mass,           sizeof(double) * nmax)                  != 0 ||
        launch_cudaMalloc((void**)&var.d_force,          sizeof(double) * nmax * NDIMS)          != 0 ||
        launch_cudaMalloc((void**)&var.d_coord,          sizeof(double) * nmax * NDIMS)          != 0 ||
        launch_cudaMalloc((void**)&var.d_dvoldt,         sizeof(double) * nmax)                  != 0 ||
        launch_cudaMalloc((void**)&var.d_edvoldt,        sizeof(double) * emax)                  != 0 ||
        launch_cudaMalloc((void**)&var.d_volume_n,       sizeof(double) * nmax)                  != 0 ||
        launch_cudaMalloc((void**)&var.d_plstrain,       sizeof(double) * emax)                  != 0 ||
        launch_cudaMalloc((void**)&var.d_delta_plstrain, sizeof(double) * emax)                  != 0 ||
        launch_cudaMalloc((void**)&var.d_dpressure,      sizeof(double) * emax)                  != 0 ||
        launch_cudaMalloc((void**)&var.d_strain,         sizeof(double) * emax * NSTR)           != 0 ||
        launch_cudaMalloc((void**)&var.d_stress,         sizeof(double) * emax * NSTR)           != 0 ||
        launch_cudaMalloc((void**)&var.d_bdrye,          sizeof(double) * bsmax)                 != 0 ||
        launch_cudaMalloc((void**)&var.d_bdryf,          sizeof(double) * bsmax)                 != 0 ||
        launch_cudaMalloc((void**)&var.d_bc_top_e,       sizeof(double) * bsmax)                 != 0 ||
        launch_cudaMalloc((void**)&var.d_bc_top_f,       sizeof(double) * bsmax)                 != 0 ||
        launch_cudaMalloc((void**)&var.d_bc_top_n,       sizeof(double) * bsmax)                 != 0) {
        std::cerr << "Error: cannot allocate GPU memory\n";
        std::exit(1);
    }

#ifdef THREED
    if (launch_cudaMalloc((void**)&var.d_shpdy, sizeof(double) * emax * NODES_PER_ELEM) != 0) {
        std::cerr << "Error: cannot allocate GPU memory\n";
        std::exit(1);
    }
#endif

    if (launch_cudaMallocHost((void**)&var.h_int_tmp,    sizeof(int) * emax * NODES_PER_ELEM)  != 0 ||
        launch_cudaMallocHost((void**)&var.h_double_tmp, sizeof(double) * emax * NSTR)         != 0) {
        std::cerr << "Error: cannot allocated host memory\n";
        std::exit(1);
    }

    #pragma omp parallel for
    for (int e = 0; e < var.nelem; e++) {
        for (int i = 0; i < NODES_PER_ELEM; i++) {
            var.h_int_tmp[i * var.nelem + e] = (*var.connectivity)[e][i];
        }
    }

    if (launch_cudaMemcpyAsync(var.d_connectivity, var.h_int_tmp, var.nelem * NODES_PER_ELEM * sizeof(int), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for connectivity\n";
        std::exit(1);
    }

    #pragma omp parallel for
    for (int n = 0; n < var.nnode; n++) {
            var.h_int_tmp[n] = (*var.bcflag)[n];
    }

    if (launch_cudaMemcpyAsync(var.d_bcflag, var.h_int_tmp, var.nnode * sizeof(unsigned int), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for bcflag\n";
        std::exit(1);
    }

    int bs = static_cast<int>(var.bfacets[4].size());

    #pragma omp parallel for
    for (int e = 0; e < bs; e++) {
        var.h_int_tmp[e] = var.bfacets[4][e].first;
    }

    if (launch_cudaMemcpyAsync(var.d_bdrye, var.h_int_tmp, bs * sizeof(int), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for volume\n";
        std::exit(1);
    }

    #pragma omp parallel for
    for (int e = 0; e < bs; e++) {
        var.h_int_tmp[e] = var.bfacets[4][e].second;
    }

    if (launch_cudaMemcpyAsync(var.d_bdryf, var.h_int_tmp, bs * sizeof(int), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for volume\n";
        std::exit(1);
    }

    bs = static_cast<int>(var.bfacets[5].size());

    #pragma omp parallel for
    for (int e = 0; e < bs; e++) {
        var.h_int_tmp[e] = var.bfacets[5][e].first;
    }

    if (launch_cudaMemcpyAsync(var.d_bc_top_e, var.h_int_tmp, bs * sizeof(int), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for volume\n";
        std::exit(1);
    }

    #pragma omp parallel for
    for (int e = 0; e < bs; e++) {
        var.h_int_tmp[e] = var.bfacets[5][e].second;
    }

    if (launch_cudaMemcpyAsync(var.d_bc_top_f, var.h_int_tmp, bs * sizeof(int), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for volume\n";
        std::exit(1);
    }

    bs = static_cast<int>(var.bnodes[5].size());

    #pragma omp parallel for
    for (int e = 0; e < bs; e++) {
        var.h_int_tmp[e] = var.bnodes[5][e];
    }

    if (launch_cudaMemcpyAsync(var.d_bc_top_n, var.h_int_tmp, bs * sizeof(int), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for volume\n";
        std::exit(1);
    }

    #pragma omp parallel for
    for (int e = 0; e < var.nelem; e++) {
        for (int i = 0; i < NODES_PER_ELEM; i++) {
            var.h_double_tmp[i * var.nelem + e] = (*var.shpdx)[e][i];
        }
    }

    if (launch_cudaMemcpyAsync(var.d_shpdx, var.h_double_tmp, var.nelem * NODES_PER_ELEM * sizeof(double), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for shpdx\n";
        std::exit(1);
    }

#ifdef THREED
    #pragma omp parallel for
    for (int e = 0; e < var.nelem; e++) {
        for (int i = 0; i < NODES_PER_ELEM; i++) {
            var.h_double_tmp[i * var.nelem + e] = (*var.shpdy)[e][i];
        }
    }

    if (launch_cudaMemcpyAsync(var.d_shpdy, var.h_double_tmp, var.nelem * NODES_PER_ELEM * sizeof(double), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for shpdx\n";
        std::exit(1);
    }
#endif

    #pragma omp parallel for
    for (int e = 0; e < var.nelem; e++) {
        for (int i = 0; i < NODES_PER_ELEM; i++) {
            var.h_double_tmp[i * var.nelem + e] = (*var.shpdz)[e][i];
        }
    }

    if (launch_cudaMemcpyAsync(var.d_shpdz, var.h_double_tmp, var.nelem * NODES_PER_ELEM * sizeof(double), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for shpdz\n";
        std::exit(1);
    }

    #pragma omp parallel for
    for (int e = 0; e < var.nelem; e++) {
        for (int i = 0; i < NSTR; i++) {
            var.h_double_tmp[i * var.nelem + e] = (*var.strain)[e][i];
        }
    }

    if (launch_cudaMemcpyAsync(var.d_strain, var.h_double_tmp, var.nelem * NSTR * sizeof(double), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for strain\n";
        std::exit(1);
    }

    #pragma omp parallel for
    for (int e = 0; e < var.nelem; e++) {
        for (int i = 0; i < NSTR; i++) {
            var.h_double_tmp[i * var.nelem + e] = (*var.stress)[e][i];
        }
    }

    if (launch_cudaMemcpyAsync(var.d_stress, var.h_double_tmp, var.nelem * NSTR * sizeof(double), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for stress\n";
        std::exit(1);
    }

    #pragma omp parallel for
    for (int n = 0; n < var.nnode; n++) {
        var.h_double_tmp[n] = (*var.temperature)[n];
    }

    if (launch_cudaMemcpyAsync(var.d_temperature, var.h_double_tmp, var.nnode * sizeof(double), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for temperature\n";
        std::exit(1);
    }

    #pragma omp parallel for
    for (int n = 0; n < var.nnode; n++) {
        var.h_double_tmp[n] = (*var.tmass)[n];
    }

    if (launch_cudaMemcpyAsync(var.d_tmass, var.h_double_tmp, var.nnode * sizeof(double), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for tmass\n";
        std::exit(1);
    }

    #pragma omp parallel for
    for (int n = 0; n < var.nnode; n++) {
        var.h_double_tmp[n] = (*var.mass)[n];
    }

    if (launch_cudaMemcpyAsync(var.d_mass, var.h_double_tmp, var.nnode * sizeof(double), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for mass\n";
        std::exit(1);
    }

    #pragma omp parallel for
    for (int n = 0; n < var.nnode; n++) {
        var.h_double_tmp[n] = (*var.volume_n)[n];
    }

    if (launch_cudaMemcpyAsync(var.d_volume_n, var.h_double_tmp, var.nnode * sizeof(double), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for volume_n\n";
        std::exit(1);
    }

    #pragma omp parallel for
    for (int n = 0; n < var.nnode; n++) {
        for (int i = 0; i < NDIMS; i++) {
            var.h_double_tmp[n * NDIMS + i] = (*var.vel)[n][i];
        }
    }

    if (launch_cudaMemcpyAsync(var.d_vel, var.h_double_tmp, var.nnode * NDIMS * sizeof(double), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for vel\n";
        std::exit(1);
    }

    #pragma omp parallel for
    for (int n = 0; n < var.nnode; n++) {
        for (int i = 0; i < NDIMS; i++) {
            var.h_double_tmp[n * NDIMS + i] = (*var.coord)[n][i];
        }
    }

    if (launch_cudaMemcpyAsync(var.d_coord, var.h_double_tmp, var.nnode * NDIMS * sizeof(double), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for coord\n";
        std::exit(1);
    }

    #pragma omp parallel for
    for (int e = 0; e < var.nelem; e++) {
        var.h_double_tmp[e] = (*var.volume)[e];
    }

    if (launch_cudaMemcpyAsync(var.d_volume, var.h_double_tmp, var.nelem * sizeof(double), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for volume\n";
        std::exit(1);
    }

    #pragma omp parallel for
    for (int e = 0; e < var.nelem; e++) {
        var.h_double_tmp[e] = (*var.plstrain)[e];
    }

    if (launch_cudaMemcpyAsync(var.d_plstrain, var.h_double_tmp, var.nelem * sizeof(double), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for volume\n";
        std::exit(1);
    }

    #pragma omp parallel for
    for (int e = 0; e < var.nelem; e++) {
        var.h_double_tmp[e] = (*var.delta_plstrain)[e];
    }

    if (launch_cudaMemcpyAsync(var.d_delta_plstrain, var.h_double_tmp, var.nelem * sizeof(double), 0) != 0) {
        std::cerr << "Error: cannot copy from host to device for volume\n";
        std::exit(1);
    }

    if(launch_set_constant_parameters(var.dt, param.bc.surface_temperature) != 0) {
        std::cerr << "Error: cannot set gpu constant parameters\n";
        std::exit(1);
    }
}

void delete_gpu_variable(Variables& var) {
    launch_cudaFree(var.d_connectivity);
    launch_cudaFree(var.d_bcflag);
    launch_cudaFree(var.d_shpdx);
#ifdef THREED
    launch_cudaFree(var.d_shpdy);
#endif
    launch_cudaFree(var.d_shpdz);
    launch_cudaFree(var.d_temperature);
    launch_cudaFree(var.d_support);
    launch_cudaFree(var.d_volume);
    launch_cudaFree(var.d_tmass);
    launch_cudaFree(var.d_vel);
    launch_cudaFree(var.d_strain_rate);
    launch_cudaFree(var.d_dvoldt);
    launch_cudaFree(var.d_edvoldt);
    launch_cudaFree(var.d_volume_n);
    launch_cudaFree(var.d_plstrain);
    launch_cudaFree(var.d_delta_plstrain);
    launch_cudaFree(var.d_dpressure);
    launch_cudaFree(var.d_stress);
    launch_cudaFree(var.d_strain);
    launch_cudaFree(var.d_force);
    launch_cudaFree(var.d_mass);
    launch_cudaFree(var.d_coord);
    launch_cudaFree(var.d_bdrye);
    launch_cudaFree(var.d_bdryf);
    launch_cudaFree(var.d_bc_top_e);
    launch_cudaFree(var.d_bc_top_f);
    launch_cudaFree(var.d_bc_top_n);

    launch_cudaFreeHost(var.h_int_tmp);
    launch_cudaFreeHost(var.h_double_tmp);
}

