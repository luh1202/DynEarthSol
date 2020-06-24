#include <algorithm>  // For std::max_element
#include <cmath>
#include <cstdio>
#include <iterator>  // For std::distance
#include <iostream>

#ifdef USE_OMP
#include <omp.h>
#else
#include <ctime>
#endif

#include "constants.hpp"
#include "parameters.hpp"
#include "binaryio.hpp"
#include "geometry.hpp"
#include "markerset.hpp"
#include "matprops.hpp"
#include "output.hpp"

#ifdef WIN32
#ifdef _MSC_VER
#define snprintf _snprintf
#endif // _MSC_VER
namespace std { using ::snprintf; }
#endif // WIN32

Output::Output(const Param& param, double start_time, int start_frame) :
    modelname(param.sim.modelname),
    start_time(start_time),
    is_averaged(param.sim.is_outputting_averaged_fields),
    average_interval(param.mesh.quality_check_step_interval),
    has_marker_output(param.sim.has_marker_output),
    frame(start_frame),
    time0(0)
{}


Output::~Output()
{}


void Output::write_velocity(const Variables& var)
{
    int NTO = 10;//average_interval;//for Large Sumatra,300; for small, 250
    MarkerSet& ms = *(var.markersets[0]);
    const double* v = var.vel->data();
    const double* coord0 = var.coord->data();
    if (var.steps % NTO == 0) {
        int n = 4, m[n];
        // For 90km
        m[0] = 2087; // @ x_5e3 y_-9.5e3
        m[1] = 17520;//17454; // @ x_45e3 y_-10e3+
        m[2] = 32933; // @ x_85e3 y_-9.5e3
        m[3] = 17469; // @ x_45e3 y_-10e3-
        for (int k=0; k<n; ++k) {

            int e = ms.get_elem(m[k]);
            int *conn = (*var.connectivity)[e];
            double vm = 0, vx =0., vy = 0., vz = 0., center[NDIMS] = {0};
            ave_velocity(var, e, vm, vx, vy, vz);

            for (int i=0; i<NODES_PER_ELEM; ++i) {
                for (int j=0; j<NDIMS; ++j)
                    center[j] += (*var.coord)[conn[i]][j];
            }
            for (int d=0; d<NDIMS; ++d)
                center[d] /= NODES_PER_ELEM;

            double dt = var.dt;

            char bbuffer[256];
            double M_S = (*var.MAX_shear)[e];//p[0]
#ifdef RS
            double s1 = (*var.MAX_shear_0)[e];//p[2]//var.avg_shear_stress / var.slip_area;//(*var.state1)[e];
            double SV = (*var.slip_velocity)[e];
            double FC = (*var.friction_coefficient)[e];
            double R_S = var.seismic_eff; //(*var.RS_shear)[e];
            int failure = var.number_plf; //(*var.Failure_mode)[e];
            double pls = (*var.plstrain)[e];
            double cl = var.S_E+var.K_E;//var.avg_vm / var.slip_area;//(*var.CL)[e];
#endif
                        /*
#ifdef THREED
            std::snprintf(bbuffer, 255, "%10d\t%12.12e\t%12.6e\t%12.6e\t%12.6e\t%12.6e\t%12.6e\t%12.6e\t%12.6e\t%12.6e\t%12.6e\t%12.6e\t%3d\n",
                                  var.steps, var.time/YEAR2SEC, dt/YEAR2SEC, SV, s1,
                                  FC, center[0], center[1], center[2], M_S, R_S, vm, failure);
#else
            std::snprintf(bbuffer, 255, "%10d\t%12.12e\t%12.6e\t%12.6e\t%12.6e\t%12.6e\t%12.2e\t%12.6e\t%12.6e\t%12.6e\t%12.6e\t%12.6e\t%3d\n",
                                  var.steps, var.time/YEAR2SEC, dt/YEAR2SEC, SV, s1, FC, cl,
                                  center[0], center[1], M_S, R_S, vx, failure);
#endif
            */
            std::snprintf(bbuffer, 255, "%10d\t%12.12e\t%12.6e\t%12.6e\t%12.6e\t%12.6e\t%12.6e\t%12.6e\t%12.6e\t%12.6e\t%12.6e\t%12.6e\t%3d\n",
                                  var.steps, var.time/YEAR2SEC, dt/YEAR2SEC, R_S, center[0], center[NDIMS-1], M_S,
#ifdef RS
                                  s1, cl, FC, vz, vx, failure);
#else
                                  0., 0., 0., vz, vx, 0.);
#endif
            //int aa = 1e6, i0 = var.steps/aa;
            std::string ss = std::to_string(static_cast<long long>(m[k]));
            //std::string a = std::to_string(static_cast<long long>(i0));
            //a = std::string(4 - a.length(), '0') + a;
            //std::string name("tmp_result//" + ss + "_velocity_" + a +".dat");
            std::string name("tmp_result/" + ss + "_velocity.dat");
            std::FILE* d;
            if (frame == 0)
                d = std::fopen(name.c_str(), "w");
            else
                d = std::fopen(name.c_str(), "a");

            if (d == NULL) {
                std::cerr << "Error: cannot open file '" << name << "' for writing\n";
                std::exit(2);
            }

            if (std::fputs(bbuffer, d) == EOF) {
                std::cerr << "Error: failed writing to file '" << name << "'\n";
                std::cerr << "\tbbuffer written:\n";
                std::cerr << bbuffer << '\n';
                std::exit(2);
            }
            std::fclose(d);
        }
    }
}

void Output::ave_velocity(const Variables& var, int e, double &vm, double &vx, double &vy, double &vz)
{
    double v[NDIMS];
    const int *conn = (*var.connectivity)[e];
    for (int i=0; i<NODES_PER_ELEM; ++i) {
        for (int j=0; j<NDIMS; ++j) {
            v[j] += (*var.vel)[conn[i]][j];
        }
    }
    vx = v[0]/NODES_PER_ELEM;
    vz = v[NDIMS-1]/NODES_PER_ELEM;
#ifdef THREED
    vy = v[1]/NODES_PER_ELEM;
    vm = std::sqrt(std::pow(vx,2)+std::pow(vy,2)+std::pow(vz,2));
#else
    vm = std::sqrt(std::pow(vx,2)+std::pow(vz,2));
#endif
}



void Output::write_info(const Variables& var, double dt)
{
#ifdef USE_OMP
    double run_time = omp_get_wtime() - start_time;
#else
    double run_time = double(std::clock()) / CLOCKS_PER_SEC;
#endif

    char buffer[256];
    std::snprintf(buffer, 255, "%6d\t%10d\t%12.6e\t%12.4e\t%12.6e\t%8d\t%8d\t%8d\n",
                  frame, var.steps, var.time, dt, run_time,
                  var.nnode, var.nelem, var.nseg);

    std::string filename(modelname + ".info");
    std::FILE* f;
    if (frame == 0)
        f = std::fopen(filename.c_str(), "w");
    else
        f = std::fopen(filename.c_str(), "a");

    if (f == NULL) {
        std::cerr << "Error: cannot open file '" << filename << "' for writing\n";
        std::exit(2);
    }

    if (std::fputs(buffer, f) == EOF) {
        std::cerr << "Error: failed writing to file '" << filename << "'\n";
        std::cerr << "\tbuffer written:\n";
        std::cerr << buffer << '\n';
        std::exit(2);
    }

    std::fclose(f);
}


void Output::write(const Variables& var, bool is_averaged)
{
    double dt = var.dt;
    double inv_dt = 1 / var.dt;
    if (average_interval && is_averaged) {
        dt = (var.time - time0) / average_interval;
        inv_dt = 1.0 / (var.time - time0);
    }
    write_info(var, dt);

    char filename[256];
    std::snprintf(filename, 255, "%s.save.%06d", modelname.c_str(), frame);
    BinaryOutput bin(filename);

    bin.write_array(*var.coord, "coordinate", var.coord->size());
    bin.write_array(*var.connectivity, "connectivity", var.connectivity->size());

    bin.write_array(*var.vel, "velocity", var.vel->size());
    if (average_interval && is_averaged) {
        // average_velocity = displacement / delta_t
        double *c0 = coord0.data();
        const double *c = var.coord->data();
        for (int i=0; i<coord0.num_elements(); ++i) {
            c0[i] = (c[i] - c0[i]) * inv_dt;
        }
        bin.write_array(coord0, "velocity averaged", coord0.size());
    }

    bin.write_array(*var.temperature, "temperature", var.temperature->size());

    bin.write_array(*var.plstrain, "plastic strain", var.plstrain->size());
#ifdef RS
    bin.write_array(*var.state1, "state1", var.state1->size());
    bin.write_array(*var.slip_velocity, "slip_velocity", var.slip_velocity->size());
    bin.write_array(*var.CL, "CL", var.CL->size());
#endif

    // Strain rate and plastic strain rate do not need to be checkpointed,
    // so we don't have to distinguish averged/non-averaged variants.
    double_vec *delta_plstrain = var.delta_plstrain;
    if (average_interval && is_averaged) {
        // average_strain_rate = delta_strain / delta_t
        delta_plstrain = &delta_plstrain_avg;
    }
    for (std::size_t i=0; i<delta_plstrain_avg.size(); ++i) {
        delta_plstrain_avg[i] *= inv_dt;
    }
    bin.write_array(*delta_plstrain, "plastic strain-rate", delta_plstrain->size());

    tensor_t *strain_rate = var.strain_rate;
    if (average_interval && is_averaged) {
        // average_strain_rate = delta_strain / delta_t
        strain_rate = &strain0;
        double *s0 = strain0.data();
        const double *s = var.strain->data();
        for (int i=0; i<strain0.num_elements(); ++i) {
            s0[i] = (s[i] - s0[i]) * inv_dt;
        }
    }
    bin.write_array(*strain_rate, "strain-rate", strain_rate->size());

    bin.write_array(*var.strain, "strain", var.strain->size());
    bin.write_array(*var.stress, "stress", var.stress->size());

    if (average_interval && is_averaged) {
        double *s = stress_avg.data();
        double tmp = 1.0 / (average_interval + 1);
        for (int i=0; i<stress_avg.num_elements(); ++i) {
            s[i] *= tmp;
        }
        bin.write_array(stress_avg, "stress averaged", stress_avg.size());
    }

    double_vec tmp(var.nelem);
    for (int e=0; e<var.nelem; ++e) {
        tmp[e] = var.mat->rho(e);
    }
    bin.write_array(tmp, "density", tmp.size());

    for (int e=0; e<var.nelem; ++e) {
        tmp[e] = elem_quality(*var.coord, *var.connectivity, *var.volume, e);
    }
    bin.write_array(tmp, "mesh quality", tmp.size());

    for (int e=0; e<var.nelem; ++e) {
        tmp[e] = var.mat->visc(e);
    }
    bin.write_array(tmp, "viscosity", tmp.size());
    // bin.write_array(*var.mass, "mass", var.mass->size());
    // bin.write_array(*var.tmass, "tmass", var.tmass->size());
    // bin.write_array(*var.volume_n, "volume_n", var.volume_n->size());
    // bin.write_array(*var.volume, "volume", var.volume->size());
    // bin.write_array(*var.edvoldt, "edvoldt", var.edvoldt->size());

    for (int e=0; e<var.nelem; ++e) {
        // Find the most abundant marker mattype in this element
        int_vec &a = (*var.elemmarkers)[e];
        tmp[e] = std::distance(a.begin(), std::max_element(a.begin(), a.end()));
    }
    bin.write_array(tmp, "material", tmp.size());

    bin.write_array(*var.force, "force", var.force->size());

    bin.write_array(*var.coord0, "coord0", var.coord0->size());

    bin.write_array(*var.bcflag, "bcflag", var.bcflag->size());

    if (has_marker_output) {
        for (auto ms=var.markersets.begin(); ms!=var.markersets.end(); ++ms)
            (*ms)->write_save_file(var, bin);
    }

    bin.close();
    std::cout << "  Output # " << frame
              << ", step = " << var.steps
              << ", time = " << var.time / YEAR2SEC << " yr"
#ifdef ATS
              << ", hmin = " << var.hmin << " m"
              << ", dt_elastic = " << var.dt_elastic/YEAR2SEC << " yr"
              << ", vmax = " << var.vmax << " m/s"
#endif
              << ", dt = " << dt / YEAR2SEC << " yr.\n";

    frame ++;

    {
        // check for NaN in coordinate
        for (int i=0; i<var.nnode; i++)
            for (int j=0; j<NDIMS; j++) {
                if (std::isnan((*var.coord)[i][j])) {
                    std::cerr << "Error: coordinate becomes NaN\n";
                    std::exit(11);
                }
                if (std::isinf((*var.coord)[i][j])) {
                    std::cerr << "Error: coordinate becomes Infinity\n";
                    std::exit(11);
                }
            }
    }
}


void Output::average_fields(Variables& var)
{
    // In the first time step of each interval, some old fields are
    // stored for later use.
    // It is guaranteed that remeshing won't occur within the interval.
    if (var.steps % average_interval == 1) {
        time0 = var.time;

        if (coord0.size() != var.coord->size()) {
            double *tmp = new double[(var.coord)->num_elements()];
            coord0.reset(tmp, var.coord->size());
        }
        std::copy(var.coord->begin(), var.coord->end(), coord0.begin());

        if (strain0.size() != var.strain->size()) {
            double *tmp = new double[(var.strain)->num_elements()];
            strain0.reset(tmp, var.strain->size());
        }
        std::copy(var.strain->begin(), var.strain->end(), strain0.begin());

        if (stress_avg.size() != var.stress->size()) {
            double *tmp = new double[(var.stress)->num_elements()];
            stress_avg.reset(tmp, var.stress->size());
        }
        std::copy(var.stress->begin(), var.stress->end(), stress_avg.begin());

        delta_plstrain_avg = *var.delta_plstrain;
    }
    else {
        // Averaging stress & plastic strain
        // (PS: dt-weighted average would be better, but difficult to do)
        double *s_avg = stress_avg.data();
        const double *s = var.stress->data();
        for (int i=0; i<stress_avg.num_elements(); ++i) {
            s_avg[i] += s[i];
        }
        for (std::size_t i=0; i<delta_plstrain_avg.size(); ++i) {
            delta_plstrain_avg[i] += (*var.delta_plstrain)[i];
        }
    }
}


void Output::write_checkpoint(const Param& param, const Variables& var)
{
    char filename[256];
    std::snprintf(filename, 255, "%s.chkpt.%06d", modelname.c_str(), frame);
    BinaryOutput bin(filename);

    double_vec tmp(2);
    tmp[0] = var.time;
    tmp[1] = var.compensation_pressure;
    bin.write_array(tmp, "time compensation_pressure", tmp.size());

    bin.write_array(*var.segment, "segment", var.segment->size());
    bin.write_array(*var.segflag, "segflag", var.segflag->size());
    // Note: regattr is not needed for restarting
    // bin.write_array(*var.regattr, "regattr", var.regattr->size());

    bin.write_array(*var.volume_old, "volume_old", var.volume_old->size());
    if (param.mat.is_plane_strain)
        bin.write_array(*var.stressyy, "stressyy", var.stressyy->size());

    for (auto ms=var.markersets.begin(); ms!=var.markersets.end(); ++ms)
        (*ms)->write_chkpt_file(bin);
}

