#include <cmath>
#include <iostream>

#include "3x3-C/dsyevh3.h"

#include "constants.hpp"
#include "parameters.hpp"
#include "matprops.hpp"
#include "rheology.hpp"
#include "utils.hpp"


static void principal_stresses3(const double* s, double p[3], double v[3][3])
{
    /* s is a flattened stress vector, with the components {XX, YY, ZZ, XY, XZ, YZ}.
     * Returns the eigenvalues p and eignvectors v.
     * The eigenvalues are ordered such that p[0] <= p[1] <= p[2].
     */

    // unflatten s to a 3x3 tensor, only the upper part is needed.
    double a[3][3];
    a[0][0] = s[0];
    a[1][1] = s[1];
    a[2][2] = s[2];
    a[0][1] = s[3];
    a[0][2] = s[4];
    a[1][2] = s[5];

    dsyevh3(a, v, p);

    // reorder p and v
    if (p[0] > p[1]) {
        double tmp, b[3];
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
        double tmp, b[3];
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
        double tmp, b[3];
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
}


static void principal_stresses2(const double* s, double p[2],
                                double& cos2t, double& sin2t)
{
    /* 's' is a flattened stress vector, with the components {XX, ZZ, XZ}.
     * Returns the eigenvalues 'p', and the direction cosine of the
     * eigenvectors in the X-Z plane.
     * The eigenvalues are ordered such that p[0] <= p[1].
     */

    // center and radius of Mohr circle
    double s0 = 0.5 * (s[0] + s[1]);
    double rad = second_invariant(s);

    // principal stresses in the X-Z plane
    p[0] = s0 - rad;
    p[1] = s0 + rad;

    {
        // direction cosine and sine of 2*theta
        const double eps = 1e-15;
        double a = 0.5 * (s[0] - s[1]);
        double b = - rad; // always negative
        if (b < -eps) {
            cos2t = a / b;
            sin2t = s[2] / b;
        }
        else {
            cos2t = 1;
            sin2t = 0;
        }
    }
}

static void sort_principal_stresses_with_syy(double p[3], const double syy, const double p_tmp[2])
{
    if (syy > p_tmp[1]) {
        p[0] = p_tmp[0];
        p[1] = p_tmp[1];
        p[2] = syy;
    }
    else if (syy < p_tmp[0]) {
        p[0] = syy;
        p[1] = p_tmp[0];
        p[2] = p_tmp[1];
    }
    else {
        p[0] = p_tmp[0];
        p[1] = syy;
        p[2] = p_tmp[1];
    }

}

static void elastic(double bulkm, double shearm, const double* de, double* s)
{
    /* increment the stress s according to the incremental strain de */
    double lambda = bulkm - 2. /3 * shearm;
    double dev = trace(de);

    for (int i=0; i<NDIMS; ++i)
        s[i] += 2 * shearm * de[i] + lambda * dev;
    for (int i=NDIMS; i<NSTR; ++i)
        s[i] += 2 * shearm * de[i];
}


static void maxwell(double bulkm, double shearm, double viscosity, double dt,
                    double dv, const double* de, double* s)
{
    // non-dimensional parameter: dt/ relaxation time
    double tmp = 0.5 * dt * shearm / viscosity;
    double f1 = 1 - tmp;
    double f2 = 1 / (1  + tmp);

    double dev = trace(de) / NDIMS;
    double s0 = trace(s) / NDIMS;

    // convert back to total stress
    for (int i=0; i<NDIMS; ++i)
        s[i] = ((s[i] - s0) * f1 + 2 * shearm * (de[i] - dev)) * f2 + s0 + bulkm * dv;
    for (int i=NDIMS; i<NSTR; ++i)
        s[i] = (s[i] * f1 + 2 * shearm * de[i]) * f2;
}


static void viscous(double bulkm, double viscosity, double total_dv,
                    const double* edot, double* s)
{
    /* Viscous Model + incompressibility enforced by bulk modulus */

    double dev = trace(edot) / NDIMS;

    for (int i=0; i<NDIMS; ++i)
        s[i] = 2 * viscosity * (edot[i] - dev) + bulkm * total_dv;
    for (int i=NDIMS; i<NSTR; ++i)
        s[i] = 2 * viscosity * edot[i];
}


static void elasto_plastic(double bulkm, double shearm,
                           double amc, double anphi, double anpsi,
                           double hardn, double ten_max,
                           const double* de, double& depls, double* s,
                           int &failure_mode)
{
    /* Elasto-plasticity (Mohr-Coulomb criterion)
     *
     * failure_mode --
     *   0: no failure
     *   1: tensile failure
     *  10: shear failure
     */

    // elastic trial stress
    elastic(bulkm, shearm, de, s);
    depls = 0;
    failure_mode = 0;

    //
    // transform to principal stress coordinate system
    //
    // eigenvalues (principal stresses)
    double p[NDIMS];
#ifdef THREED
    // eigenvectors
    double v[3][3];
    principal_stresses3(s, p, v);
#else
    // In 2D, we only construct the eigenvectors from
    // cos(2*theta) and sin(2*theta) of Mohr circle
    double cos2t, sin2t;
    principal_stresses2(s, p, cos2t, sin2t);
#endif

    // composite (shear and tensile) yield criterion
    double fs = p[0] - p[NDIMS-1] * anphi + amc;
    double ft = p[NDIMS-1] - ten_max;

    if (fs > 0 && ft < 0) {
        // no failure
        return;
    }

    // yield, shear or tensile?
    double pa = std::sqrt(1 + anphi*anphi) + anphi;
    double ps = ten_max * anphi - amc;
    double h = p[NDIMS-1] - ten_max + pa * (p[0] - ps);
    double a1 = bulkm + 4. / 3 * shearm;
    double a2 = bulkm - 2. / 3 * shearm;

    double alam;
    if (h < 0) {
        // shear failure
        failure_mode = 10;

        alam = fs / (a1 - a2*anpsi + a1*anphi*anpsi - a2*anphi + 2*std::sqrt(anphi)*hardn);
        p[0] -= alam * (a1 - a2 * anpsi);
#ifdef THREED
        p[1] -= alam * (a2 - a2 * anpsi);
#endif
        p[NDIMS-1] -= alam * (a2 - a1 * anpsi);

        // 2nd invariant of plastic strain
#ifdef THREED
        /* // plastic strain in the principle directions, depls2 is always 0
         * double depls1 = alam;
         * double depls3 = -alam * anpsi;
         * double deplsm = (depls1 + depls3) / 3;
         * depls = std::sqrt(((depls1-deplsm)*(depls1-deplsm) +
         *                    (-deplsm)*(-deplsm) +
         *                    (depls3-deplsm)*(depls3-deplsm) +
         *                    deplsm*deplsm) / 2);
         */
        // the equations above can be reduce to:
        depls = std::fabs(alam) * std::sqrt((7 + 4*anpsi + 7*anpsi*anpsi) / 18);
#else
        /* // plastic strain in the principle directions
         * double depls1 = alam;
         * double depls2 = -alam * anpsi;
         * double deplsm = (depls1 + depls2) / 2;
         * depls = std::sqrt(((depls1-deplsm)*(depls1-deplsm) +
         *                    (depls2-deplsm)*(depls2-deplsm) +
         *                    deplsm*deplsm) / 2);
         */
        // the equations above can be reduce to:
        depls = std::fabs(alam) * std::sqrt((3 + 2*anpsi + 3*anpsi*anpsi) / 8);
#endif
    }
    else {
        // tensile failure
        failure_mode = 1;

        alam = ft / a1;
        p[0] -= alam * a2;
#ifdef THREED
        p[1] -= alam * a2;
#endif
        p[NDIMS-1] -= alam * a1;

        // 2nd invariant of plastic strain
#ifdef THREED
        /* double depls1 = 0;
         * double depls3 = alam;
         * double deplsm = (depls1 + depls3) / 3;
         * depls = std::sqrt(((depls1-deplsm)*(depls1-deplsm) +
         *                    (-deplsm)*(-deplsm) +
         *                    (depls3-deplsm)*(depls3-deplsm) +
         *                    deplsm*deplsm) / 2);
         */
        depls = std::fabs(alam) * std::sqrt(7. / 18);
#else
        /* double depls1 = 0;
         * double depls3 = alam;
         * double deplsm = (depls1 + depls3) / 2;
         * depls = std::sqrt(((depls1-deplsm)*(depls1-deplsm) +
         *                    (depls2-deplsm)*(depls2-deplsm) +
         *                    deplsm*deplsm) / 2);
         */
        depls = std::fabs(alam) * std::sqrt(3. / 8);
#endif
    }

    // rotate the principal stresses back to global axes
    {
#ifdef THREED
        double ss[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
        for(int m=0; m<3; m++) {
            for(int n=m; n<3; n++) {
                for(int k=0; k<3; k++) {
                    ss[m][n] += v[m][k] * v[n][k] * p[k];
                }
            }
        }
        s[0] = ss[0][0];
        s[1] = ss[1][1];
        s[2] = ss[2][2];
        s[3] = ss[0][1];
        s[4] = ss[0][2];
        s[5] = ss[1][2];
#else
        double dc2 = (p[0] - p[1]) * cos2t;
        double dss = p[0] + p[1];
        s[0] = 0.5 * (dss + dc2);
        s[1] = 0.5 * (dss - dc2);
        s[2] = 0.5 * (p[0] - p[1]) * sin2t;
#endif
    }
}


static void elasto_plastic2d(double bulkm, double shearm,
                             double amc, double anphi, double anpsi,
                             double hardn, double ten_max,
                             const double* de, double& depls,
                             double* s, double &syy,
                             int &failure_mode)
{
    /* Elasto-plasticity (Mohr-Coulomb criterion) */

    /* This function is derived from geoFLAC.
     * The original code in geoFLAC assumes 2D plane strain formulation,
     * i.e. there are 3 principal stresses (PSs) and only 2 principal strains
     * (Strain_yy, Strain_xy, and Strain_yz all must be 0).
     * Here, the code is adopted to pure 2D or 3D plane strain.
     *
     * failure_mode --
     *   0: no failure
     *   1: tensile failure, all PSs exceed tensile limit
     *   2: tensile failure, 2 PSs exceed tensile limit
     *   3: tensile failure, 1 PS exceeds tensile limit
     *  10: pure shear failure
     *  11, 12, 13: tensile + shear failure
     *  20, 21, 22, 23: shear + tensile failure
     */

    depls = 0;
    failure_mode = 0;

    // elastic trial stress
    double a1 = bulkm + 4. / 3 * shearm;
    double a2 = bulkm - 2. / 3 * shearm;
    double sxx = s[0] + de[1]*a2 + de[0]*a1;
    double szz = s[1] + de[0]*a2 + de[1]*a1;
    double sxz = s[2] + de[2]*2*shearm;
    syy += (de[0] + de[1]) * a2; // Stress YY component, plane strain


    //
    // transform to principal stress coordinate system
    //
    // eigenvalues (principal stresses)
    double p[3];

    // In 2D, we only construct the eigenvectors from
    // cos(2*theta) and sin(2*theta) of Mohr circle
    double cos2t, sin2t;
    int n1, n2, n3;

    {
        // center and radius of Mohr circle
        double s0 = 0.5 * (sxx + szz);
        double rad = 0.5 * std::sqrt((sxx-szz)*(sxx-szz) + 4*sxz*sxz);

        // principal stresses in the X-Z plane
        double si = s0 - rad;
        double sii = s0 + rad;

        // direction cosine and sine of 2*theta
        const double eps = 1e-15;
        if (rad > eps) {
            cos2t = 0.5 * (szz - sxx) / rad;
            sin2t = -sxz / rad;
        }
        else {
            cos2t = 1;
            sin2t = 0;
        }

        // sort p.s.
#if 1
        //
        // 3d plane strain
        //
        if (syy > sii) {
            // syy is minor p.s.
            n1 = 0;
            n2 = 1;
            n3 = 2;
            p[0] = si;
            p[1] = sii;
            p[2] = syy;
        }
        else if (syy < si) {
            // syy is major p.s.
            n1 = 1;
            n2 = 2;
            n3 = 0;
            p[0] = syy;
            p[1] = si;
            p[2] = sii;
        }
        else {
            // syy is intermediate
            n1 = 0;
            n2 = 2;
            n3 = 1;
            p[0] = si;
            p[1] = syy;
            p[2] = sii;
        }
#else
        /* XXX: This case gives unreasonable result. Don't know why... */

        //
        // pure 2d case
        //
        n1 = 0;
        n2 = 2;
        p[0] = si;
        p[1] = syy;
        p[2] = sii;
#endif
    }

    // Possible tensile failure scenarios
    // 1. S1 (least tensional or greatest compressional principal stress) > ten_max:
    //    all three principal stresses must be greater than ten_max.
    //    Assign ten_max to all three and don't do anything further.
    if( p[0] >= ten_max ) {
        s[0] = s[1] = syy = ten_max;
        s[2] = 0.0;
        failure_mode = 1;
        return;
    }

    // 2. S2 (intermediate principal stress) > ten_max:
    //    S2 and S3 must be greater than ten_max.
    //    Assign ten_max to these two and continue to the shear failure block.
    if( p[1] >= ten_max ) {
        p[1] = p[2] = ten_max;
        failure_mode = 2;
    }

    // 3. S3 (most tensional or least compressional principal stress) > ten_max:
    //    Only this must be greater than ten_max.
    //    Assign ten_max to S3 and continue to the shear failure block.
    else if( p[2] >= ten_max ) {
        p[2] = ten_max;
        failure_mode = 3;
    }


    // shear yield criterion
    double fs = p[0] - p[2] * anphi + amc;
    if (fs >= 0.0) {
        // Tensile failure case S2 or S3 could have happened!!
        // XXX: Need to rationalize why exit without doing anything further.
        s[0] = sxx;
        s[1] = szz;
        s[2] = sxz;
        return;
    }

    failure_mode += 10;

    // shear failure
    const double alams = fs / (a1 - a2*anpsi + a1*anphi*anpsi - a2*anphi + hardn);
    p[0] -= alams * (a1 - a2 * anpsi);
    p[1] -= alams * (a2 - a2 * anpsi);
    p[2] -= alams * (a2 - a1 * anpsi);

    // 2nd invariant of plastic strain
    depls = 0.5 * std::fabs(alams + alams * anpsi);

    //***********************************
    // The following seems redundant but... this is how it goes in geoFLAC.
    //
    // Possible tensile failure scenarios
    // 1. S1 (least tensional or greatest compressional principal stress) > ten_max:
    //    all three principal stresses must be greater than ten_max.
    //    Assign ten_max to all three and don't do anything further.
    if( p[0] >= ten_max ) {
        s[0] = s[1] = syy = ten_max;
        s[2] = 0.0;
        failure_mode += 20;
        return;
    }

    // 2. S2 (intermediate principal stress) > ten_max:
    //    S2 and S3 must be greater than ten_max.
    //    Assign ten_max to these two and continue to the shear failure block.
    if( p[1] >= ten_max ) {
        p[1] = p[2] = ten_max;
        failure_mode += 20;
    }

    // 3. S3 (most tensional or least compressional principal stress) > ten_max:
    //    Only this must be greater than ten_max.
    //    Assign ten_max to S3 and continue to the shear failure block.
    else if( p[2] >= ten_max ) {
        p[2] = ten_max;
        failure_mode += 20;
    }
    //***********************************


    // rotate the principal stresses back to global axes
    {
        double dc2 = (p[n1] - p[n2]) * cos2t;
        double dss = p[n1] + p[n2];
        s[0] = 0.5 * (dss + dc2);
        s[1] = 0.5 * (dss - dc2);
        s[2] = 0.5 * (p[n1] - p[n2]) * sin2t;
        syy = p[n3];
    }
}

static void elasto_plastic_rs(double bulkm, double shearm,
                              double amc, double anphi, double anpsi,
                              double hardn, double ten_max,
                              const double* de, double pore_pressure_factor, 
                              double &depls, double* s,
                              int &failure_mode)
{
    /* Elasto-plasticity (Mohr-Coulomb criterion)
     *
     * failure_mode --
     *   0: no failure
     *   1: tensile failure
     *  10: shear failure
     */

    // elastic trial stress
    elastic(bulkm, shearm, de, s);
    depls = 0;
    failure_mode = 0;

    //
    // transform to principal stress coordinate system
    //
    // eigenvalues (principal stresses)
    double p[NDIMS];
#ifdef THREED
    // eigenvectors
    double v[3][3];
    principal_stresses3(s, p, v);
#else
    // In 2D, we only construct the eigenvectors from
    // cos(2*theta) and sin(2*theta) of Mohr circle
    double cos2t, sin2t;
    principal_stresses2(s, p, cos2t, sin2t);
#endif

    // composite (shear and tensile) yield criterion
    double fs = p[0] - p[NDIMS-1] * anphi + amc;
    double ft = p[NDIMS-1] - ten_max;

    if (fs > 0 && ft < 0) {
        // no failure
        return;
    }

    // yield, shear or tensile?
    double pa = std::sqrt(1 + anphi*anphi) + anphi;
    double ps = ten_max * anphi - amc;
    double h = p[NDIMS-1] - ten_max + pa * (p[0] - ps);
    double a1 = bulkm + 4. / 3 * shearm;
    double a2 = bulkm - 2. / 3 * shearm;

    double alam;
    if (h < 0) {
        // shear failure
        failure_mode = 10;

        alam = fs / (a1 - a2*anpsi + a1*anphi*anpsi - a2*anphi + 2*std::sqrt(anphi)*hardn);
        p[0] -= alam * (a1 - a2 * anpsi);
#ifdef THREED
        p[1] -= alam * (a2 - a2 * anpsi);
#endif
        p[NDIMS-1] -= alam * (a2 - a1 * anpsi);

        // 2nd invariant of plastic strain
#ifdef THREED
        /* // plastic strain in the principle directions, depls2 is always 0
         * double depls1 = alam;
         * double depls3 = -alam * anpsi;
         * double deplsm = (depls1 + depls3) / 3;
         * depls = std::sqrt(((depls1-deplsm)*(depls1-deplsm) +
         *                    (-deplsm)*(-deplsm) +
         *                    (depls3-deplsm)*(depls3-deplsm) +
         *                    deplsm*deplsm) / 2);
         */
        // the equations above can be reduce to:
        depls = std::fabs(alam) * std::sqrt((7 + 4*anpsi + 7*anpsi*anpsi) / 18);
#else
        /* // plastic strain in the principle directions
         * double depls1 = alam;
         * double depls2 = -alam * anpsi;
         * double deplsm = (depls1 + depls2) / 2;
         * depls = std::sqrt(((depls1-deplsm)*(depls1-deplsm) +
         *                    (depls2-deplsm)*(depls2-deplsm) +
         *                    deplsm*deplsm) / 2);
         */
        // the equations above can be reduce to:
        depls = std::fabs(alam) * std::sqrt((3 + 2*anpsi + 3*anpsi*anpsi) / 8);
#endif
    }
    else {
        // tensile failure
        failure_mode = 1;

        alam = ft / a1;
        p[0] -= alam * a2;
#ifdef THREED
        p[1] -= alam * a2;
#endif
        p[NDIMS-1] -= alam * a1;

        // 2nd invariant of plastic strain
#ifdef THREED
        /* double depls1 = 0;
         * double depls3 = alam;
         * double deplsm = (depls1 + depls3) / 3;
         * depls = std::sqrt(((depls1-deplsm)*(depls1-deplsm) +
         *                    (-deplsm)*(-deplsm) +
         *                    (depls3-deplsm)*(depls3-deplsm) +
         *                    deplsm*deplsm) / 2);
         */
        depls = std::fabs(alam) * std::sqrt(7. / 18);
#else
        /* double depls1 = 0;
         * double depls3 = alam;
         * double deplsm = (depls1 + depls3) / 2;
         * depls = std::sqrt(((depls1-deplsm)*(depls1-deplsm) +
         *                    (depls2-deplsm)*(depls2-deplsm) +
         *                    deplsm*deplsm) / 2);
         */
        depls = std::fabs(alam) * std::sqrt(3. / 8);
#endif
    }

    // rotate the principal stresses back to global axes
    {
#ifdef THREED
        double ss[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
        for(int m=0; m<3; m++) {
            for(int n=m; n<3; n++) {
                for(int k=0; k<3; k++) {
                    ss[m][n] += v[m][k] * v[n][k] * p[k];
                }
            }
        }
        s[0] = ss[0][0];
        s[1] = ss[1][1];
        s[2] = ss[2][2];
        s[3] = ss[0][1];
        s[4] = ss[0][2];
        s[5] = ss[1][2];
#else
        double dc2 = (p[0] - p[1]) * cos2t;
        double dss = p[0] + p[1];
        s[0] = 0.5 * (dss + dc2);
        s[1] = 0.5 * (dss - dc2);
        s[2] = 0.5 * (p[0] - p[1]) * sin2t;
#endif
    }
}


static void elasto_plastic2d_rs(double bulkm, double shearm,
                                double amc, double anphi, double anpsi,
                                double hardn, double ten_max,
                                const double* de, double pore_pressure_factor, 
                                double& depls,
                                double* s, double &syy,
                                int &failure_mode,
                                double &sig1,
                                double &sig3,
                                double &I2pp_diff)
{
    /* Elasto-plasticity (Mohr-Coulomb criterion) for rate-state friction. */

    /* This function is derived from geoFLAC.
     * The original code in geoFLAC assumes 2D plane strain formulation,
     * i.e. there are 3 principal stresses (PSs) and only 2 principal strains
     * (Strain_yy, Strain_xy, and Strain_yz all must be 0).
     * Here, the code is adopted to pure 2D or 3D plane strain.
     *
     * failure_mode --
     *   0: no failure
     *   1: tensile failure, all PSs exceed tensile limit
     *   2: tensile failure, 2 PSs exceed tensile limit
     *   3: tensile failure, 1 PS exceeds tensile limit
     *  10: pure shear failure
     *  11, 12, 13: tensile + shear failure
     *  20, 21, 22, 23: shear + tensile failure
     */
    double E = 9*bulkm*shearm/(3*bulkm+shearm);
    double nu = (3*bulkm-2*shearm)/(6*bulkm+2*shearm);

    I2pp = 0.0; // in case not updated below.
    I2pp_diff = 0.0;
    sig1 = 0.0;
    sig3 = 0.0;
    depls = 0.0;
    failure_mode = 0;

    // elastic trial stress
    double a1 = bulkm + 4. / 3 * shearm;
    double a2 = bulkm - 2. / 3 * shearm;
    double sxx = s[0] + de[1]*a2 + de[0]*a1;
    double szz = s[1] + de[0]*a2 + de[1]*a1;
    double sxz = s[2] + de[2]*2*shearm;
    syy += (de[0] + de[1]) * a2; // Stress YY component, plane strain

    //
    // transform to principal stress coordinate system
    //
    // eigenvalues (principal stresses)
    double p[NSTR];
    double p_reduced[NSTR]; // pore pressure-reduced principal stresses 

    // In 2D, we only construct the eigenvectors from
    // cos(2*theta) and sin(2*theta) of Mohr circle
    double cos2t, sin2t;
    int n1, n2, n3;

    {
        // center and radius of Mohr circle
        double s0 = 0.5 * (sxx + szz);
        double rad = 0.5 * std::sqrt((sxx-szz)*(sxx-szz) + 4*sxz*sxz);

        // principal stresses in the X-Z plane
        double si = s0 - rad;
        double sii = s0 + rad;

        // direction cosine and sine of 2*theta
        const double eps = 1e-15;
        if (rad > eps) {
            cos2t = 0.5 * (szz - sxx) / rad;
            sin2t = -sxz / rad;
        }
        else {
            cos2t = 1;
            sin2t = 0;
        }

        // sort p.s.
#if 1
        //
        // 3d plane strain
        //
        if (syy > sii) {
            // syy is minor p.s.
            n1 = 0;
            n2 = 1;
            n3 = 2;
            p[0] = si;
            p[1] = sii;
            p[2] = syy;
        }
        else if (syy < si) {
            // syy is major p.s.
            n1 = 1;
            n2 = 2;
            n3 = 0;
            p[0] = syy;
            p[1] = si;
            p[2] = sii;
        }
        else {
            // syy is intermediate
            n1 = 0;
            n2 = 2;
            n3 = 1;
            p[0] = si;
            p[1] = syy;
            p[2] = sii;
        }
#else
        /* XXX: This case gives unreasonable result. Don't know why... */

        //
        // pure 2d case
        //
        n1 = 0;
        n2 = 2;
        p[0] = si;
        p[1] = syy;
        p[2] = sii;
#endif
    }
    // pore pressure for shear zone material/s
    double pp = 0.0;
	for (int i=0; i<NSTR; i++)
	    pp += p[i]
	pp *= (pore_pressure_factor / 3.0);
    //ten_max += pp;

    // Principal stresses reduced by pore pressure
    for (int i=0; i<3; i++)
        p_reduced[i] = p[i] - pp;
    I2pp = I2_principal(E, nu, p_reduced);

    // Possible tensile failure scenarios
    // 1. S1 (least tensional or greatest compressional principal stress) > ten_max:
    //    all three principal stresses must be greater than ten_max.
    //    Assign ten_max to all three and don't do anything further.
    if( p[0] >= ten_max ) {
        s[0] = s[1] = syy = ten_max;
        s[2] = 0.0;
        failure_mode = 1;
    
        double tmp[NSTR];
        // save reduced principal stresses
        sig1 = sig3 = (ten_max - pp); 
        tmp[0] = tmp[1] = tmp[2] = sig1;
        double I2pp_new = I2_principal(E, nu, tmp);
        // save the difference between I2 of reduced principal stresses before and after yielding.
	    I2pp_diff = I2pp - I2pp_new;
        return;
    }

    // 2. S2 (intermediate principal stress) > ten_max:
    //    S2 and S3 must be greater than ten_max.
    //    Assign ten_max to these two and continue to the shear failure block.
    if( p[1] >= ten_max ) {
        p[1] = p[2] = ten_max;
        failure_mode = 2;
    }

    // 3. S3 (most tensional or least compressional principal stress) > ten_max:
    //    Only this must be greater than ten_max.
    //    Assign ten_max to S3 and continue to the shear failure block.
    else if( p[2] >= ten_max ) {
        p[2] = ten_max;
        failure_mode = 3;
    }

    // shear yield criterion
    
    double fs = p_reduced[0] - p_reduced[2] * anphi + amc;
    if (fs >= 0.0) {
        // No shear failure. Save the elastic guess and return.
        s[0] = sxx;
        s[1] = szz;
        s[2] = sxz;
        
        // Since yiedling didn't occur, principal stresses are still those of the elastic guesses.
        sig1 = p_reduced[0];
        sig3 = p_reduced[2];
	    I2pp_diff = 0.0; // I2pp must have not changed.

        return;
    }

    failure_mode += 10;

    // shear failure
    const double alams = fs / (a1 - a2*anpsi + a1*anphi*anpsi - a2*anphi + hardn);
    p[0] -= alams * (a1 - a2 * anpsi);
    p[1] -= alams * (a2 - a2 * anpsi);
    p[2] -= alams * (a2 - a1 * anpsi);

    // 2nd invariant of plastic strain
    depls = 0.5 * std::fabs(alams + alams * anpsi);

    //***********************************
    // The following seems redundant but... this is how it goes in geoFLAC.
    //
    // Possible tensile failure scenarios
    // 1. S1 (least tensional or greatest compressional principal stress) > ten_max:
    //    all three principal stresses must be greater than ten_max.
    //    Assign ten_max to all three and don't do anything further.
    if( p[0] >= ten_max ) {
        s[0] = s[1] = syy = ten_max;
        s[2] = 0.0;
        failure_mode += 20;
        sig1 = sig3 = (ten_max - pp);

        double tmp[NSTR];
        tmp[0] = tmp[1] = tmp[2] = sig1;
        double I2pp_new = I2_principal(E, nu, tmp);
	    I2pp_diff = I2pp - I2pp_new;
        return;
    }

    // 2. S2 (intermediate principal stress) > ten_max:
    //    S2 and S3 must be greater than ten_max.
    //    Assign ten_max to these two and continue to the shear failure block.
    if( p[1] >= ten_max ) {
        p[1] = p[2] = ten_max;
        failure_mode += 20;
    }

    // 3. S3 (most tensional or least compressional principal stress) > ten_max:
    //    Only this must be greater than ten_max.
    //    Assign ten_max to S3 and continue to the shear failure block.
    else if( p[2] >= ten_max ) {
        p[2] = ten_max;
        failure_mode += 20;
    }
    //***********************************

    sig1 = p[0]-pp;
    sig3 = p[2]-pp;
    double tmp[NSTR] = {sig1, syy, sig3}
    double I2pp_new = I2_principal(E, nu, tmp);
	I2pp_diff = I2pp - I2pp_new;
    
    // rotate the principal stresses back to global axes
    {
        double dc2 = (p[n1] - p[n2]) * cos2t;
        double dss = p[n1] + p[n2];
        s[0] = 0.5 * (dss + dc2);
        s[1] = 0.5 * (dss - dc2);
        s[2] = 0.5 * (p[n1] - p[n2]) * sin2t;
        syy = p[n3];
    }
}


void update_stress(const Variables& var, tensor_t& stress,
                   double_vec& stressyy,
                   tensor_t& strain, double_vec& plstrain,
                   double_vec& delta_plstrain, tensor_t& strain_rate, double_vec& dpressure)
{

    #pragma omp parallel for default(none)                           \
        shared(var, stress, stressyy, dpressure, strain, plstrain, delta_plstrain, \
        strain_rate, dpressure, std::cerr)
    for (int e=0; e<var.nelem; ++e) {
        int rheol_type = var.mat->rheol_type;

        // stress, strain and strain_rate of this element
        double* s = stress[e];
        double& syy = stressyy[e];
        double* es = strain[e];
        double* edot = strain_rate[e];
        double old_s = trace(s);

        // anti-mesh locking correction on strain rate
        if(1){
            double div = trace(edot);
            //double div2 = ((*var.volume)[e] / (*var.volume_old)[e] - 1) / var.dt;
            for (int i=0; i<NDIMS; ++i) {
                edot[i] += ((*var.edvoldt)[e] - div) / NDIMS;  // XXX: should NDIMS -> 3 in plane strain?
            }
        }

        // update strain with strain rate
        for (int i=0; i<NSTR; ++i) {
            es[i] += edot[i] * var.dt;
        }

        // modified strain increment
        double de[NSTR];
        for (int i=0; i<NSTR; ++i) {
            de[i] = edot[i] * var.dt;
        }

        switch (rheol_type) {
        case MatProps::rh_elastic:
            {
                double bulkm = var.mat->bulkm(e);
                double shearm = var.mat->shearm(e);
                elastic(bulkm, shearm, de, s);
            }
            break;
        case MatProps::rh_viscous:
            {
                double bulkm = var.mat->bulkm(e);
                double viscosity = var.mat->visc(e);
                double total_dv = trace(es);
                viscous(bulkm, viscosity, total_dv, edot, s);
            }
            break;
        case MatProps::rh_maxwell:
            {
                double bulkm = var.mat->bulkm(e);
                double shearm = var.mat->shearm(e);
                double viscosity = var.mat->visc(e);
                double dv = (*var.volume)[e] / (*var.volume_old)[e] - 1;
                maxwell(bulkm, shearm, viscosity, var.dt, dv, de, s);
            }
            break;
        case MatProps::rh_ep:
            {
                double depls = 0;
                double bulkm = var.mat->bulkm(e);
                double shearm = var.mat->shearm(e);
                double amc, anphi, anpsi, hardn, ten_max;
                var.mat->plastic_props(e, plstrain[e],
                                       amc, anphi, anpsi, hardn, ten_max);
                int failure_mode;
                if (var.mat->is_plane_strain) {
                    elasto_plastic2d(bulkm, shearm, amc, anphi, anpsi, hardn, ten_max,
                                     de, depls, s, syy, failure_mode);
                }
                else {
                    elasto_plastic(bulkm, shearm, amc, anphi, anpsi, hardn, ten_max,
                                   de, depls, s, failure_mode);
                }
                plstrain[e] += depls;
                delta_plstrain[e] = depls;
            }
            break;
        case MatProps::rh_evp:
            {
                double depls = 0;
                double bulkm = var.mat->bulkm(e);
                double shearm = var.mat->shearm(e);
                double viscosity = var.mat->visc(e);
                double dv = (*var.volume)[e] / (*var.volume_old)[e] - 1;
                // stress due to maxwell rheology
                double sv[NSTR];
                for (int i=0; i<NSTR; ++i) sv[i] = s[i];
                maxwell(bulkm, shearm, viscosity, var.dt, dv, de, sv);
                double svII = second_invariant2(sv);

                double amc, anphi, anpsi, hardn, ten_max;
                var.mat->plastic_props(e, plstrain[e],
                                       amc, anphi, anpsi, hardn, ten_max);
                // stress due to elasto-plastic rheology
                double sp[NSTR], spyy;
                for (int i=0; i<NSTR; ++i) sp[i] = s[i];
                int failure_mode;
                if (var.mat->is_plane_strain) {
                    spyy = syy;
                    elasto_plastic2d(bulkm, shearm, amc, anphi, anpsi, hardn, ten_max,
                                     de, depls, sp, spyy, failure_mode);
                }
                else {
                    elasto_plastic(bulkm, shearm, amc, anphi, anpsi, hardn, ten_max,
                                   de, depls, sp, failure_mode);
                }
                double spII = second_invariant2(sp);

                // use the smaller as the final stress
                if (svII < spII)
                    for (int i=0; i<NSTR; ++i) s[i] = sv[i];
                else {
                    for (int i=0; i<NSTR; ++i) s[i] = sp[i];
                    plstrain[e] += depls;
                    delta_plstrain[e] = depls;
                    syy = spyy;
                }
            }
            break;
        case MatProps::rh_ep_rs:
            {
                double depls = 0;
                double bulkm = var.mat->bulkm(e);
                double shearm = var.mat->shearm(e);
                double amc, anphi, anpsi, hardn, ten_max;
                int failure_mode;

                var.mat->plastic_props(e, plstrain[e],
                                       amc, anphi, anpsi, hardn, ten_max);
                // re-calculate some of the plasticity parameters
                // for the rate-and-state friction law.
                get_rate_state_plastic_props(var, e, amc, anphi, ten_max);

                // some potentially useful quantities 
                // are compute but not stored for now.
                double sig1, sig3, I2pp_diff;
                if (var.mat->is_plane_strain) {
                    elasto_plastic2d_rs(bulkm, shearm,
                                amc, anphi, anpsi,
                                hardn, ten_max,
                                de, var.mat->pore_pressure_factor, depls, s, syy, failure_mode,
                                sig1, sig3, I2pp_diff);
                }
                else {
                    elasto_plastic_rs(bulkm, shearm,
                              amc, anphi, anpsi,
                              hardn, ten_max,
                              de, var.mat->pore_pressure_factor, depls, s, failure_mode,
                              sig1, sig3, I2pp_diff);
                }
                plstrain[e] += depls;
                delta_plstrain[e] = depls;
                //update_max_shear(var,e,s);
            }
            break;
        case MatProps::rh_evp_rs:
            {
                double depls = 0;
                double bulkm = var.mat->bulkm(e);
                double shearm = var.mat->shearm(e);
                //double viscosity = var.mat->visc(e);
                double viscosity = var.mat->visc(e);
                double dv = (*var.volume)[e] / (*var.volume_old)[e] - 1;
                // stress due to maxwell rheology
                double sv[NSTR];
                for (int i=0; i<NSTR; ++i) sv[i] = s[i];
                maxwell(bulkm, shearm, viscosity, var.dt, dv, de, sv);
                double svII = second_invariant2(sv);

                double amc, anphi, anpsi, hardn, ten_max;
                int failure_mode;
                var.mat->plastic_props(e, plstrain[e],
                                       amc, anphi, anpsi, hardn, ten_max);
                // stress due to elasto-plastic rheology
                double sp[NSTR], spyy;
                for (int i=0; i<NSTR; ++i) sp[i] = s[i];

                // re-calculate some of the plasticity parameters
                // for the rate-and-state friction law.
                get_rate_state_plastic_props(var, e, amc, anphi, ten_max);
                
                // some potentially useful quantities 
                // are compute but not stored for now.
                double sig1, sig3, I2pp_diff;
                if (var.mat->is_plane_strain) {
                    spyy = syy;
                    elasto_plastic2d_rs(bulkm, shearm,
                                amc, anphi, anpsi,
                                hardn, ten_max,
                                de, var.mat->pore_pressure_factor, depls, s, spyy, failure_mode,
                                sig1, sig3, I2pp_diff);
                }
                else {
                    elasto_plastic_rs(bulkm, shearm,
                              amc, anphi, anpsi,
                              hardn, ten_max,
                              de, var.mat->pore_pressure_factor, depls, s, failure_mode,
                              sig1, sig3, I2pp_diff);
                }
                double spII = second_invariant2(sp);

                // use the smaller as the final stress
                if (svII < spII)
                    for (int i=0; i<NSTR; ++i) s[i] = sv[i];
                else {
                    for (int i=0; i<NSTR; ++i) s[i] = sp[i];
                    plstrain[e] += depls;
                    delta_plstrain[e] = depls;
                    syy = spyy;
                }
                //update_max_shear(var,e,s);
            }
            break;    
        default:
            std::cerr << "Error: unknown rheology type: " << rheol_type << "\n";
            std::exit(1);
            break;
        }
	    dpressure[e] = trace(s) - old_s;
    // std::cerr << "stress " << e << ": ";
    // print(std::cerr, s, NSTR);
    // std::cerr << '\n';
    }
}


void get_rate_state_plastic_props(const Variables &var,const int e, double &amc, double &anphi, double &ten_max)
{
    double centerxzT[3];
    double direct_a, evolution_b, a_b, char_vel, mu_static, normal_traction, normal;
    
    a_b = var.mat->a_b(e); // input file-provided value
    // a_b can be overwritten by a custom function that is
    // called in the following function.
    get_rate_state_parameters(var, e, double *centerxzT, direct_a, evolution_b, a_b, char_vel, mu_static, normal_traction, normal);

    double SV = get_slip_velocity(var, e);
    double mu_RS = mu_static + a_b * std::log(SV / char_vel);
    double phi_RS = std::fabs(std::atan(mu_RS)); 

    // amc = 2 * cohesion * sqrt(anphi)
    // since anphi might have been changed,
    // amc_new = 2 * cohesion * sqrt(anphi_new)
    //         = 2 * cohesion * sqrt(anphi) * sqrt(anphi_new/anphi)
    //         = amc_old * sqrt(anphi_new/anphi)
    // 1. retrieve cohesion from amc
    const double cohesion = 0.5 * cohesion / std::sqrt(anphi);
    // 2. assign anphi_new to anphi_new_to_old.
    double anphi_new_to_old = (1.0 + std::sin(phi_RS)) / (1.0 - std::sin(phi_RS)); 
    // 3. divide by anphi to make anphi_new_to_old the desired ratio.
    anphi_new_to_old /= anphi;
    // 4. update amc with the new anphi.
    amc *= std::sqrt(anphi_new_to_old);
    
    ten_max = (phi_RS == 0.0)? 1.0e9 : std::min(1e9, cohesion/mu_RS);
}

double get_slip_velocity(const Variables &var, const int e)
{
    double V_slip_m[3]={0., 0., 0.};
    double vm_sqrsum = 0.0, vm = 0.0;
    const int *conn = (*var.connectivity)[e];

    for (int i=0; i<NODES_PER_ELEM; ++i) {
        for (int j=0; j<NDIMS; ++j)
            V_slip_m[j] += (*var.vel)[conn[i]][j];
    }
    for (int i=0; i < NDIMS; ++i) {
        V_slip_m[i] /= NODES_PER_ELEM;
        vm_sqrsum += V_slip_m[i]*V_slip_m[i];
    }
    vm = std::sqrt(vm_sqrsum);
    
    if (std::isnan(vm) {
        std::cerr << "Error: V_slip becomes NaN\n";
        std::exit(11);
    }
    if (std::isinf(vm) {
        std::cerr << "Error: V_slip becomes inf\n";
        std::exit(11);
    }

    return std::max(1e-19, vm);
}

void get_rate_state_parameters( const Varialbes &var, const int e, double *centerxzT, double &direct_a, double &evolution_b, double &a_b, double &char_vel, double &mu_static, double &normal_traction, double &normal)
{
    find_element_center_vars(var, e, center_xzT);
    // find_normal_traction(p, normal_traction, normal);

    // center_xzT[2] is element mean temperautre
    f_v_gabbro(center_xzT[2], a_b, char_vel, mu_static);
}

void find_element_center_vars(Variables& var, int e, 
        double *center_xzT )
{
    const int *conn = (*var.connectivity)[e];
    center_xzT[0] = 0.; // mean x coord of element e
    center_xzT[1] = 0.; // mean z coord of element e
    center_xzT[2] = 0.; // mean temperature of element e
    for (int i=0; i<NODES_PER_ELEM; ++i){
        center_xzT[0] += (*var.coord)[conn[i]][0];
        center_xzT[1] += (*var.coord)[conn[i]][NDIMS-1];
        center_xzT[2] += (*var.temperature)[conn[i]];
    }
    for (int i=0; i<3; ++i)
        center_xzT[i] /= NODES_PER_ELEM;
}

void find_normal_traction( const double *p, double &normal_traction, double &normal )
{
    normal_traction = 0.5*std::fabs(p[0] + p[NDIMS-1]);
    for (int i=0; i<NDIMS; ++i) normal += p[i] / NDIMS;
    normal = std::fabs(normal);
}

double I2_principal(const double E, const double nu, const double p[3])
{
    return ( 0.5*(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]) - nu*(p[0]*p[1]+p[1]*p[2]+p[2]*p[0]) ) / E;
}
    
void f_v_granite(const double T, double &a_b, double &characteristic_velocity, double &static_friction_coefficient)
{
    int nlayers = 6, n;
    double ref_T[] = {0., 100., 350., 450., 1000., 2000};
    double ref_a_b[] = {0.004, -0.004, -0.004, 0.015, 0.1195, 0.3095};
    for (n = 1; n < nlayers; n++) {
	    if (T <= ref_T[n]) break;
    }
    a_b = ref_a_b[n-1] + (ref_a_b[n] - ref_a_b[n-1]) * (T - ref_T[n-1]) / (ref_T[n] - ref_T[n-1]);
    static_friction_coefficient = 0.6;// - a_b * log(3.3e-9/1e-6);
    characteristic_velocity = 1e-9;
}

void f_v_gabbro(const double T, double &a_b, double &characteristic_velocity, double &static_friction_coefficient)
{
    int nlayers = 6, n;
    double ref_T[] = {0., 100., 416., 520., 1000., 2000,};
    double ref_a_b[] = {0.0035, -0.0035, -0.0035, 0.001, 0.0218, 0.065};
    for (n = 1; n < nlayers; n++) {
        if (T <= ref_T[n]) break;
    }
    a_b = ref_a_b[n-1] + (ref_a_b[n] - ref_a_b[n-1]) * (T - ref_T[n-1]) / (ref_T[n] - ref_T[n-1]);
    static_friction_coefficient = 0.6;// - a_b * log(3.3e-9/1e-6);
    characteristic_velocity = 1e-6;
}

#if 0
void get_principal_stresses(const double* s, double p[NDIMS])
{
#ifdef THREED
    // eigenvectors
    double v[3][3];
    principal_stresses3(s, p, v);
#else
    // In 2D, we only construct the eigenvectors from
    // cos(2*theta) and sin(2*theta) of Mohr circle
    double cos2t, sin2t;
    principal_stresses2(s, p, cos2t, sin2t);
#endif
}

void update_max_shear(Variables &var, int e, const double *s, int rheol_type)
{
    double p[NDIMS];
    get_principal_stresses(s, p)

    if( rheol_type == MatProps::rh_ep_rs) {
        (*var.MAX_shear)[e] = ten_max;
        (*var.MAX_shear_0)[e] = amc;
        (*var.MAX_shear_0)[e] = amc;//p[2]
		(*var.strain_energy)[e] = hardn;
		if (center_z < -10e3) {
		    var.avg_vm += SV * (*var.volume)[e];
		    var.avg_shear_stress += ten_max * (*var.volume)[e];
		    var.slip_area += (*var.volume)[e];
		}

        (*var.RS_shear)[e] = normal; //shear_max * (1 - var.mat->pore_pressure_factor);
        (*var.friction_coefficient)[e] = std::sqrt(anphi); 
        (*var.Failure_mode)[e] = (failure_mode != 0/*amc > yield_stress*//*cohesion * std::sqrt(anphi)*/)? 1 : 0; // 1 if shear greater than yield
    }
    (*var.MAX_shear)[e] = p[0];
    (*var.MAX_shear_0)[e] = p[NDIMS-1];
}

void update_ten_max_amc( const double p[3], const double pp, double &ten_max, double &amc)
{
    ten_max = p[0]-pp;
    hardn = hardn_P - ((p[0]-pp)*(p[0]-pp)+(p[1]-pp)*(p[1]-pp)+(p[2]-pp)*(p[2]-pp)-2*v*((p[0]-pp)*(p[1]-pp)+(p[1]-pp)*(p[2]-pp)+(p[2]-pp)*(p[0]-pp)))/2/E; 
    amc = p[2]-pp;
}
#endif

void update_rheol_type_material(const Variables &var, const int e, int &rheol_type, int &material)
{
    // Find the most abundant marker mattype in this element
    int_vec &a = (*var.elemmarkers)[e];
    material = std::distance(a.begin(), std::max_element(a.begin(), a.end()));
    //if (material != 0 )
    if ((*var.CL)[e] >= 0.0)
        rheol_type = MatProps::rh_maxwell_rs;
}

double get_rate_state_viscosity(const Variables &var, const int e, const double* edot, const int material)
{    
    if (material == 10) {
        double direct_a, evolution_b, char_vel, mu_static;
        find_friction_parameters(45.0e3, direct_a, evolution_b,
                            char_vel, mu_static);
    
        double normal_traction, normal;
        find_normal_traction(p, normal_traction, normal);

        // compute effective friction
        double mu_dyn = (mu_static + evolution_b*(*var.state1)[e])/direct_a;
        double A = 2.0*char_vel*std::exp(-mu_dyn);

        double SV = std::max(1e-19, (*var.slip_velocity)[e]);
        double shear_max = std::fabs(direct_a * normal * std::asinh(SV/A)));
        double s2n = shear_max / normal;
        double eff_friction = std::sqrt((1.0/s2n-s2n);

        double min_strain_rate = 1e-45;
        double eII = std::max(second_invariant(edot), min_strain_rate);
        double viscosity = eff_friction*shear_max/eII;

        // store these in global data arrays:
        (*var.RS_shear)[e] = shear_max * (1.0 - var.mat->pore_pressure_factor);
        (*var.friction_coefficient)[e] = viscosity;
        
        return viscosity;
    }
    else 
        return var.mat->visc(e);
}

void find_friction_parameters(double x, double &direct_a, double &evolution_b, double &characteristic_velocity, double &static_friction_coefficient)
{
    double SV = std::max(1e-19, (*var.slip_velocity)[e]);
    double T1 = 0, T2 = 43e3,T3 = 45e3, T4 = 45e3, T5 = 47e3, T6 = 90e3;
    double T = std::fabs(x);
    double vc1 = 1e-6, vc2 = 1e-6, vc3 = 1e-6, vc4 = 1e-6, vc5 = 1e-6, vc6 = 1e-6;
    double a1 = 0.019, a2 = 0.019, a3 = 0.015, a4 = 0.015, a5 = 0.019, a6 = 0.019;
    double b1 = 0.015, b2 = 0.015, b3 = 0.019, b4 = 0.019, b5 = 0.015, b6 = 0.015;
    double f0_1 = 0.6, f0_2 = 0.6, f0_3 = 0.6, f0_4 = 0.6, f0_5 = 0.6, f0_6 = 0.6;

    if (T < T2) {
        double f = (f0_2-f0_1)/(T2-T1);
        double a = (a2-a1)/(T2-T1);
        double b = (b2-b1)/(T2-T1);
        double vc = (vc2-vc1)/(T2-T1);
        static_friction_coefficient = T*f+(f0_1-f*T1);
        direct_a = T*a+(a1-a*T1);
        evolution_b = T*b+(b1-b*T1);
        characteristic_velocity = T*vc+(vc1-vc*T1);
    }
    else if (T >= T2 && T < T3) {
        double f = (f0_3-f0_2)/(T3-T2);
        double a = (a3-a2)/(T3-T2);
        double b = (b3-b2)/(T3-T2);
        double vc = (vc3-vc2)/(T3-T2);
        static_friction_coefficient = T*f+(f0_2-f*T2);
        direct_a = T*a+(a2-a*T2);
        evolution_b = T*b+(b2-b*T2);
        characteristic_velocity = T*vc+(vc2-vc*T2);
    }
    else if (T >= T3 && T < T4) {
        double f = (f0_4-f0_3)/(T4-T3);
        double a = (a4-a3)/(T4-T3);
        double b = (b4-b3)/(T4-T3);
        double vc = (vc4-vc3)/(T4-T3);
        static_friction_coefficient = T*f+(f0_3-f*T3);
        direct_a = T*a+(a3-a*T3);
        evolution_b = T*b+(b3-b*T3);
        characteristic_velocity = T*vc+(vc3-vc*T3);
    }
    else if (T >= T4 && T < T5) {
        double f = (f0_5-f0_4)/(T5-T4);
        double a = (a5-a4)/(T5-T4);
        double b = (b5-b4)/(T5-T4);
        double vc = (vc5-vc4)/(T5-T4);
        static_friction_coefficient = T*f+(f0_4-f*T4);
        direct_a = T*a+(a4-a*T4);
        evolution_b = T*b+(b4-b*T4);
        characteristic_velocity = T*vc+(vc4-vc*T4);
    }
    else {
        double f = (f0_6-f0_5)/(T6-T5);
        double a = (a6-a5)/(T6-T5);
        double b = (b6-b5)/(T6-T5);
        double vc = (vc6-vc5)/(T6-T5);
	    static_friction_coefficient = T*f+(f0_5-f*T5);
        direct_a = T*a+(a5-a*T5);
        evolution_b = T*b+(b5-b*T5);
        characteristic_velocity = T*vc+(vc5-vc*T5);
    }
}
#endif