#ifndef DYNEARTHSOL3D_RHEOLOGY_HPP
#define DYNEARTHSOL3D_RHEOLOGY_HPP

void update_stress(const Variables& var, tensor_t& stress, double_vec& stressyy,
                   double_vec& dpressure, tensor_t& strain, double_vec& plstrain,
                   double_vec& delta_plstrain, tensor_t& strain_rate);
#ifdef RS
void friction_variables(double &T, double &direct_a, double &evolution_b, double &characteristic_velocity, double &static_friction_coefficient);
void update_state1(const Variables &var, double_vec &state1, double_vec &slip_velocity, int a);
#endif

#endif
