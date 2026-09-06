// Dumps reference trajectories from the (patched) MRS C++ model.
// Each row is: the state at step k, and the throttle applied to reach step k+1.

#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>

#include "multirotor_model.hpp"
#include "uav_system.hpp"

using namespace mrs_multirotor_simulator;

static void writeStateHeader(FILE* f, const char* cmd_names) {
  fprintf(f, "step,%s,x0,x1,x2,v0,v1,v2", cmd_names);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) fprintf(f, ",R%d%d", i, j);
  fprintf(f, ",w0,w1,w2,rpm0,rpm1,rpm2,rpm3\n");
}

static void writeStateRow(FILE* f, int k, const Eigen::VectorXd& cmd, const MultirotorModel::State& st) {
  fprintf(f, "%d", k);
  for (int i = 0; i < cmd.size(); i++) fprintf(f, ",%.17g", cmd(i));
  for (int i = 0; i < 3; i++) fprintf(f, ",%.17g", st.x(i));
  for (int i = 0; i < 3; i++) fprintf(f, ",%.17g", st.v(i));
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) fprintf(f, ",%.17g", st.R(i, j));
  for (int i = 0; i < 3; i++) fprintf(f, ",%.17g", st.omega(i));
  for (int i = 0; i < 4; i++) fprintf(f, ",%.17g", st.motor_rpm(i));
  fprintf(f, "\n");
}

static MultirotorModel::ModelParams makeParams() {
  MultirotorModel::ModelParams p;  // x500 defaults
  p.ground_enabled        = false;
  p.takeoff_patch_enabled = false;
  return p;
}

static void writeParams(const std::string& path, const MultirotorModel::ModelParams& p) {
  FILE* f = fopen(path.c_str(), "w");
  fprintf(f, "n_motors %d\n", p.n_motors);
  fprintf(f, "g %.17g\nmass %.17g\nkf %.17g\nkm %.17g\n", p.g, p.mass, p.kf, p.km);
  fprintf(f, "prop_radius %.17g\narm_length %.17g\nbody_height %.17g\n", p.prop_radius, p.arm_length, p.body_height);
  fprintf(f, "motor_time_constant %.17g\nmin_rpm %.17g\nmax_rpm %.17g\n", p.motor_time_constant, p.min_rpm, p.max_rpm);
  fprintf(f, "air_resistance_coeff %.17g\n", p.air_resistance_coeff);
  for (int i = 0; i < 3; i++) fprintf(f, "J%d%d %.17g\n", i, i, p.J(i, i));
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) fprintf(f, "A%d%d %.17g\n", i, j, p.allocation_matrix(i, j));
  fclose(f);
}

struct Scenario {
  std::string                          name;
  int                                  steps;
  MultirotorModel::State               init;
  std::vector<Eigen::Vector4d>         cmd;  // one throttle vector per step
};

static void run(const std::string& dir, const Scenario& s, const MultirotorModel::ModelParams& p, double dt) {

  MultirotorModel m(p, s.init.x, 0.0);
  m.setState(s.init);

  FILE* f = fopen((dir + "/" + s.name + ".csv").c_str(), "w");
  writeStateHeader(f, "u0,u1,u2,u3");

  for (int k = 0; k <= s.steps; k++) {

    const Eigen::Vector4d& u = s.cmd[k < s.steps ? k : s.steps - 1];
    writeStateRow(f, k, u, m.getState());

    if (k == s.steps) break;

    reference::Actuators a;
    a.motors = u;
    m.setInput(a);
    m.step(dt);
  }

  fclose(f);
  printf("%-12s %4d steps -> %s.csv\n", s.name.c_str(), s.steps, s.name.c_str());
}

// Closed rate loop: rate controller -> mixer -> model, all at dt, as in UavSystem.
static void runRateLoop(const std::string& dir, const std::string& name, int steps,
                        const Eigen::Vector3d& rate_ref, double throttle,
                        const MultirotorModel::ModelParams& p, double dt) {

  UavSystem uav(p, Eigen::Vector3d::Zero(), 0.0);

  reference::AttitudeRate cmd;
  cmd.rate_x = rate_ref(0);
  cmd.rate_y = rate_ref(1);
  cmd.rate_z = rate_ref(2);
  cmd.throttle = throttle;

  const Eigen::Vector4d row(throttle, rate_ref(0), rate_ref(1), rate_ref(2));

  FILE* f = fopen((dir + "/" + name + ".csv").c_str(), "w");
  writeStateHeader(f, "throttle,rate_x,rate_y,rate_z");

  for (int k = 0; k <= steps; k++) {
    writeStateRow(f, k, row, uav.getState());
    if (k == steps) break;
    uav.setInput(cmd);
    uav.makeStep(dt);
  }
  fclose(f);

  FILE* a = fopen((dir + "/mixer_allocation.txt").c_str(), "w");
  Eigen::MatrixXd alloc = uav.getMixerAllocation();
  for (int i = 0; i < alloc.rows(); i++)
    for (int j = 0; j < alloc.cols(); j++) fprintf(a, "M%d%d %.17g\n", i, j, alloc(i, j));
  fclose(a);

  printf("%-12s %4d steps -> %s.csv, mixer_allocation.txt\n", name.c_str(), steps, name.c_str());
}

// Attitude command: attitude controller -> rate -> mixer -> model, as in UavSystem.
static void runAttitude(const std::string& dir, const std::string& name, int steps,
                        const Eigen::Matrix3d& orientation, double throttle,
                        const MultirotorModel::ModelParams& p, double dt) {

  UavSystem uav(p, Eigen::Vector3d::Zero(), 0.0);

  reference::Attitude cmd;
  cmd.orientation = orientation;
  cmd.throttle    = throttle;

  Eigen::VectorXd row(10);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) row(3 * i + j) = orientation(i, j);
  row(9) = throttle;

  FILE* f = fopen((dir + "/" + name + ".csv").c_str(), "w");
  std::string names;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) names += "Rd" + std::to_string(i) + std::to_string(j) + ",";
  names += "throttle";
  writeStateHeader(f, names.c_str());

  for (int k = 0; k <= steps; k++) {
    writeStateRow(f, k, row, uav.getState());
    if (k == steps) break;
    uav.setInput(cmd);
    uav.makeStep(dt);
  }
  fclose(f);

  printf("%-12s %4d steps -> %s.csv\n", name.c_str(), steps, name.c_str());
}

// Velocity command: velocity -> acceleration -> attitude -> rate -> mixer -> model.
static void runVelocity(const std::string& dir, const std::string& name, int steps,
                        const Eigen::Vector3d& velocity, double heading,
                        const MultirotorModel::ModelParams& p, double dt) {

  UavSystem uav(p, Eigen::Vector3d::Zero(), 0.0);

  reference::VelocityHdg cmd(velocity, heading);

  Eigen::VectorXd row(4);
  row << velocity(0), velocity(1), velocity(2), heading;

  FILE* f = fopen((dir + "/" + name + ".csv").c_str(), "w");
  writeStateHeader(f, "vx,vy,vz,heading");

  for (int k = 0; k <= steps; k++) {
    writeStateRow(f, k, row, uav.getState());
    if (k == steps) break;
    uav.setInput(cmd);
    uav.makeStep(dt);
  }
  fclose(f);
  printf("%-14s %4d steps -> %s.csv\n", name.c_str(), steps, name.c_str());
}

// Position command: the whole cascade.
static void runPosition(const std::string& dir, const std::string& name, int steps,
                        const Eigen::Vector3d& position, double heading,
                        const MultirotorModel::ModelParams& p, double dt) {

  UavSystem uav(p, Eigen::Vector3d::Zero(), 0.0);

  reference::Position cmd;
  cmd.position = position;
  cmd.heading  = heading;

  Eigen::VectorXd row(4);
  row << position(0), position(1), position(2), heading;

  FILE* f = fopen((dir + "/" + name + ".csv").c_str(), "w");
  writeStateHeader(f, "px,py,pz,heading");

  for (int k = 0; k <= steps; k++) {
    writeStateRow(f, k, row, uav.getState());
    if (k == steps) break;
    uav.setInput(cmd);
    uav.makeStep(dt);
  }
  fclose(f);
  printf("%-14s %4d steps -> %s.csv\n", name.c_str(), steps, name.c_str());
}

static MultirotorModel::State restState(double rpm) {
  MultirotorModel::State s;
  s.x         = Eigen::Vector3d::Zero();
  s.v         = Eigen::Vector3d::Zero();
  s.v_prev    = Eigen::Vector3d::Zero();
  s.R         = Eigen::Matrix3d::Identity();
  s.omega     = Eigen::Vector3d::Zero();
  s.motor_rpm = Eigen::VectorXd::Constant(4, rpm);
  return s;
}

int main(int argc, char** argv) {

  const std::string dir = argc > 1 ? argv[1] : ".";
  const double      dt  = 0.01;

  MultirotorModel::ModelParams p = makeParams();
  writeParams(dir + "/params.txt", p);

  const double hover_rpm      = sqrt((p.mass * p.g) / (p.n_motors * p.kf));
  const double hover_throttle = (hover_rpm - p.min_rpm) / (p.max_rpm - p.min_rpm);

  std::vector<Scenario> scenarios;

  {  // gravity, quadratic drag, and the min_rpm floor
    Scenario s{"free_fall", 100, restState(0.0), {}};
    s.cmd.assign(s.steps, Eigen::Vector4d::Zero());
    scenarios.push_back(s);
  }

  {  // the equilibrium
    Scenario s{"hover", 100, restState(hover_rpm), {}};
    s.cmd.assign(s.steps, Eigen::Vector4d::Constant(hover_throttle));
    scenarios.push_back(s);
  }

  {  // asymmetric thrust: all three torque axes, orthonormalization under rotation
    Scenario s{"tumble", 500, restState(0.0), {}};
    s.cmd.assign(s.steps, Eigen::Vector4d(0.55, 0.40, 0.52, 0.44));
    scenarios.push_back(s);
  }

  {  // nonzero omega at equal thrust isolates the gyroscopic term
    Scenario s{"spin_down", 200, restState(hover_rpm), {}};
    s.init.omega = Eigen::Vector3d(2.0, -1.5, 3.0);
    s.cmd.assign(s.steps, Eigen::Vector4d::Constant(hover_throttle));
    scenarios.push_back(s);
  }

  {  // everything nonzero, and a command that changes every step
    std::mt19937                           rng(0);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    auto sym = [&](double a) { return a * (2.0 * uni(rng) - 1.0); };

    Scenario s{"random", 200, restState(0.0), {}};
    s.init.x     = Eigen::Vector3d(sym(5), sym(5), sym(5));
    s.init.v     = Eigen::Vector3d(sym(3), sym(3), sym(3));
    s.init.omega = Eigen::Vector3d(sym(2), sym(2), sym(2));
    s.init.R     = Eigen::AngleAxisd(sym(M_PI), Eigen::Vector3d(sym(1), sym(1), sym(1)).normalized()).toRotationMatrix();
    for (int i = 0; i < 4; i++) s.init.motor_rpm(i) = p.min_rpm + uni(rng) * (p.max_rpm - p.min_rpm);
    for (int k = 0; k < s.steps; k++) s.cmd.push_back(Eigen::Vector4d(uni(rng), uni(rng), uni(rng), uni(rng)));
    scenarios.push_back(s);
  }

  for (const Scenario& s : scenarios) run(dir, s, p, dt);

  runRateLoop(dir, "rate_step", 300, Eigen::Vector3d(0.0, 1.0, 0.0), hover_throttle, p, dt);

  runAttitude(dir, "attitude_step", 300,
              Eigen::AngleAxisd(0.25, Eigen::Vector3d(0.6, 0.8, 0.0)).toRotationMatrix(),
              hover_throttle, p, dt);

  runVelocity(dir, "velocity_step", 500, Eigen::Vector3d(1.0, -0.5, 0.8), 0.4, p, dt);
  runPosition(dir, "position_step", 1500, Eigen::Vector3d(3.0, -2.0, 5.0), 0.5, p, dt);

  printf("hover_rpm %.17g  hover_throttle %.17g\n", hover_rpm, hover_throttle);
  return 0;
}