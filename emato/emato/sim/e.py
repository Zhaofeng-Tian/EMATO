import numpy as np
import casadi as ca
import time
from emato.util.util import get_gradient_profile
from emato.obj.poly.quintic import Traj



class EMATO():
    def __init__(self, traj, param, dv):
        self.param = param
        self.dv = dv
        self.nx = 2 ; self.nu = 1
        self.vr = dv ; self.xr = param.xr
        self.xmin = param.xmin.copy() ; self.xmax = param.xmax.copy()
        self.umin = param.umin ; self.umax = param.umax
        self.vmin = param.vmin; self.vmax = param.vmax
        self.bmin = param.bmin; self.bmax = param.bmax
        self.w1 = param.w1 ;self.w2 = param.w2 ;self.w3 = param.w3
        self.T = traj.Traj_t[-1] ; 
        self.N = traj.Traj_len; self.dt = param.dt
        print("Trajectory length: ", self.N)
        # self.dmin = param.dmin; self.dmax = param.dmax; self.dinit = param.dinit
        self.theta = 0.
        _, self.gd_profile = get_gradient_profile(param.gd_profile_type)
        self.use_gd_prediction = param.use_gd_prediction

        self.k1,self.k2,self.k3 = param.k
        # self.b0, self.b1, self.b2, self.b3 = param.b
        self.c0,self.c1,self.c2 = param.c
        self.o0,self.o1,self.o2,self.o3,self.o4 = param.o
        self.xs = traj.get_x(i=0)
        self.xe = traj.get_x(i=-1)
        self.gdec = traj.Traj_theta.copy()
        #self.p = ca.vertcat(self.SL, self.VL, self.sec, self.vec, self.gdec)
        # self.p = ca.MX.sym('p',1)
        self.S = ca.MX.sym('S', self.N)
        self.U = ca.MX.sym('U', self.N)  # Control input vector
        self.V = ca.MX.sym('V', self.N)  # Velocity vector
        self.A = ca.MX.sym('A', self.N)  # Apparent Acceleration
        self.B = ca.MX.sym('B', self.N)
        self.Jerk = ca.MX.sym('Jerk',self.N)
        self.fu = self.o0 + self.o1 * self.V + self.o2 * self.V**2 + \
                    self.o3 * self.V**3 +self. o4 * self.V**4 + \
                        (self.c0 + self.c1 * self.V + self.c2 * self.V**2) * self.U
        
        # self.J = ca.sum1(self.w1 * (self.V - self.VL)**2 + self.w2 * self.A**2 + self.w2*self.B**2 + self.w3 * self.fu)
        # self.J1 = self.w1 * self.Jerk**2
        # self.J2 = self.w2 * (self.A**2 + self.B**2)
        # self.J3 = self.w3 * self.fu
        # self.J1 = 1 * self.Jerk**2
        # self.J2 = 10 * (self.A**2 + self.B**2)
        # self.J3 = 38.91* self.w3 * self.fu/(self.S[-1]-self.S[0])
        # self.J = ca.sum1(self.J1 + self.J2 + self.J3)
        self.J = self.w1 * ca.sum1(self.Jerk**2) +\
                self.w2 * ca.sum1(self.A**2 + self.B**2)+ \
                self.w3 *ca.sum1(self.fu)/ (self.S[-1] - self.S[0])

        self.gfa = self.A-   ( self.U -self.B- self.k1 * self.V**2 \
                              - self.k2 * ca.cos(self.gdec) - self.k3 * ca.sin(self.gdec) ) 
        self.gfv = [] ; self.gfd = []; self.gfj = []
        for i in range(self.N-1):
            self.gfj.append(self.A[i+1]-(self.A[i]+self.Jerk[i]*self.dt))
        for i in range(self.N-1):
            self.gfv.append(self.V[i+1] - (self.V[i] + self.A[i] * self.dt))
        for i in range(self.N-1):
            self.gfd.append(self.S[i+1]-(self.S[i] + self.V[i]*self.dt + 0.5*self.A[i]* self.dt**2))
        self.gss = self.S[0] - self.xs[0]
        self.gvs = self.V[0] - self.xs[1]
        self.gas = self.A[0] - self.xs[2]
        self.gse = self.S[-1] - self.xe[0]
        self.gve = self.V[-1] - self.xe[1]
        self.gae = self.A[-1] - self.xe[2]
        self.g = ca.vertcat(self.gfa, *self.gfv, *self.gfd, *self.gfj, self.gss, self.gvs,\
                            self.gas, self.gse, self.gve, self.gae)
        # self.g = ca.vertcat(self.gfa, *self.gfv, *self.gfd, self.gss, self.gvs,\
        #                     self.gas, self.gse)
        self.nlp = {'x': ca.vertcat(self.S, self.V, self.A, self.U, self.B, self.Jerk), \
                    'f': self.J, 'g':self.g}
        self.opts = {
            'ipopt.print_level': 0,  # Adjust verbosity (0 to 12, with 0 being silent and 12 being very verbose)
            'print_time': True,  # Print solver execution time
            'ipopt.tol': 1e-8,  # Tolerance (adjust as needed)
        }
        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp, self.opts)
        self.init_guess = np.concatenate((traj.Traj_s, traj.Traj_v, traj.Traj_a,\
                                          traj.Traj_at, traj.Traj_ab, traj.Traj_j))
        self.lb_s = -np.inf * np.ones(self.N)
        self.lb_v = np.zeros(self.N)  # v >= 0
        self.lb_u = self.umin * np.ones(self.N)  # u >= -1
        self.lb_a = -10.0 * np.ones(self.N)
        self.lb_b = self.bmin * np.ones(self.N)
        self.lb_jerk = -self.param.jmax * np.ones(self.N)
        self.lbx = np.concatenate((self.lb_s, self.lb_v, self.lb_a, self.lb_u, self.lb_b, self.lb_jerk))

        self.ub_s = np.inf * np.ones(self.N)
        self.ub_v = 27.* np.ones(self.N)  # v >= 0
        self.ub_u = self.umax * np.ones(self.N)  # u >= -1
        self.ub_a = 10.0 * np.ones(self.N)
        self.ub_b = self.bmax * np.ones(self.N)
        self.ub_jerk = self.param.jmax* np.ones(self.N)
        self.ubx = np.concatenate((self.ub_s, self.ub_v, self.ub_a, self.ub_u, self.ub_b, self.ub_jerk))

        self.lbg = np.zeros(4*self.N-3+6)  # Lower bounds of g
        # self.ubg = np.concatenate(( self.dmax * np.ones(self.N), np.zeros(3*self.N)))  # Upper bounds of g
        self.ubg = np.zeros(4*self.N-3+6)

        self.solve_time = 0.

    def solve(self):
        t1 = time.time()
        res = self.solver(x0=self.init_guess, lbx=self.lbx, ubx=self.ubx,\
                          lbg = self.lbg, ubg = self.ubg)
        self.res = np.array(res['x']).flatten()
        t2 = time.time()
        self.solve_time = t2-t1
        
        otraj_s = self.res[0:self.N]
        otraj_v = self.res[self.N:2*self.N]
        otraj_a = self.res[2*self.N: 3*self.N]
        otraj_at = self.res[3*self.N: 4*self.N]
        otraj_ab = self.res[4*self.N: 5*self.N]
        otraj_j = self.res[5*self.N:]

        otraj = Traj(self.param, None, self.gd_profile, self.dv, \
                     (otraj_s,otraj_v,otraj_a,otraj_at, otraj_ab, otraj_j), False)



        #　return otraj_s, otraj_v, otraj_a, otraj_at, otraj_av
        return otraj



class EMATO_R():
    def __init__(self, traj, param, dv):
        self.param = param
        self.dv = dv
        self.nx = 2 ; self.nu = 1
        self.vr = dv ; self.xr = param.xr
        self.xmin = param.xmin.copy() ; self.xmax = param.xmax.copy()
        self.umin = param.umin ; self.umax = param.umax
        self.vmin = param.vmin; self.vmax = param.vmax
        self.bmin = param.bmin; self.bmax = param.bmax
        self.w1 = param.w1 ;self.w2 = param.w2 ;self.w3 = param.w3
        self.T = traj.Traj_t[-1] ; 
        self.N = traj.Traj_len; self.dt = param.dt
        print("Trajectory length: ", self.N)
        # self.dmin = param.dmin; self.dmax = param.dmax; self.dinit = param.dinit
        self.theta = 0.
        _, self.gd_profile = get_gradient_profile(param.gd_profile_type)
        self.use_gd_prediction = param.use_gd_prediction

        self.k1,self.k2,self.k3 = param.k
        # self.b0, self.b1, self.b2, self.b3 = param.b
        self.c0,self.c1,self.c2 = param.c
        self.o0,self.o1,self.o2,self.o3,self.o4 = param.o
        self.xs = traj.get_x(i=0)
        self.xe = traj.get_x(i=-1)
        self.gdec = traj.Traj_theta.copy()
        #self.p = ca.vertcat(self.SL, self.VL, self.sec, self.vec, self.gdec)
        # self.p = ca.MX.sym('p',1)
        self.S = ca.MX.sym('S', self.N)
        self.U = ca.MX.sym('U', self.N)  # Control input vector
        self.V = ca.MX.sym('V', self.N)  # Velocity vector
        self.A = ca.MX.sym('A', self.N)  # Apparent Acceleration
        self.B = ca.MX.sym('B', self.N)
        self.Jerk = ca.MX.sym('Jerk',self.N)
        self.fu = self.o0 + self.o1 * self.V + self.o2 * self.V**2 + \
                    self.o3 * self.V**3 +self. o4 * self.V**4 + \
                        (self.c0 + self.c1 * self.V + self.c2 * self.V**2) * self.U
        
        # self.J = ca.sum1(self.w1 * (self.V - self.VL)**2 + self.w2 * self.A**2 + self.w2*self.B**2 + self.w3 * self.fu)
        # self.J1 = self.w1 * self.Jerk**2
        # self.J2 = self.w2 * (self.A**2 + self.B**2)
        # self.J3 = self.w3 * self.fu
        # self.J1 = 1 * self.Jerk**2
        # self.J2 = 10 * (self.A**2 + self.B**2)
        # self.J3 = 38.91* self.w3 * self.fu/(self.S[-1]-self.S[0])
        # self.J = ca.sum1(self.J1 + self.J2 + self.J3)
        self.J = self.w1 * ca.sum1(self.Jerk**2) +\
                self.w2 * ca.sum1(self.A**2 + self.B**2)+ \
                self.w3 *ca.sum1(self.fu)/ (self.S[-1] - self.S[0])

        self.gfa = self.A-   ( self.U -self.B- self.k1 * self.V**2 \
                              - self.k2 * ca.cos(self.gdec) - self.k3 * ca.sin(self.gdec) ) 
        self.gfv = [] ; self.gfd = []; self.gfj = []
        for i in range(self.N-1):
            self.gfj.append(self.A[i+1]-(self.A[i]+self.Jerk[i]*self.dt))
        for i in range(self.N-1):
            self.gfv.append(self.V[i+1] - (self.V[i] + self.A[i] * self.dt))
        for i in range(self.N-1):
            self.gfd.append(self.S[i+1]-(self.S[i] + self.V[i]*self.dt + 0.5*self.A[i]* self.dt**2))
        self.gss = self.S[0] - self.xs[0]
        self.gvs = self.V[0] - self.xs[1]
        self.gas = self.A[0] - self.xs[2]
        self.gse = self.S[-1] - self.xe[0]
        self.gve = self.V[-1] - self.xe[1]
        # self.gae = self.A[-1] - self.xe[2]
        # self.g = ca.vertcat(self.gfa, *self.gfv, *self.gfd, *self.gfj, self.gss, self.gvs,\
        #                     self.gas, self.gse, self.gve, self.gae)
        self.g = ca.vertcat(self.gfa, *self.gfv, *self.gfd, *self.gfj, self.gss, self.gvs,\
                            self.gas, self.gve, self.gse)
        self.nlp = {'x': ca.vertcat(self.S, self.V, self.A, self.U, self.B, self.Jerk), \
                    'f': self.J, 'g':self.g}
        self.opts = {
            'ipopt.print_level': 0,  # Adjust verbosity (0 to 12, with 0 being silent and 12 being very verbose)
            'print_time': True,  # Print solver execution time
            'ipopt.tol': 1e-8,  # Tolerance (adjust as needed)
        }
        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp, self.opts)
        self.init_guess = np.concatenate((traj.Traj_s, traj.Traj_v, traj.Traj_a,\
                                          traj.Traj_at, traj.Traj_ab, traj.Traj_jerk))
        self.lb_s = -np.inf * np.ones(self.N)
        self.lb_v = np.zeros(self.N)  # v >= 0
        self.lb_u = self.umin * np.ones(self.N)  # u >= -1
        self.lb_a = -5.0 * np.ones(self.N)
        self.lb_b = self.bmin * np.ones(self.N)
        self.lb_jerk = -self.param.jmax * np.ones(self.N)
        self.lbx = np.concatenate((self.lb_s, self.lb_v, self.lb_a, self.lb_u, self.lb_b, self.lb_jerk))

        self.ub_s = np.inf * np.ones(self.N)
        self.ub_v = 27.* np.ones(self.N)  # v >= 0
        self.ub_u = self.umax * np.ones(self.N)  # u >= -1
        self.ub_a = 5.0 * np.ones(self.N)
        self.ub_b = self.bmax * np.ones(self.N)
        self.ub_jerk = self.param.jmax* np.ones(self.N)
        self.ubx = np.concatenate((self.ub_s, self.ub_v, self.ub_a, self.ub_u, self.ub_b, self.ub_jerk))

        self.lbg = np.zeros(4*self.N-3+4)  # Lower bounds of g
        # self.ubg = np.concatenate(( self.dmax * np.ones(self.N), np.zeros(3*self.N)))  # Upper bounds of g
        self.ubg = np.zeros(4*self.N-3+4)
        self.lbg = np.append(self.lbg, 0)
        self.ubg = np.append(self.ubg, 0)



        self.solve_time = 0.

    def solve(self):
        t1 = time.time()
        res = self.solver(x0=self.init_guess, lbx=self.lbx, ubx=self.ubx,\
                          lbg = self.lbg, ubg = self.ubg)
        self.res = np.array(res['x']).flatten()
        t2 = time.time()
        self.solve_time = t2-t1
        
        otraj_s = self.res[0:self.N]
        otraj_v = self.res[self.N:2*self.N]
        otraj_a = self.res[2*self.N: 3*self.N]
        otraj_at = self.res[3*self.N: 4*self.N]
        otraj_ab = self.res[4*self.N: 5*self.N]
        otraj_j = self.res[5*self.N:]

        otraj = Traj(self.param, None, self.gd_profile, self.dv, \
                     (otraj_s,otraj_v,otraj_a,otraj_at, otraj_ab, otraj_j), False)



        #　return otraj_s, otraj_v, otraj_a, otraj_at, otraj_av
        return otraj