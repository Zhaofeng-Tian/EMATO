import numpy as np
import osqp
from scipy import sparse
import casadi as ca
import time 


class NLP:
    def __init__(self, param, traj_l, xc):
        self.id = "NLP"
        self.nx = 2 ; self.nu = 1
        self.vr = param.vr ; self.xr = param.xr
        self.xmin = param.xmin.copy() ; self.xmax = param.xmax.copy()
        self.umin = param.umin ; self.umax = param.umax
        self.vmin = param.vmin; self.vmax = param.vmax
        self.bmin = param.bmin; self.bmax = param.bmax
        self.w1 = param.w1 ;self.w2 = param.w2 ;self.w3 = param.w3
        self.T = param.T ; self.dmin = param.dmin; self.dmax = param.dmax
        self.N = param.N ; self.dt = param.dt ; self.dinit = param.dinit
        self.param = param
        self.theta = 0.
        _, self.gd_profile = get_gradient_profile(feature=param.gd_profile_type)
        self.use_gd_prediction = param.use_gd_prediction

        self.k1,self.k2,self.k3 = param.k
        # self.b0, self.b1, self.b2, self.b3 = param.b
        self.c0,self.c1,self.c2 = param.c
        self.o0,self.o1,self.o2,self.o3,self.o4 = param.o

        self.traj_sl, self.traj_vl, self.traj_al = traj_l
        self.xc = xc
        self.res = np.concatenate((self.traj_sl[:self.N]-self.dinit ,self.traj_vl[:self.N], 1.1*np.ones(self.N), 0.2*np.ones(self.N), np.zeros(self.N) ))

        self.SL = ca.MX.sym('SL', self.N)
        self.VL = ca.MX.sym('VL', self.N)
        self.sec = ca.MX.sym('sec', 1)
        self.vec = ca.MX.sym('vec', 1)
        if self.use_gd_prediction:
            self.gdec = ca.MX.sym('gdec', self.N)
        else:
            self.gdec = ca.MX.sym('gdec', 1)

        self.p = ca.vertcat(self.SL, self.VL, self.sec, self.vec, self.gdec)
        self.S = ca.MX.sym('S', self.N)
        self.U = ca.MX.sym('U', self.N)  # Control input vector
        self.V = ca.MX.sym('V', self.N)  # Velocity vector
        self.A = ca.MX.sym('A', self.N)  # Apparent Acceleration
        self.B = ca.MX.sym('B', self.N)

        self.p_value = None

        self.fu = self.o0 + self.o1 * self.V + self.o2 * self.V**2 + \
                    self.o3 * self.V**3 +self. o4 * self.V**4 + \
                        (self.c0 + self.c1 * self.V + self.c2 * self.V**2) * self.U
        # self.J = ca.sum1(self.w1 * (self.V - self.VL)**2 + self.w2 * self.A**2 + self.w3 * self.fu)
        self.J = ca.sum1(self.w1 * (self.V - self.VL)**2 + self.w2 * self.A**2 + self.w2*self.B**2 + self.w3 * self.fu)
        self.gacc = self.p[:self.N] - (self.S + self.T * self.V + self.dmin)
        self.gfa = self.A-   ( self.U -self.B- self.k1 * self.V**2 - self.k2 * ca.cos(self.gdec) - self.k3 * ca.sin(self.gdec) ) 
        self.gfv = [] ; self.gfd = []
        for i in range(self.N-1):
            self.gfv.append(self.V[i+1] - (self.V[i] + self.A[i] * self.dt))
        for i in range(self.N-1):
            self.gfd.append(self.S[i+1]-(self.S[i] + self.V[i]*self.dt + 0.5*self.A[i]* self.dt**2))
        self.gd0 = self.S[0] - self.sec
        self.gv0 = self.V[0] - self.vec 
        self.g = ca.vertcat(self.gacc, self.gfa,*self.gfv,*self.gfd, self.gd0, self.gv0)

        self.nlp = {'x': ca.vertcat(self.S, self.V,self.U,self.A, self.B), 'f': self.J, 'g': self.g, 'p':self.p}
        self.opts = {
            'ipopt.print_level': 0,  # Adjust verbosity (0 to 12, with 0 being silent and 12 being very verbose)
            'print_time': True,  # Print solver execution time
            'ipopt.tol': 1e-8,  # Tolerance (adjust as needed)
        }
        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp, self.opts)
        # Initial guess and bounds
        self.init_guess = np.concatenate((self.traj_sl[:self.N]-self.dinit ,self.traj_vl[:self.N], 1.1*np.ones(self.N), 0.2*np.ones(self.N), np.zeros(self.N) ))
        # Lower bounds for v and u
        self.lb_s = -np.inf * np.ones(self.N)
        self.lb_v = np.zeros(self.N)  # v >= 0
        self.lb_u = self.umin * np.ones(self.N)  # u >= -1
        self.lb_a = -3.0 * np.ones(self.N)
        self.lb_b = self.bmin * np.ones(self.N)
        self.lbx = np.concatenate((self.lb_s, self.lb_v, self.lb_u, self.lb_a, self.lb_b))

        self.ub_s = np.inf * np.ones(self.N)
        self.ub_v = 27.* np.ones(self.N)  # v >= 0
        self.ub_u = self.umax * np.ones(self.N)  # u >= -1
        self.ub_a = 2.0 * np.ones(self.N)
        self.ub_b = self.bmax * np.ones(self.N)
        self.ubx = np.concatenate((self.ub_s, self.ub_v, self.ub_u, self.ub_a, self.ub_b))

        self.lbg = np.zeros(4*self.N)  # Lower bounds of g
        self.ubg = np.concatenate(( self.dmax * np.ones(self.N), np.zeros(3*self.N)))  # Upper bounds of g

        self.solve_time = 0.

    def solve(self):
        t1 = time.time()
        res = self.solver(x0=self.init_guess, lbx=self.lbx, ubx=self.ubx,\
                          lbg = self.lbg, ubg = self.ubg, p = self.p_value)
        self.res = np.array(res['x']).flatten()
        t2 = time.time()
        self.solve_time = t2-t1
        # S, V, U, A, B
        # at = np.array(self.res['x'][2*self.N:3*self.N]).flatten()[0]
        # av = np.array(self.res['x'][3*self.N:4*self.N]).flatten()[0]
        # ab = np.array(self.res['x'][4*self.N:5*self.N]).flatten()[0]

        at = self.res[2*self.N]
        av = self.res[3*self.N]
        ab = self.res[4*self.N]
        print("at: ", at, " av: ", av , "ab: ", ab)
        return at, av, ab

    def update(self, xc, traj_l, if_dynamic_vr = True):
        self.xc = xc
        self.traj_sl, self.traj_vl, self.traj_al = traj_l
        resx = self.get_res()
        self.init_guess = resx
        if self.use_gd_prediction:
            # print("resx[1:self.N] shape: ", resx[1:self.N].shape)
            # print("resx[-1] shape: ", resx[-1].shape)
            s_index = np.concatenate((resx[1:self.N], [resx[self.N-1]]))
            index_gd = np.round(s_index).astype(np.int)
            assert len(index_gd) == self.N
            # print("resx ", resx)
            # assert 1==2 , "check resx shape"
            # p_gd
            p_gd = self.gd_profile[index_gd]
            # print("s_index: ", s_index)
            # print("index gd: ", index_gd)
            # print("predicted gradient profile: ", p_gd)

            # assert 1==2, "check p_gd"
            self.p_value = ca.vertcat(self.traj_sl[:self.N], self.traj_vl[:self.N], xc[0], xc[1], p_gd)
        else:
            self.p_value = ca.vertcat(self.traj_sl[:self.N], self.traj_vl[:self.N], xc[0], xc[1], xc[-1])

    def get_res(self):
        # return self.res['x']
        return self.res
    def get_traj_s(self):
        return self.res[:self.N]

    def get_solve_info(self):
        return {'solve_time': self.solve_time, 'executed_its': 1}

    def set_gd_profile(self,gd_profile):
        self.gd_profile = gd_profile
        
from emato.util.util import *
# Energy-Model-Aware-Trajectory-Optimization
class NLP_J:
    def __init__(self, param, traj_l, xc):
        self.id = "NLP"
        self.nx = 2 ; self.nu = 1
        self.vr = param.vr ; self.xr = param.xr
        self.xmin = param.xmin.copy() ; self.xmax = param.xmax.copy()
        self.umin = param.umin ; self.umax = param.umax
        self.vmin = param.vmin; self.vmax = param.vmax
        self.bmin = param.bmin; self.bmax = param.bmax
        self.w1 = param.w1 ;self.w2 = param.w2 ;self.w3 = param.w3
        self.T = param.T ; self.dmin = param.dmin; self.dmax = param.dmax
        self.N = param.N ; self.dt = param.dt ; self.dinit = param.dinit
        self.param = param
        self.theta = 0.
        #  self.gd_profile = None
        _, self.gd_profile = get_gradient_profile(feature=param.gd_profile_type)
        self.use_gd_prediction = param.use_gd_prediction

        self.k1,self.k2,self.k3 = param.k
        # self.b0, self.b1, self.b2, self.b3 = param.b
        self.c0,self.c1,self.c2 = param.c
        self.o0,self.o1,self.o2,self.o3,self.o4 = param.o

        self.traj_sl, self.traj_vl, self.traj_al = traj_l
        self.xc = xc
        self.res = np.concatenate((self.traj_sl[:self.N]-self.dinit ,self.traj_vl[:self.N], np.zeros(self.N), 1.1*np.ones(self.N), 0.2*np.ones(self.N), np.zeros(self.N) ))

        self.SL = ca.MX.sym('SL', self.N)
        self.VL = ca.MX.sym('VL', self.N)
        self.sec = ca.MX.sym('sec', 1)
        self.vec = ca.MX.sym('vec', 1)
        if self.use_gd_prediction:
            self.gdec = ca.MX.sym('gdec', self.N)
        else:
            self.gdec = ca.MX.sym('gdec', 1)

        self.p = ca.vertcat(self.SL, self.VL, self.sec, self.vec, self.gdec)
        self.S = ca.MX.sym('S', self.N)
        self.U = ca.MX.sym('U', self.N)  # Control input vector
        self.V = ca.MX.sym('V', self.N)  # Velocity vector
        self.A = ca.MX.sym('A', self.N)  # Apparent Acceleration
        self.B = ca.MX.sym('B', self.N)
        self.Jerk = ca.MX.sym('Jerk', self.N)

        self.p_value = None
        self.quintic_w = (0.01, 200,0.005)
        self.fu = self.o0 + self.o1 * self.V + self.o2 * self.V**2 + \
                    self.o3 * self.V**3 +self. o4 * self.V**4 + \
                        (self.c0 + self.c1 * self.V + self.c2 * self.V**2) * self.U
        
        # self.J = ca.sum1(self.w1 * (self.V - self.VL)**2 + self.w2 * self.A**2 + self.w2*self.B**2 + self.w3 * self.fu)
        # self.J = 10 * ca.sum1(self.Jerk**2) + ca.sum1(self.fu)/ (self.S[-1] - self.S[0])
        self.J = self.w1 * ca.sum1(self.Jerk**2) +\
                 self.w2 * ca.sum1(self.A**2 + self.B**2)+ \
                 self.w3 *ca.sum1(self.fu)/ (self.S[-1] - self.S[0])\
                 + 0.01*ca.sum1(self.w1 * (self.V - self.VL)**2)
        
        """
        Constraints
        """
        # ACC constraints: [N] (length)
        # self.gacc = self.p[:self.N] - (self.S + self.T * self.V + self.dmin)
        self.gacc = self.p[:self.N] - (self.S  + self.dmin)
        # Dynamic Constraints: [N]
        self.gfa = self.A-   ( self.U -self.B- self.k1 * self.V**2 - self.k2 * ca.cos(self.gdec) - self.k3 * ca.sin(self.gdec) ) 
        self.gfv = [] ; self.gfd = []; self.gfj = []

        # V dynamic: [N-1]
        for i in range(self.N-1):
            self.gfv.append(self.V[i+1] - (self.V[i] + self.A[i] * self.dt))
        # A dynamic: [N-1]
        for i in range(self.N-1):
            self.gfd.append(self.S[i+1]-(self.S[i] + self.V[i]*self.dt + 0.5*self.A[i]* self.dt**2))
        # Jerk dynamic: [N-1]
        for i in range(self.N-1):
            self.gfj.append(self.A[i+1]-(self.A[i]+self.Jerk[i]*self.dt))
        
        # Boundary Value: [2]
        self.gd0 = self.S[0] - self.sec
        self.gv0 = self.V[0] - self.vec 

        """
        Concat all the defined constraints
        """
        self.g = ca.vertcat(self.gacc, self.gfa,*self.gfv,*self.gfd, *self.gfj, self.gd0, self.gv0)

        self.nlp = {'x': ca.vertcat(self.S, self.V,self.U,self.Jerk, self.A, self.B), 'f': self.J, 'g': self.g, 'p':self.p}
        self.opts = {
            'ipopt.print_level': 0,  # Adjust verbosity (0 to 12, with 0 being silent and 12 being very verbose)
            'print_time': True,  # Print solver execution time
            'ipopt.tol': 1e-8,  # Tolerance (adjust as needed)
        }
        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp, self.opts)
        # Initial guess and bounds
        self.init_guess = np.concatenate((self.traj_sl[:self.N]-self.dinit,\
                                          self.traj_vl[:self.N],\
                                          1.1*np.ones(self.N),\
                                            np.zeros(self.N), \
                                            0.2*np.ones(self.N),\
                                            np.zeros(self.N) ))
        print(" Guess length: ", len(self.init_guess))
        # Lower bounds for v and u
        self.lb_s = -np.inf * np.ones(self.N)
        self.lb_v = np.zeros(self.N)  # v >= 0
        self.lb_u = self.umin * np.ones(self.N)  # u >= -1
        self.lb_jerk = - self.param.jerkmax * np.ones(self.N)
        self.lb_a = -3.0 * np.ones(self.N)
        self.lb_b = self.bmin * np.ones(self.N)
        self.lbx = np.concatenate((self.lb_s, self.lb_v, self.lb_u, self.lb_jerk, self.lb_a, self.lb_b))

        self.ub_s = np.inf * np.ones(self.N)
        self.ub_v = 27.* np.ones(self.N)  # v >= 0
        self.ub_u = self.umax * np.ones(self.N)  # u >= -1
        self.ub_jerk =  self.param.jerkmax * np.ones(self.N)
        self.ub_a = 2.0 * np.ones(self.N)
        self.ub_b = self.bmax * np.ones(self.N)
        self.ubx = np.concatenate((self.ub_s, self.ub_v, self.ub_u, self.ub_jerk,self.ub_a, self.ub_b))

        self.lbg = np.zeros(5*self.N-1)  # Lower bounds of g
        self.ubg = np.concatenate(( self.dmax * np.ones(self.N), np.zeros(4*self.N-1)))  # Upper bounds of g
        assert len(self.lbg) == len(self.ubg), "wrong length for box constraints" 
        self.solve_time = 0.

    # s, v, u, jerk, a, b  where u:= at, a:= av
    def solve(self):
        t1 = time.time()
        assert len(self.init_guess) == 300 , " Guess wrong len: "+str(len(self.init_guess))
        res = self.solver(x0=self.init_guess, lbx=self.lbx, ubx=self.ubx,\
                          lbg = self.lbg, ubg = self.ubg, p = self.p_value)
        self.res = np.array(res['x']).flatten()
        t2 = time.time()
        self.solve_time = t2-t1


        at = self.res[2*self.N]
        jerk = self.res[3*self.N]
        av = self.res[4*self.N]
        ab = self.res[5*self.N]
        print("at: ", at, " av: ", av , "ab: ", ab)
        return at, av, ab, jerk

    def update(self, xc, traj_l, if_dynamic_vr = True):
        self.xc = xc
        self.traj_sl, self.traj_vl, self.traj_al = traj_l
        resx = self.get_res()
        self.init_guess = resx
        if self.use_gd_prediction:
            # print("resx[1:self.N] shape: ", resx[1:self.N].shape)
            # print("resx[-1] shape: ", resx[-1].shape)
            s_index = np.concatenate((resx[1:self.N], [resx[self.N-1]]))
            index_gd = np.round(s_index).astype(np.int)
            assert len(index_gd) == self.N
            # print("resx ", resx)
            # assert 1==2 , "check resx shape"
            # p_gd
            p_gd = self.gd_profile[index_gd]
            # print("s_index: ", s_index)
            # print("index gd: ", index_gd)
            # print("predicted gradient profile: ", p_gd)

            # assert 1==2, "check p_gd"
            self.p_value = ca.vertcat(self.traj_sl[:self.N], self.traj_vl[:self.N], xc[0], xc[1], p_gd)
        else:
            self.p_value = ca.vertcat(self.traj_sl[:self.N], self.traj_vl[:self.N], xc[0], xc[1], xc[-1])

    def get_res(self):
        # return self.res['x']
        return self.res

    def get_traj_s(self):
        return self.res[:self.N]

    def get_solve_info(self):
        return {'solve_time': self.solve_time, 'executed_its': 1}

    def set_gd_profile(self,gd_profile):
        self.gd_profile = gd_profile


