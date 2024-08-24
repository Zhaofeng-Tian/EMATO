from emato.obj.poly.quintic import QuinticPolynomial, Traj
import numpy as np
from emato.util.util import *

class FrenetTraj:
    def __init__(self,road, k, o, c):
        self.road = road
        # all trajectory, not number
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.clon= 0.0
        self.clat = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.dx = []
        self.dy = []
        self.yaw = []
        # arc coordinate not lon
        self.dl = []
        self.l = []
        self.v = []
        self.a = []
        self.jerk = []
        self.k = []; self.k2 = []

        self.dfl = []
        self.fl = []
        self.fv = []
        self.fa = []
        self.fjerk = []
        self.fyaw = []
        self.fk = []; self.fk2 = []

        self.if_sd_collision = True
        self.if_xy_collision = True
        self.if_over_curvy = False
        self.if_over_jerky = False

        self.r_collision = np.inf
        self.r_curvature = np.inf
        self.r_jerk = np.inf
        self.r_invalid = None
        self.r_v = None
        self.r_d = None
        self.r_fe = None
        self.r_complex1 = None
        self.r_complex2 = None
        self.r_complex3 = None

        self.k,self.o,self.c = k,o,c
        
        
 

    def setup(self,xc,xg,road,ptime,dt, gd_profile):
                   # (xs, vxs, axs, xe, vxe, axe, time,dt):
            cs,cs_d,cs_dd,cd,cd_d,cd_dd = xc
            gs,gs_d,gs_dd,gd, gd_d, gd_dd = xg
            Tlat = QuinticPolynomial(cd, cd_d, cd_dd, gd, gd_d,gd_dd, ptime, dt)
            Tlon = QuinticPolynomial(cs,cs_d,cs_dd, gs, gs_d,gs_dd, ptime, dt)
            self.t, self.d, self.d_d, self.d_dd, self.d_ddd = Tlat.get_traj()
            _, self.s, self.s_d, self.s_dd, self.s_ddd = Tlon.get_traj()
            # assert len(self.s) == 50, " len is not 50!! len is " + str(len(self.s))



            # ************* Based on x, y
            self.x, self.y, _ = road.frenet_to_global(self.s, self.d)
            self.dx = np.diff(self.x); self.dx = np.append(self.dx, self.dx[-1])
            self.dy = np.diff(self.y); self.dy = np.append(self.dy, self.dy[-1])
            self.yaw = np.arctan2(self.dy, self.dx)
            self.dl = np.hypot(self.dx,self.dy)
            self.l = np.cumsum(self.dl)
            # Calculate velocity along the arc
            self.v = self.dl / dt
            # Calculate acceleration along the arc
            self.a = np.diff(self.v) / dt
            self.a = np.append(self.a, self.a[-1])  # Extend to match length
            # Calculate jerk along the arc
            self.jerk = np.diff(self.a) / dt
            self.jerk = np.append(self.jerk, self.jerk[-1])  # Extend to match length
            ddx = np.diff(self.dx); ddx = np.append(ddx, ddx[-1])
            ddy = np.diff(self.dy); ddy = np.append(ddy, ddy[-1])
            diff_yaw = np.diff(self.yaw); diff_yaw = np.append(diff_yaw, diff_yaw[-1])
            self.cur = diff_yaw/self.dl
            self.cur2 = np.abs(self.dx * ddy - self.dy * ddx) / (self.dx**2 + self.dy**2)**(3/2)

            # *********** Based on s, d
            self.dfl = np.hypot(self.d_d*dt, self.s_d*dt)
            self.fl = np.cumsum(self.dfl)
            self.fv = np.hypot(self.s_d, self.d_d)
            self.fa = np.hypot(self.s_dd, self.d_dd)
            self.fjerk = np.hypot(self.s_ddd, self.d_ddd)
            self.fyaw =  np.arctan2(self.s_d*dt, self.d_d*dt)

            diff_fyaw = np.diff(self.fyaw); diff_yaw = np.append(diff_fyaw, diff_fyaw[-1])
            self.fcur = diff_yaw/self.dfl
            self.fcur2 = np.abs(self.s_dd * self.d_d - self.s_d * self.d_dd) / (self.s_d**2 + self.d_d**2)**(3/2)


            self.theta = get_traj_theta(gd_profile, self.s)
            self.sin = np.sin(self.theta)
            self.cos = np.cos(self.theta)

            self.ar = self.k[0]*self.v**2 + self.k[1]*self.cos + self.k[2]*self.sin
            atemp = self.a + self.ar
            self.at = np.where(atemp<0,0,atemp)
            self.ab = self.at-self.a-self.ar
            self.fr = self.o[0] + self.o[1]*self.v + self.o[2]*self.v**2+self.o[3]* self.v**3 + self.o[4]*self.v**2+(self.c[0]+self.c[1]*self.v + self.c[2]*self.v**2)*self.at
            self.fc = np.cumsum(self.fr*dt) # this is total consumption of the traj
            self.fe_ds = self.fr *dt / self.s_d * dt
            self.fe = self.fr*dt / self.dl

            self.far = self.k[0]*self.fv**2 + self.k[1]*self.cos + self.k[2]*self.sin
            atemp = self.fa + self.far
            self.fat = np.where(atemp<0,0,atemp)
            self.fab = self.fat-self.fa-self.far
            self.ffr = self.o[0] + self.o[1]*self.fv + self.o[2]*self.fv**2+self.o[3]* self.fv**3 + self.o[4]*self.fv**2+(self.c[0]+self.c[1]*self.fv + self.c[2]*self.fv**2)*self.fat
            self.ffc = np.cumsum(self.ffr*dt)
            self.ffe_ds = self.ffr*dt / self.s_d * dt 
            self.ffe =  self.ffr*dt / self.dfl  # ml/m  to L/100 km

            # 
            # print("x, y: ", self.x, self.y)
            # print("dx, dy", self.dx, self.dy)
            # print("yaw", self.yaw)
            # print("dl:", self.dl)
            # print("l",self.l)
            # print("v: ", self.v)
            # print("a: ", self.a)
            # print("jerk: ", self.jerk)
            # print("k: ", self.k)
            # print("k2: ", self.k2)
            # print("fr: ", self.fr)
            # print("fc: ", self.fc)
            # print("fe_ds", self.fe_ds)
            # print("fe: ", self.fe)

            # # 
            # print("x, y: ", self.x, self.y)
            # print("dx, dy", self.dx, self.dy)
            # print("yaw", self.yaw)
            # print("dl:", self.dl)
            # print("l",self.l)
            # print("v: ", self.v)
            # print("a: ", self.a)
            # print("jerk: ", self.jerk)
            # print("k: ", self.k)
            # print("k2: ", self.k2)
            # print("fr: ", self.fr)
            # print("fc: ", self.fc)
            # print("fe_ds", self.fe_ds)
            # print("fe: ", self.fe)

    
    def calc_reward(self, traj_traffic_list, rsafe, jerkmax, kmax, desired_v, desired_d):
        """
        This is function only calculates the basic reward component for frenet algorithm,
        for more complex objective settings, alg.frenet.py will handle them.
        """

        """
        Collision
        """
        self.if_sd_collision, self.if_xy_collision= self.check_collision(traj_traffic_list, rsafe)
        if self.if_xy_collision:
            self.r_collision = np.inf
        else:
            self.r_collision = 0


        """
        Curvature
        """
        if any(abs(self.cur) > 3) :
            self.if_over_curvy = True
            self.r_curvature = np.inf
        else:
            self.if_over_curvy = False
            self.r_curvature = 0


        """
        Jerk
        """
        self.flon_jerk = np.sum(self.s_ddd**2) 
        self.flat_jerk = np.sum(self.d_ddd**2) 
        if np.max(self.flon_jerk + self.flat_jerk) > 10**2:
            self.if_over_jerky = True
            self.r_jerk = np.inf
        else:
            self.if_over_jerky = False
            self.r_jerk = self.flon_jerk + self.flat_jerk

        """
        End state: Lon s_d (longitute v) and Lat d
        """
        self.r_v = (self.s_d[-1] - desired_v)**2
        self.r_d = (self.d[-1] - desired_d)**2

        """
        Fuel efficiency:
        fuel consumption of the trajectory / traveled s (not arc length but solely length on s)
        """
        self.r_fe = self.fc[-1] / (self.s[-1] - self.s[0])

        self.r_invalid = self.r_collision+ self.r_curvature +self.r_jerk

        self.r_complex1 = self.r_jerk + self.r_v + self.r_d

        self.r_complex2 = self.r_jerk + self.r_v + self.r_d + 100* self.r_fe 
        
        # assert len(self.r_fe) == 1, "rfe len"
        # assert len(self.r_complex2) == 1, "length wrong" 
        
        self.r_complex3 = self.r_fe



    def check_collision(self,traj_traffic_list, rsafe):
        if len(traj_traffic_list) == 0:
            return(False, False)

        else:

            sd_count = 0
            xy_count = 0
            for traj in traj_traffic_list:
                traffic_s, traffic_d = traj[:,0],traj[:,1]
                traffic_x, traffic_y, _ = self.road.frenet_to_global(traffic_s, traffic_d)
                sd_dist_square = (traffic_s - self.s)**2 + (traffic_d - self.d)**2
                xy_dist_square = (traffic_x - self.x)**2 + (traffic_y - self.y)**2

                # print("sd_dist_squre: ", sd_dist_square)
                # print("xy_dist_squre: ", xy_dist_square)

                if np.min(sd_dist_square) < rsafe**2:
                    sd_count += 1
                    # print("sd collsion!!!!")


                if np.min(xy_dist_square) < rsafe**2:
                    xy_count += 1

            return(sd_count>0, xy_count>0)
            
