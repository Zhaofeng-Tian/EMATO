import numpy as np
import matplotlib.pyplot as plt 

class Recorder:
    def __init__(self):
        self.Tt = []
        # Leading car's trajectory
        self.Tsl = [] # s coords leading car 
        self.Tvl = [] # v profile leading car
        self.Tavl = [] # apparent acceleration leading car
        self.Tatl = [] # traction acceleration leading car
        self.Tarl = []
        self.Tfrl = [] # fuel rate leading car ml/s
        self.Tfcl = [] # fuel consumption leading car ml
        self.Tthetal = []
    
        # Ego car's trajectory
        self.Tse = [] # s coords ego car 
        self.Tve = [] # v profile ego car
        self.Tave = [] # apparent acceleration ego car
        self.Tate = [] # traction acceleration ego car
        self.Tare = []
        self.Tfre = [] # fuel rate ego car ml/s
        self.Tfce = [] # fuel consumption ego car ml
        self.Tthetae = []

        self.Tst = []
        self.Teit = []
    
    def record(self, state_l, state_e, t):
        sl,vl,avl,atl,arl,frl,fcl,thetal = state_l
        se,ve,ave,ate,are,fre,fce,thetae = state_e

        self.Tt.append(t)
        self.Tsl.append(sl)
        self.Tvl.append(vl)
        self.Tavl.append(avl)
        self.Tatl.append(atl)
        self.Tarl.append(arl)
        self.Tfrl.append(frl)
        self.Tfcl.append(fcl)
        self.Tthetal.append(thetal)

        self.Tse.append(se)
        self.Tve.append(ve)
        self.Tave.append(ave)
        self.Tate.append(ate)
        self.Tare.append(are)
        self.Tfre.append(fre)
        self.Tfce.append(fce)
        self.Tthetae.append(thetae)

    def record_solve_info(self, info):
        self.Tst.append(info['solve_time'])
        self.Teit.append(info['executed_its'])


    def plot_trajectory(self):
        plt.figure(figsize=(12, 8))

        # Plotting
        plt.subplot(4, 2, 1)
        plt.plot(self.Tt, self.Tsl, label="Leader's Position")
        plt.plot(self.Tt, self.Tse, label="Ego's Position")
        plt.ylabel('Position')
        plt.legend()

        plt.subplot(4, 2, 2)
        plt.plot(self.Tt, self.Tvl, label="Leader's Velocity")
        plt.plot(self.Tt, self.Tve, label="Ego's Velocity")
        plt.ylabel('Velocity')
        plt.legend()

        plt.subplot(4, 2, 3)
        plt.plot(self.Tt, self.Tavl, label="Leader's Apparent Acceleration")
        plt.plot(self.Tt, self.Tave, label="Ego's Apparent Acceleration")
        plt.ylabel('Acceleration')
        plt.legend()

        plt.subplot(4, 2, 4)
        plt.plot(self.Tt, self.Tate, label="Ego's Traction Acceleration")
        plt.plot(self.Tt, self.Tatl, label="Leading's Traction Acceleration")
        plt.ylabel('Actual Acceleration')
        plt.legend()

        plt.subplot(4, 2, 5)
        plt.plot(self.Tt, self.Tfre, label="Ego Fuel Rate")
        plt.plot(self.Tt, self.Tfrl, label="Leading Fuel Rate")
        plt.ylabel('Fuel Rate')
        plt.legend()

        plt.subplot(4, 2, 6)
        plt.plot(self.Tt, self.Tfce, label="Ego Fuel Consumption")
        plt.plot(self.Tt, self.Tfcl, label="Leading Fuel Consumption")
        plt.ylabel('Total Fuel Consumption')
        plt.legend()

        plt.subplot(4, 2, 7)
        plt.plot(self.Tt, self.Tare, label="Ego Resistance Acceleration")
        plt.plot(self.Tt, self.Tarl, label="Leading Resistance Acceleration")
        plt.ylabel('Resistance Acceleration')
        
        plt.subplot(4, 2, 8)
        plt.plot(self.Tt, self.Tthetae, label="Ego Gradient")
        plt.plot(self.Tt, self.Tthetal, label="Leading Gradient")
        plt.ylabel('Gradient')

        plt.tight_layout()
        plt.show()        