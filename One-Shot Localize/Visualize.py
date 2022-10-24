import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import math

"""
Visualization file:
Saved results in this file, was also used to visualize the results
"""


#Plot from older data
if False:
       x = np.linspace(0, 1.5, 7)
       y = [0.0014, 0.0307, 0.0771, 0.1146, 0.1790, 0.2211, 0.2836]

       # plot
       fig, ax = plt.subplots()

       ax.plot(x, y, linewidth=2.0)

       ax.set(xlim=(0, 1.5), xticks=np.arange(0, 1.75, 0.25),
              ylim=(0, 0.3), yticks=np.arange(0, 0.35, 0.05))

       ax.set_ylabel('MSE')
       ax.set_xlabel('Standard deviation')

       plt.show()

#Plot with saved result values
if True:
       x = [5730**0.5, 1146**0.5, 573**0.5, 400**0.5, 114.6**0.5, 57.3**0.5, 0.573**0.5]
       #y_10dbprio = [4.512, 4.294, 4.505, 4.512]
       #y_20 = [-4.0644, -4.3438, -5.9533, -6.2976]
       y_m5p3 = [-5.7698, -5.7779, -6.2206, -6.8968, -8.2276, -8.5840, -9.4185]
       #y_20_07 = [-5.7845, -5.7811, -6.0304, , -8.3032, -8.6706, -9.5352]

       y_0t = [-3.7847, -4.4317, -4.7667, -4.9667, -6.0334, -6.4002, -8.2421]
       y_10t = [-21.6206, -21.8206, -21.6206, -21.54, -21.6206, -21.7446, -21.7446]

       #y_15 = [4.36, -3.69, -10.16]
       y_4p6 = [3.2115, 2.6927, 1.2776, 0.3037, -1.3027, -2.0846, -4.8618]
       #y_10_07 = [3.2115, 2.6927, 1.2776, , -1.3027, -2.0473, -4.2552]
       #y_5 = [3.44, -4.76, -11.59]
       n_4p6 = [4.6347, 4.6347, 4.6347, 4.6347, 4.6347, 4.6347, 4.6347]
       n_m5p3 = [-5.2838, -5.2838, -5.2838, -5.2838, -5.2838, -5.2838, -5.2838]
       n_0t = [-2.7227, -2.7227, -2.7227, -2.7227, -2.7227, -2.7227, -2.7227]
       n_10t = [-12.7227, -12.7227, -12.7227, -12.7227, -12.7227, -12.7227, -12.7227]

       # plot
       fig, ax = plt.subplots()
       plt.title('MSE of Data-driven approach')
       ax.semilogx()
       ax.plot(x, n_4p6, linewidth=1.0, label="4.6dB Noise prior floor", linestyle='dashed', color='red')
       ax.plot(x, y_4p6, linewidth=1.0, label="4.6dB Noise prior", color='red', marker='o')
       #ax.plot(x, y_20_07, linewidth=1.0, label=f"-20dB Noise prior, {chr(945)} = 0.7 ", color='lime', marker='o')
       ax.plot(x, y_0t, linewidth=1.0, label="-2.7dB Noise prior", color='purple', marker='o')
       ax.plot(x, n_0t, linewidth=1.0, label="-2.7dB Noise prior", color='purple', linestyle='dashed', marker='o')
       ax.plot(x, n_m5p3, linewidth=1.0, label="-5.3dB Noise prior floor", linestyle='dashed', color='blue')
       ax.plot(x, y_m5p3, linewidth=1.0, label="-5.3dB Noise prior", color='blue', marker='o')
       #ax.plot(x, y_10t, linewidth=1.0, label="-12.7dB Noise prior", color='green', marker='o')
       #ax.plot(x, n_10t, linewidth=1.0, label="-12.7dB Noise prior", color='green', linestyle='dashed', marker='o')


       ax.set(xlim=(0.573**0.5, 5731**0.5), xticks=[0.573**0.5, 57.3**0.5, 400**0.5, 1146**0.5, 5730**0.5], xticklabels=[0.573**0.5, 57.3**0.5, 400**0.5, 1146**0.5, 5730**0.5],
              ylim=(-10, 6), yticks=[-10, -8, -6, -4, -2, 0, 2, 4, 6])
       #, 573**0.5 , 573**0.5, 573**0.5
       plt.gca().invert_xaxis()
       ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
       ax.set_ylabel('MSE [dB]')
       ax.set_xlabel('Observation standard deviation [degrees]')
       #ax.set_xlabel(r'$\frac{1}{r^2}$ [dB]')
       plt.legend()
       plt.savefig("Results Encoder.png", dpi=1800)
       plt.show()