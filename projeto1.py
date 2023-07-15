import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import textwrap

def set_contours_for_square(x_points, y_points, total_points):
    cc1 = [i for i in range(x_points)]
    cc2 = [i for i in range(2*x_points-1,x_points*y_points-1,x_points)]
    cc3 = [i for i in range(total_points-x_points, total_points)]
    cc4 = [i for i in range(x_points, (y_points-1)*x_points, x_points)]
    print( [i for i in range(x_points, (y_points-1)*x_points, x_points)])
    return cc1, cc2, cc3, cc4

Lx = 1.0
Ly = 1.0
nx = ny = 10
npoints = nx*ny
dx = Lx/(nx-1)
dy = Ly/(ny-1)

Q = 20.0
rho = 1.0
cv = 1.0
k = 1.0
dt = 0.02

alpha = k/(rho*cv)

Xv = np.linspace(0, Lx, nx)
Yv = np.linspace(0, Ly, ny)

X,Y = np.meshgrid(Xv, Yv)
X = np.reshape(X, npoints)
Y = np.reshape(Y, npoints)

cc1, cc2, cc3, cc4= set_contours_for_square(nx, ny, npoints)

points_of_contours = cc1 + cc2 + cc3 + cc4

all_points = [i for i in range(npoints)]

inner_points = list(set(all_points)-set(points_of_contours))

T_contours = np.zeros((npoints), dtype='float')

for i in cc1:
    T_contours[i] += X[i]*X[i]
for j in cc2:
    T_contours[j] += Y[j]*Y[j]*Y[j] - 2.0
for k in cc3:
    T_contours[k] += X[k]*X[k] + 3.0
for l in cc4:
    T_contours[l] += Y[l] + 1.0

A = np.zeros((npoints, npoints), dtype='float')
b = np.zeros((npoints), dtype='float')

for i in points_of_contours:
    A[i,i] = 1.0
    b[i] = T_contours[i]

for i in inner_points:
    A[i,i+1] = (-alpha)/(dx*dx)
    A[i, i] = 1/dt +(2*alpha)/(dx*dx) + (2*alpha)/(dy*dy)
    A[i, i-1] = (-alpha)/(dx*dx)
    A[i, i+nx] = (-alpha)/(dy*dy)
    A[i, i-ny] = (-alpha)/(dy*dy)

T = np.zeros(npoints, dtype='float')

for point in points_of_contours:
    T[point] = T_contours[point]

Qvec = (Q/(rho*cv))*np.ones((npoints),dtype='float')

matplot_font = {'fontname':'Roboto', 'color':'black', 'weight':'normal', 'size':12}
t =  "Condições do problema: Q = " + str(Q) + " ; rho = " + str( rho) + \
" ; cv = " + str(cv) + " ; k = " + str(k) + " ; alpha = " + str(alpha)+" ; dt="+str(dt)+\
" ; Lx = " + str(Lx) + " ; Ly = " + str(Ly) + " ; nx = " + str(nx) + " ; ny = " + str(ny)
tt = textwrap.fill(t, width=70)

for t in range(0,10):
    for point in inner_points:
        b[point] = T[point]/dt + Qvec[point]
    T = np.linalg.solve(A, b)
    Z = T.reshape(ny, nx)
    surf = plt.imshow( Z, interpolation='quadric', origin='lower', cmap=cm.jet, extent=(X.min(), X.max(), Y.min(), Y.max()))
    cbar = plt.colorbar(surf ,shrink=1.0, aspect=20)
    tempo = t*dt

    cbar.set_label('Temperatura [°C]',fontdict=matplot_font)
    plt.title("Gráfico" + " " + str(t) + " " + "corresponde a " +
    str(tempo) + " segundos",fontdict=matplot_font)
    plt.ylabel("Posição em y da placa",fontdict=matplot_font)
    plt.xlabel("Posição em x da placa",fontdict=matplot_font)

    plt.grid(color='black', linestyle='solid', linewidth=0.5)
    labx = np.linspace(X.min(),X.max(),nx)
    laby = np.linspace(Y.min(),Y.max(),ny)
    plt.xticks(labx)
    plt.yticks(laby)
    plt.gcf().autofmt_xdate()

    plt.text(0.5, -0.3, tt, ha='center', va='top',fontdict= matplot_font )

    plt.show()
