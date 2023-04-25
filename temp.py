from PDEs import Grid,BoundaryCondition,diffusion_solver
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go

""" FINITE DIFFERENCE PDEs """

# Weeks 20,21,22

grid = Grid(N=10, a=0, b=1)
bc_left = BoundaryCondition('dirichlet', [lambda t: 0], grid)
bc_right = BoundaryCondition('dirichlet', [lambda t: 0], grid)
t_steps = 1000
x = grid.x

result = diffusion_solver(grid,
                    bc_left,
                    bc_right,
                    IC = 0,
                    D=0.1,
                    q=1,
                    dt=0.1,
                    t_steps=t_steps,
                    method='explicit-euler',
                    storage = 'sparse')

u = result.u
x = result.x
t_span = result.t

""" ANIMATING SOLUTION """


fig,ax = plt.subplots()

line, = ax.plot(x,u[:,0])
ax.set_ylim(0,10)
ax.set_xlim(grid.left,grid.right)

ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('Diffusion equation')

def animate(i):
    line.set_data((x,u[:,i]))
    return line,

ani = FuncAnimation(fig, animate, frames=t_steps, interval=100, blit=True)
plt.show()

""" PLOT SOLUTION AS 3D SURFACE """

fig = go.Figure(data=[go.Surface(z=u, x=t_span, y=x)])

fig.update_layout(
    title='u(x,t)',
    autosize=False,
    scene=dict(
        xaxis=dict(range=[0, 5]),
        xaxis_title='t',
        yaxis_title='x',
        zaxis_title='u(x, t)'),
    width=500,
    height=500,
    margin=dict(l=65, r=50, b=65, t=90)
)

fig.show()