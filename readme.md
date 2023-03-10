### **Koopman Operator Notes**

This repository serves as the testing ground for my research at Boston University (being performed with the intent of being published). This readme currently discusses noteworthy models being evaluated and the results of their tests with the Koopman control equation (KCE). It also serves as a testing ground for my ideas on gradient flow mapping, etc.


___
### **Notes on Coordinate Descent**

*Coordinate descent* is a method for iteratively minimizing the cost of multiple objective functions with overlapping parameters. Similar to gradient descent, the algorithm is exited when the change in objective function is zero, or falls below some tolerance. We will be working under the assumption that the objective function is continuously differentiable and its gradient can be found through the FDM calculation method (to be developed more in the future).


___
### **Linear System:**
**State/Model Selection:** linear/second-order/discrete

$$
    x = \left[ \begin{matrix}
        x \\
        y \\
        \dot x \\
        \dot y
    \end{matrix} \right]
$$

With the model equation written in terms of linear matrices A and B.

$$
\begin{matrix}
    x^+ = Ax + Bu \\
    \textrm{where } u = C(x_g - x)
\end{matrix}
$$

Where $x_g$ is a goal position and

$$
\begin{aligned}
    A = \begin{bmatrix}
        1 & 0 & \Delta t & 0 \\
        0 & 1 & 0 & \Delta t \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 1 \\
    \end{bmatrix}
&&
    B = \begin{bmatrix}
        0 & 0 \\
        0 & 0 \\
        \Delta t & 0 \\
        0 & \Delta t \\
    \end{bmatrix}
&&
    C = \begin{bmatrix}
        10 & 0 & 2.5 & 0 \\
        0 & 10 & 0 & 2.5
    \end{bmatrix}
\end{aligned}
$$

**Observation Functions Structure:**

It is important to note that because the system is already linear, and the control policy is linearly dependent on the states, the state and policy functions can reside solely in the $h$ observation list.

$$
\begin{aligned}
    \Psi_x = x
    &&
    \Psi_u = 1
    &&
    h = \begin{bmatrix}
        x \\
        u
    \end{bmatrix}
\end{aligned}
$$

In other words, for a wholistically linear system, we can represent the model in terms of solely $h$. We could also write is as $x,u \in h^+ = \mathcal{K}_u h$.

The Koopman operator for this was formed using the KCE and resulted in good behavior...

<p align="center">
    <img src=./.figures/point.png width=450 />
</p>


___
### **Anchor System:**
**State/model equations:** linear/holonomic/first-order/discrete

$$
x = \begin{bmatrix}
    x \\
    y
\end{bmatrix}
= \begin{bmatrix}
    x_1 \\
    x_2
\end{bmatrix}
$$

and the system is modeled as...

$$
\begin{matrix}
    x^+ = Ax + Bu \\
    \textrm{where } u = C(x_g - x)
\end{matrix}
$$

The interesting component introduced here is the idea that the state is not observed directly, but instead observed through *anchors*. For practical purposes we assume there are four randomly distrubuted anchors which, when called, give the object its respective $L_2$-norm squared distance. That is,

$$
    d_i(x) = (x - a_i)^\intercal (x - a_i)
$$

Where $d_i$ references the distance from the anchor, $a_i$, located at index $i$. The goal here is to utilize the EDMD learning structure to find a correlation between the anchor distances and the control policy.

**Observation Function Structure:**

Here the observables are listed as

$$
\begin{aligned}
    \Psi = \begin{bmatrix}
        x \\
        u \\
        \texttt{vec}(x x^\intercal) \\
        \texttt{vec}(u u^\intercal) \\
        \texttt{vec}(x u^\intercal)
    \end{bmatrix}
    &&
    h = \begin{bmatrix}
        \Psi \\
        d_1(x) \\
        \vdots \\
        d_{N_a} \\
        \texttt{vec}(x a_i^\intercal) \\
        \vdots \\
        \texttt{vec}(x a_{N_a}^\intercal)
    \end{bmatrix}
\end{aligned}
$$

In general we work under the assumption that the initial state of the system is known. That said, the $h$ observation list ideally updates this value as time progresses using predictions from the anchor values.


___
### **Differential Drive System:**
**State/Model Selection:** nonlinear/first-order/discrete

$$
    x = \left[ \begin{matrix}
        x \\
        y \\
        \theta
    \end{matrix} \right]
$$

$$
    \dot{x} = \left[ \begin{matrix}
        \dot x_1 \\
        \dot x_2 \\
        \dot x_3
    \end{matrix} \right]
    = \left[ \begin{matrix}
        \dot{x} \\
        \dot{y} \\
        \dot{\theta}
    \end{matrix} \right]
    = \left[ \begin{matrix}
        \textrm{cos} (x_3) (u_1 + u_2) \\
        \textrm{sin} (x_3) (u_1 + u_2) \\
        \frac{1}{R} (u_1 + u_2)
    \end{matrix} \right]
$$

This can then be converted to the discrete function form using a sufficiently small step-size parameter, $\Delta t$.

$$
    x^+ = x + \Delta t \cdot \dot{x}
$$

$$
    \begin{bmatrix}
        f_1(x_1) \\
        f_2(x_2) \\
        f_3(x_3)
    \end{bmatrix}
    =
    \begin{bmatrix}
        x_1^+ \\
        x_2^+ \\
        x_3^+
    \end{bmatrix}
    =
    \begin{bmatrix}
        x_1 \\
        x_2 \\
        x_3
    \end{bmatrix}
    + \Delta t
    \begin{bmatrix}
        \cos(x_3) (u_1 + u_2) \\ 
        \sin(x_3) (u_1 + u_2) \\
        \frac{1}{R} (u_1 + u_2)
    \end{bmatrix}
$$

**Observation Functions Structure:**

It is important to note that for this system, the $cos(x_3)$ and $sin(x_3)$ terms must be included in $\Psi_u$ because of their bilinear combination in the model equations. For this reason, the measured observation list, $h$ must contain a method for propagating them forward as well as computing the control.

$$
\begin{aligned}
    \Psi_x = x
    &&
    \Psi_u = \left[ \begin{matrix} 
        \cos(x_3) \\
        \sin(x_3) \\
        1
    \end{matrix} \right]
    &&
    h = \left[ \begin{matrix} 
        1 \\
        u_1 \\
        u_2 
    \end{matrix} \right]
\end{aligned}
$$

Because the $\cos$ and $\sin$ terms are contained in the $\Psi_u$ observables, they must be propagated when moving from $\Psi$ to $\Psi^+$. It may be possible to use a linearization of the two trigonometric functions.

$$
\begin{aligned}
    & f_3(x_3) = x_3 + \Delta t \frac{1}{R}(u_1 + u_2) \\
    & \cos(f_3(x_3^+)) = \cos(x_3 + \Delta t \frac{1}{R}(u_1 + u_2)) \\
    & L(\cos(f_3(x_3))) = L(\cos(x_3^+)) = \cos(x_3) + \left( \frac{d}{dx} \cos(x_3) \right) (x_3^+ - x_3)\\
    & L_3(\cos(x_3^+)) = \cos(x_3) - \sin(x_3) (x_3^+ - x_3) \\
    & L_3(\cos(x_3^+)) = \cos(x_3) - \sin(x_3) (x_3 + \Delta t \frac{1}{R}(u_1 + u_2) - x_3) \\
    & L_3(\cos(x_3^+)) = \cos(x_3) - \sin(x_3) (\Delta t \frac{1}{R}(u_1 + u_2)) \\
    \\
    & \cos(x_3^+) \approx \cos(x_3) - \sin(x_3) (\Delta t \frac{1}{R}(u_1 + u_2))
\end{aligned}
$$

And through similar methods we can also show that...

$$
    \sin(x_3^+) \approx \sin(x_3) - \cos(x_3) (\Delta t \frac{1}{R}(u_1 + u_2))
$$

Meaning that for a small enough time-step, $\Delta t$, the function $\cos(x_3)$ can be propagated using a bilinear combination with the model equation $f_3$.

<p align="center">
    <img src=./.figures/donald.png width=400 />
    <img src=./.figures/donaldError.png width=400 />
</p>

To propagate the function $\cos(x_3) u_1$ we can explore similar series of steps.

$$
\begin{aligned}
    & \cos(x_3^+) u_1 = \cos(x_3 + \Delta t \frac{1}{R}(u_1 + u_2)) u_1 \\
    & L( \cos(x_3^+) u_1 ) = \cos(x_3) u_1 - \sin(x_3) u_1 (x_3 + \Delta t \frac{1}{R}(u_1 + u_2) - x_3) \\
    & L( \cos(x_3^+) u_1 ) = \cos(x_3) u_1 - \sin(x_3) u_1 (\Delta t \frac{1}{R}(u_1 + u_2)) \\
    & L( \cos(x_3^+) u_1 ) = \cos(x_3) u_1 - \Delta t \frac{1}{R} \sin(x_3) ( u_1^2 + u_1u_2 )
\end{aligned}
$$

Which shows that the bilinear term $\sin(x_3) u_i$ would require infinitely expansions to achieve an accuracy which falls below the acceptable tolerance. To avoid this issue, we can exploit the $h$ observation list and manually update the linearized terms along with the propagation of the full set of observations. In other words our observable lists become...

$$
\begin{aligned}
    \Psi_x = x
    &&
    \Psi_u = \left[ \begin{matrix} 
        \cos(x_3) \\
        \sin(x_3) \\
        1
    \end{matrix} \right]
    &&
    h = \left[ \begin{matrix} 
        1 \\
        \cos(x_3) \\
        \sin(x_3) \\
        u_1 \\
        u_2 
    \end{matrix} \right]
\end{aligned}
$$

Which facilitates a high accuracy for the propagation of $\Psi$ with constant input and over a relatively short simulation time. See below.

The left image shows the propagation of $\Psi$ with the linearization terms, and the trig functions excluded from $h$; the right image shows the same simulation with trig functions maually updated in $h$.
