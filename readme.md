### **Koopman Operator Notes**

This repository serves as the testing ground for my research being performed at Boston University with the intent of being published. This readme currently discusses noteworthy models being evaluated theoretically before being tested with the Koopman control equation (KCE) formulation. It also serves as a testing ground for my ideas on gradient flow mapping, etc.


___
#### Linear Anchor System:

State/Model Selection: linear/second-order/discrete

$$
    x = \left[ \begin{matrix}
        x \\ y \\ \dot x \\ \dot y
    \end{matrix} \right]
$$

With the model equation written in terms of linear matrices A and B.

$$
    x^+ = Ax + Bu
$$

Where

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
\end{aligned}
$$

The interesting component introduced here is the idea that the state is not observed directly, but instead observed through *anchors*. For practical purposes there are four evenly distanced *anchors* which, when called, give the object its respective linear distance. That is,

$$
    d_i(x) = ||x - a_i||_2
$$

Where $d$ is the $L_2$-norm distance function and $a$ is the center point of an active $anchor$.


___
#### Caterpillar Track System (Donald):
State/Model Selection: nonlinear/first-order/discrete

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

Observation Functions Structure:

$$
\begin{aligned}
    \Psi_x = \left[ \begin{matrix} x_1 \\ x_2 \\ x_3 \end{matrix} \right]
    &&
    \Psi_u = \left[ \begin{matrix} u_1 \\ u_2 \\ 1 \end{matrix} \right]
    &&
    h = \left[ \begin{matrix} \cos(x_3) \\ \sin(x_3) \end{matrix} \right]
\end{aligned}
$$
