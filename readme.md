### **Koopman Operator Notes**

This repository serves as the testing ground for my research at Boston University (being performed with the intent of being published). This readme currently discusses noteworthy models being evaluated and the results of their tests with the Koopman control equation (KCE). It also serves as a testing ground for my ideas on gradient flow mapping, etc.


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
    <img src=./Figures/point.png width=550 />
</p>

___
### **Anchor System:**
**State/model equations** are the same as those shown for the *Linear System*.

The interesting component introduced here is the idea that the state is not observed directly, but instead observed through *anchors*. For practical purposes there are four evenly distanced *anchors* which, when called, give the object its respective linear distance. That is,

$$
    d_i(x) = ||x - a_i||_2
$$

Where $d_i$ is the $L_2$-norm distance from the anchor, $a_i$, located at index $i$. The goal here is to utilize the EDMD learning structure to find a correlation between the *anchor* distances and the policy.

**Observation Function Structure:**

Here the observables are listed as

$$
\begin{aligned}
    \Psi_x = x
    &&
    \Psi_u = \begin{bmatrix}
        u \\
        1
    \end{bmatrix}
    &&
    h = \begin{bmatrix}
        1 \\
        d_1(x) \\
        d_2(x) \\
        \vdots \\
        d_a(x)
    \end{bmatrix}
\end{aligned}
$$

We purposely limit the $h$ observation list from having knowledge of the states so that we can demonstrate the robustness of the KCE under considerable limitations.

___
### **Caterpillar Track System:**
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

**Observation Functions Structure:**

It is important to note that for this system, the $cos(x_3)$ and $sin(x_3)$ terms must be included in $\Psi_u$ because of their bilinear combination in the model equations. For this reason, the measured observation list, $h$ must contain a method for propagating them forward as well as computing the control.

$$
\begin{aligned}
    \Psi_x = \left[ \begin{matrix}
        x_1 \\
        x_2 \\
        x_3
    \end{matrix} \right]
    &&
    \Psi_u = \left[ \begin{matrix} 
        \cos(x_3) \\
        \sin(x_3) \\
        1
    \end{matrix} \right]
    &&
    h = \left[ \begin{matrix} 
        u_1 \\
        u_2 \\
        \sum_{j=0}^{N_p} x_3^j
    \end{matrix} \right]
\end{aligned}
$$

We will consider using the Taylor series expansion of $\cos(x_3)$ and $\sin(x_3)$ with the hope that we can include enough terms to propagate the trigonomateric functions successfully.
