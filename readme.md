### **Koopman Operator Notes**

This repository serves as the testing ground for my research at Boston University (being performed with the intent of being published). This readme currently discusses noteworthy models being evaluated and the results of their tests with the Koopman control equation (KCE). It also serves as a testing ground for my ideas on gradient flow mapping, etc.

___
### **The Koopman-feedback Operator**
COMING SOON...

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
    x^+ = x + (\Delta t) u \\
    \textrm{where } u = C(x_g - x) \leftarrow \text{ideal policy}
\end{matrix}
$$

The important component introduced here is the idea that the state is not observed directly, but instead observed through *anchors*. For practical purposes we assume there are $N_a=3$ randomly distrubuted anchors which, when called, give the object its respective $L_2$ distance norm. The collection of anchor distances are thus defined by the set

$$
    d(x) = \{ \sqrt{ (x - a_i)^\intercal (x - a_i) } : \forall i \leq N_a, i \in \mathbb{N} \}.
$$

Where $d$ represents the set of anchor distances, each defined by the anchor position $a_i$, or the $i$-th anchor in the system.

To make the system more realistic and general, we also assume there is some unknown level of noise acting on the measurement terms. In other words, for any measurement of the position $x$, we receive noise in the range space defined by $p(\delta) \in [-\delta, \delta]$, and for any measurement reading we receive noise in the range space $p(\varepsilon) \in [-\varepsilon, \varepsilon]$.

| **Observation Function Structure:** |

Derived elsewhere, the observation space can be defined as

$$
\begin{aligned}
    \Psi_x = x
    &&
    \Psi_u = 1
    &&
    h =d^2(x)
\end{aligned}
$$

We can show that these results can give high fidelity w.r.t the ideal system results. Shown below are some preliminary results without descriptions (upcoming).

<p align="center">
    <img src=.figures/anchors/anim_openloop.gif height=325 />
    <img src=.figures/anchors/singlePathEnvironment.png height=325 />
</p>

___
### **Differential Drive System:**

IN PROGRESS...

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

Meaning that for a small enough time-step, $\Delta t$, the function $\cos(x_3)$ can be propagated using a bilinear combination with the model equation $f_3$. To propagate the function $\cos(x_3) u_1$ we can explore a similar series of steps.

<!-- <p align="center">
    <img src=./.figures/donald.png width=325 />
    <img src=./.figures/donaldError.png width=325 />
</p> -->

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
