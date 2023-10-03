### **Koopman Operator Notes**

This repository serves as the testing ground for my research at Boston University. This *readme.md* is in the process of being rewritten as a brief introduction to Koopman operator theory. Some notes on my research into the Koopman-feedback operator, and using operators in series are in the folder titled *.archive*.

___
### **The Koopman Operator**

General Koopman operator theory contains an assortment of methods for studying dynamical systems, whether that be through data-driven or analytical approaches. A system represented by the state space $x \in \mathbb{X} \subset \mathbb{R}^n$ can ordinarily be defined by the differential equation

$$
    \dot x = f(x,t).
$$

Where $\dot x$ is the derivative of $x$ w.r.t to time, $t$. Here, $f: \mathbb{X} \rightarrow \mathbb{Y}$ where $\mathbb{Y} \subset \mathbb{R}^n$. In many cases (if not most), $f$ is a nonlinear function of the input terms, and can be difficult to evaluate.

Koopman theory dictates that for every dynamical system $f$

$$
    \exists g \text{ s.t. } g(\dot x) = \mathcal{K} g(x)
$$

where $g : \mathbb{X} \rightarrow \mathbb{G}(\mathbb{X})$ is a Hilbert space of observation functions over $x$ (and subsequently $\dot x$). In this space, the Koopman operator is linear such that $\mathcal{K} : \mathbb{G}(\mathbb{X}) \rightarrow \mathbb{G}(\mathbb{Y})$; thus describing the full dynamics of $f$.

In practice, the Koopman operator is truncated so that it can be used in real time and on modern computers. The truncated operator also allows for the solution to be derived from data. The truncated operator is usually denoted by

$$
    \begin{aligned}
        \Psi(\dot x) & = K \Psi(x) \\
        \text{where } \Psi(x) & = [\psi_1(x) \cdots \psi_n(x)]^\intercal \subset g(x)
    \end{aligned}
$$

Where $\psi_1 \cdots \psi_n$ is the list of observation functions selected by the user. This implies that the Koopman operator problem is now a two-step process:

$$
    \begin{aligned}
        1). & \text{ the user selects the observation functions of interest, and } \\
        2). & \text{ the Koopman operator, $K$, is derived. }
    \end{aligned}
$$

There has been extensive research into the solutions to $(1)$ and $(2)$, some of which will be discussed in later sections.

**TBC...**

<!--


___
### **The Koopman-feedback Operator**

The Koopman-feedback operator (KFO) is a preliminary application for our study of using Koopman operators in series to model systems which can be broken into sub-operators via **shift functions**. The definition of a shift function, and a more in-depth discussion of them is forthcoming, but an outline of the KFO will be given here to explain the notation used in the **anchor system** application.

The KFO was first proposed as a solution to the problem where the user is given measurement data which is ambiguously correlated with the system state and control policy. The communication between the Koopman system w.r.t the true system is rendered in the diagram below.

<p align="center">
    <img src=.figures/anchors/kman_closedloop.png width=600/>
</p>

In this system, two Koopman operators are used to independently generate the control policy from feedback-related measurements, as well as the state estimation using previous position data and the appropriate control. The noteworthy components are the model and control-related Koopman operators; $\mathcal{K}_x$ and $\mathcal{K}_u$, respectively, as well as the observation spaces; $\Psi_x$, $\Psi_u$ and $h$. In the following sections, the observation spaces will be indentified for the anchor system along with a simple set of trials.

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

The important component introduced here is the idea that the state is not observed directly, but instead observed through **anchors**. For practical purposes we assume there are $N_a=3$ anchors which, when called, give the object its respective $L_2$ distance norm. The position of the anchors can be chosen by the user and are defined by the set

$$
    d(x) = \\{ \sqrt{ (x - a_i)^\intercal (x - a_i) } : \forall i \leq N_a, i \in \mathbb{N} \\}.
$$

Where $d$ represents the set of anchor distances, each defined by the anchor position $a_i$, or the $i$-th anchor in the system.

To make the system more realistic and general, we also assume there is some unknown level of noise acting on the measurement terms. In other words, for any position, $x$, given to the system we receive noise in the range space defined by $p(\delta) \in [-\delta, \delta]$, and for any measurement reading from the anchors we receive noise in the range space $p(\varepsilon) \in [-\varepsilon, \varepsilon]$.


| **Open-loop Observation Space:** |

We demonstrate elseware that the necessary observation functions to propagate the state space while also keeping track of the anchor distances are

$$
\begin{aligned}
    \Psi_x = \begin{bmatrix}
        x \\
        x^\intercal x \\
        d^2(x) \\
        1
    \end{bmatrix}
    &&
    \Psi_u = 1
    &&
    h = \begin{bmatrix}
        u \\
        u^\intercal u \\
        x^\intercal u
    \end{bmatrix}
\end{aligned}
$$

A short sim which shows the vehicle's ability to keep track of the anchor distances is shown below. The radius of the black circles represents the distance the vehicle *estimates* it is from each of the reference anchors as time progresses. The distances are represented by circles because the vehicle is never given data corresponding to the direction of each of the anchors.

<p align="center">
    <img src=.figures/anchors/anim_openloop.gif width=450 />
</p>

It should also be noted that the path taken by the vehicle is predetermined. I.e. the inputs given to the observation space $h$ are 'hard coded'.


| **Closed-loop Observation Space:** |

Derived elsewhere, the observation space for the closed-loop control and state estimation can be defined as

$$
\begin{aligned}
    \Psi_x = x
    &&
    \Psi_u = 1
    &&
    h =d^2(x)
\end{aligned}
$$

We show that these results can give high fidelity w.r.t the ideal system results. Below is an isolated trial for a non-zero initial position, with a Koopman operator-based controller that guides the vehicle to the origin using solely measurement data.

<p align="center">
    <img src=.figures/anchors/singlePathEnvironment.png height=325 />
    <img src=.figures/anchors/singlePathTrajectories.png height=325 />
</p>

As can be seen, the state estimation is slightly erratic due to the large degree of noise incorporated into the feedback terms, but overall the path construcated by the operator is smooth. Finally, we show that for many trials, with varying degrees of feedback error, the vehicle is consistently able to generate smooth paths.

<p align="center">
    <img src=.figures/anchors/multiplePathEnvironment_e0.000.png height=250 />
    <img src=.figures/anchors/multiplePathEnvironment_e0.500.png height=250 />
</p>
<p align="center">
    <img src=.figures/anchors/multiplePathEnvironment_e1.000.png height=250 />
    <img src=.figures/anchors/multiplePathEnvironment_e2.000.png height=250 />
</p>


!-->
