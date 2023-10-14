### **Observation Selection**

Here, an example for the analytical approach will be given in an attempt to clarify the notation used in the root folder's *readme.md*. Let us define the Duffing oscillator such that

$$
    \begin{aligned}
        &\ x = \begin{bmatrix}
            x_1 \\
            x_2
        \end{bmatrix} \\
        f(x,t) = &\ \dot x = \begin{bmatrix}
            \dot x_1 \\
            \dot x_2
        \end{bmatrix} = \begin{bmatrix}
            x_2 \\
            \alpha x_1 + \beta x_1^3 + \delta x_2 - \gamma \cos( \omega t )
        \end{bmatrix}.
    \end{aligned}
$$

For simplicity, we will define the coefficients of $\dot x_2$ to $\alpha = \beta = \delta = \gamma = \omega = 1$. This choice will allow for easier identification of observable functions, and will not affect the solution to $K$ that we define later.

Using our knowledge of the model, we can construct an initial observation space by including the functions of $x$ and $t$ which are in the ODE of interest

$$
    \Psi(x,t) = [x_1, x_2, x_1^3, \cos(t),...]^\intercal.
$$

Now, the operator will attempt to linearly represent the derivative of the terms included. We can find those by taking the derivative of each element:

$$
    \begin{aligned}
        & \dot x_1 & = &\ x_2 && \text{(included)} \\
        & \dot x_2 & = &\ x_1 + x_1^3 + x_2 - \cos(t) && \text{(included)} \\
        & \dot x_1^3 & = &\ 3 x_1^2 \dot x_1 = 3 x_1^2 x_2 && \text{(not included)} \\
        & \frac{d}{dt}\cos(t) & = &\ -\sin(t) && \text{(not included)}
    \end{aligned}
$$

We can now incorporate the new terms into the observation space s.t.

$$
    \Psi(x,t) = [x_1, x_2, x_1^3, x_1^2 x_2, \cos(t), \sin(t), \cdots]^\intercal.
$$

Similar to before, we take the derivative of the new terms and follow each tree until all terms are included in the observation space

$$
    \begin{aligned}
        & \frac{d}{dt}\sin(t) & = &\ \cos(t) && \text{(included)} \\
        & \frac{d}{dt} x_1^2 x_3 & = &\ 2 x_1 x_2^2 + x_1^2(x_1 + x_1^3 + x_2 - \cos(t)) && \text{(not included)}
    \end{aligned}
$$

As is clear by the second line, the highest power and complexity of the system grows on each iteration of the derivative.
