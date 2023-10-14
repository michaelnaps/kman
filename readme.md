### **Koopman Operator Notes**

This repository serves as the testing ground for my research at Boston University. This *readme.md* is in the process of being rewritten as a brief introduction to Koopman operator theory. Some notes on my research into the Koopman-feedback operator and using operators in series are in the folder titled *.archive*.

___
### **The Koopman Operator**

General Koopman operator theory contains an assortment of methods for studying dynamical systems, whether that be through data-driven or analytical approaches. A reference system represented by the state space $x \in \mathbb{X} \subset \mathbb{R}^n$ can ordinarily be defined by the differential equation

$$
    \dot x = f(x,t).
$$

Where $\dot x$ is the derivative of $x$ w.r.t to time, $t$. Here, $f: \mathbb{X} \rightarrow \mathbb{Y}$ where $\mathbb{Y} \subset \mathbb{R}^n$. In many cases (if not most), $f$ is a nonlinear function of the input terms, and can be difficult to evaluate.

Koopman theory dictates that for every dynamical system $f:$

$$
    \exists g \text{ s.t. } g(\dot x) = \mathcal{K} g(x)
$$

where $g : \mathbb{X} \rightarrow \mathbb{G}(\mathbb{X})$ is a Hilbert space of observation functions over $x$ (and subsequently $\dot x$). In this space, the Koopman operator is linear such that $\mathcal{K} : \mathbb{G}(\mathbb{X}) \rightarrow \mathbb{G}(\mathbb{Y})$; thus describing the full dynamics of $f$.

In practice, the Koopman operator is approximated to be finite so that it can be used in real time and on modern computers. The truncated operator also allows for the solution to be derived from data and is usually denoted by

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

There has been extensive research into the solution to $(2)$, some of which will be discussed in later sections. Example for the method of selecting observation terms, $(1)$, will be completed on a case-by-case basis in the subfolders to this repository.

    Primary references:
        [1] M. Budišić, R. Mohr, and I. Mezić, “Applied Koopmanism,” Chaos: An Interdisciplinary Journal
            of Nonlinear Science, vol. 22, no. 4, p. 047510, Dec. 2012, doi: 10.1063/1.4772195.
        [2] S. L. Brunton, B. W. Brunton, J. L. Proctor, and J. N. Kutz, “Koopman Invariant Subspaces and
            Finite Linear Representations of Nonlinear Dynamical Systems for Control,” PLOS ONE, vol. 11, no.
            2, p. e0150171, Feb. 2016, doi: 10.1371/journal.pone.0150171.

___

### **Extended Dynamic Mode Decomposition**

Here, we will discuss the most popular method for approximate $K$ using the data-driven least-squares approach, *extended dynamic mode decomposition* (EDMD).

In order to solve for the Koopman operator, the dimensionality of the function space must first be addressed. Being an infinitely-dimensioned space is not realistic when put in terms of real world applications so [REF] is redefined as a subset of the original space.

$$
    z = \Psi(x) \in g(x)
$$

Where $\Psi \in \mathbb{G}^{N} \subset \mathbb{G}$ is some observation function with dimension $N$ which approximates its infinitely dimensioned counterpart and is dependent on the accuracy desired by the user. More specifically, $\Psi$ can be written as the list of chosen observation functions.

$$
    \Psi(x) =\{ \psi_i(x) : \forall i \leq N, i \in \mathbb{N} \}.
$$

Where $\mathbb{N}$ is the set of natural numbers and each $\psi_i$ term is a scalar-valued function of the state variable. Combining this approach with [REF] yields the following.

$$
    \Psi(x^+) = K_N \Psi(x) + r(x,x^+)
$$

Where $r(x,x^+)$ is the residual error moving from $x$ to $x^+$ through the truncated function space, $\Psi$. Likewise, the Koopman operator is now represented as $K_N \in \R^{N \times N} \subset \R^{\infty \times \infty}$. Next, a series of data (found through data collection or simulation) is defined for use in finding $K_N$.

$$
    X = \{ x_i : x_i \in \mathbb{M},\ \forall i < P,\ i \in \mathbb{N} \}
$$

Where $X$ is a tuple of ($P$-$1$)-evenly spaced snapshots of the state. The forward snapshot of $X$ will also be defined to make notation clear.

$$
    X^+ = \{ x_{i+1} : x_{i+1} = F(x_i),\ \forall x_i \in X \}
$$

Using this data, the solution for the Koopman operator can be defined as the minimization of the residual error over the entire data set.

$$
    J = \frac{1}{2} || r(X,X^+) ||^2
$$

Which can be restated in terms of the Koopman operator approximation.

$$
    J = \frac{1}{2} ||\Psi(X^+) - K_N \Psi(X) ||^2
$$

In this form, the solution for $J$ is a simple least-squares regression such that $K_N = G^\dagger A$ where $G^\dagger$ is the pseudo-inverse of $G$. More specifically, the matrices $G$ and $A$ are composed of the observation functions for the current state and its propagation forward in time.

$$
    G = \frac{1}{P} \Psi(X) \Psi^\intercal(X),
$$

$$
    A = \frac{1}{P} \Psi(X) \Psi^\intercal(X^+).
$$

It is important to note that not all observation functions may be necessary in the final representation of the model. For this reason, single value decomposition (SVD) will be used to prioritize the higher impact terms when computing $G^\dagger$. The single value decomposition equation is restated here for completeness.

$$
    G = U S V^\intercal
$$

Where $U, V \in \R^{N \times s}$ and $S \in \R^{s \times s}$ such that the matrices represent the $s$-most prominent terms in the observation space. The pseudo-inverse can be found by exploiting the nature of the SVD results.

$$
    G^\dagger = V S^{-1} U^\intercal
$$

Note that for use in pseudocode, the calculation for $G^\dagger$ will be referred to by a call to the \Call{SVD}{G} function. This form can then be used to calculate a minimum-error Koopman operator.

$$
    K_N = \left( V S^{-1} U^\intercal \right) A
$$

Where $K_N$ is a learned approximation of the Koopman operator from data.
