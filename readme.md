### **Koopman Operator Notes**

The **kman.git** repository serves as the testing ground for my research at Boston University. This *readme.md* is in the process of being rewritten as a brief introduction to Koopman operator theory. Some notes on my research into the Koopman-feedback operator and using operators in series are in the folder titled *kman/Anchor*.

Working on bring KMAN library into C++. See **KMAN/cpp** folder.

___
### **The Koopman Operator**

General Koopman operator theory contains an assortment of methods for studying dynamical systems, whether that be through data-driven or analytical approaches. A reference system represented by the state space $x \in \mathbb{X} \subset \mathbb{R}^n$ can ordinarily be defined by the differential equation

$$
    \dot x = f(x,t).
$$

Where $\dot x$ is the derivative of $x$ w.r.t to time, $t$. Here, $f: \mathbb{X} \rightarrow \mathbb{Y}$ where $\mathbb{Y} \subset \mathbb{R}^n$. In many cases (if not most), $f$ is a nonlinear function of the input terms, and can be difficult to evaluate. Without loss of generality the term $t$ will be excluded in future notation.

Koopman theory dictates that for every dynamical system, $f$, we have that

$$
    \exists g \text{ s.t. } g(\dot x) = \mathcal{K} g(x)
$$

where $g : \mathbb{X} \rightarrow \mathbb{G}(\mathbb{X})$ is a Hilbert space of observation functions over $x$ (and subsequently $\dot x$). In this space, the Koopman operator is linear such that $\mathcal{K} : \mathbb{G}(\mathbb{X}) \rightarrow \mathbb{G}(\mathbb{Y})$; thus describing the full dynamics of $f$.

In practice, the Koopman operator is approximated to be finite so that it can be used in real time and on modern computers. The truncated operator also allows for the solution to be derived from data and is usually denoted by

$$
    \begin{aligned}
        \Psi(\dot x) & = K \Psi(x) \\
        \text{where } \Psi(x) & = \\{ \psi_i(x) : \forall i \leq n \\} \subset g(x)
    \end{aligned}
$$

Where $\psi_1 \cdots \psi_n$ is the list of observation functions selected by the user. This implies that the Koopman operator problem is now a two-step process:

$$
    \begin{aligned}
        1). & \text{ the user selects the observation functions of interest, and } \\
        2). & \text{ the Koopman operator, $K$, is derived. }
    \end{aligned}
$$

There has been extensive research into the solution to $(2)$, some of which will be discussed in later sections. Examples for the method of selecting observation terms, $(1)$, will be completed on a case-by-case basis in the subfolders to this repository.

    Primary references:
        M. Budišić, R. Mohr, and I. Mezić, “Applied Koopmanism,” Chaos: An Interdisciplinary Journal
            of Nonlinear Science, vol. 22, no. 4, p. 047510, Dec. 2012, doi: 10.1063/1.4772195.
        S. L. Brunton, B. W. Brunton, J. L. Proctor, and J. N. Kutz, “Koopman Invariant Subspaces and
            Finite Linear Representations of Nonlinear Dynamical Systems for Control,” PLOS ONE, vol.
            11, no. 2, p. e0150171, Feb. 2016, doi: 10.1371/journal.pone.0150171.

___

### **Extended Dynamic Mode Decomposition**

Here, we will discuss the most popular method for approximate $K$ using the data-driven, least-squares approach referred to as *extended dynamic mode decomposition* (EDMD). We will define the process in terms

With the understanding that we have identified the observation space of interest, we can write the truncated Koopman operator as

$$
    \Psi(\dot x) = K \Psi(x) + r(x,\dot x).
$$

Where $r(x,\dot x)$ is the residual error moving from $x$ to $\dot x$ through $\Psi$. Next, a series of data (found through data collection, etc.) is defined by

$$
    X = \\{ x_i : x_i \in \mathbb{M},\ \forall i \leq P \\}.
$$

Where $X$ is a tuple of P-evenly spaced snapshots of the state. The derivative at each snapshot can thus be calculated using a finite-difference approach, etc. to get

$$
    \dot X = \\{ \dot x_i = f(x_i) : \forall x_i \in X \\}.
$$

Using this data, the solution for the Koopman operator can be defined as the minimization of the residual error over the entire data set such that

$$
    J = \frac{1}{2} || r(X,\dot X) ||^2.
$$

Which can be restated in terms of the Koopman operator approximation:

$$
    J = \frac{1}{2} ||\Psi(\dot X) - K \Psi(X) ||^2.
$$

In this form, the solution for $J$ is a simple least-squares regression such that $K = G^\dagger A$ where $G^\dagger$ is the pseudo-inverse of $G$. More specifically, the matrices $G$ and $A$ are composed of the observation functions for the current state and its derivative such that

$$
    \begin{aligned}
        G & = \frac{1}{P} \Psi(X) \Psi^\intercal(X), \text{ and} \\
        A & = \frac{1}{P} \Psi(X) \Psi^\intercal(\dot X).
    \end{aligned}
$$

It is important to note that not all observation functions may be necessary in the final representation of the model. For this reason, single value decomposition (SVD) will be used to prioritize the higher impact terms when computing $G^\dagger$. The single value decomposition equation is restated here for completeness:

$$
    G = U S V^\intercal.
$$

Where $U, V \in \mathbb{R}^{N \times s}$ and $S \in \mathbb{R}^{s \times s}$ such that $G$ represents the $s$-most prominent terms in the observation space. The pseudo-inverse can be found by exploiting the nature of the SVD results:

$$
    G^\dagger = V S^{-1} U^\intercal.
$$

This form can then be used to calculate a minimum-error Koopman operator such that

$$
    K = \left( V S^{-1} U^\intercal \right) A,
$$

where $K$ is a learned approximation of the Koopman operator from data.

    Primary references:
        M. O. Williams, I. G. Kevrekidis, and C. W. Rowley, “A Data–Driven Approximation of the
            Koopman Operator: Extending Dynamic Mode Decomposition,” J Nonlinear Sci, vol. 25, no.
            6, pp. 1307–1346, Dec. 2015, doi: 10.1007/s00332-015-9258-5.
