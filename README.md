# New-Q-Newton-s-method-Backtracking
 
This is Python source code for the paper "New Q-Newton's method meets Backtracking line search ...", arXiv:... by Tuyen Trung Truong, in which New Q-Newton's method Backtracking is proposed. You can run functionsDirectRun7.py, proteinV2.py and StochasticGriewank2.py to test the method.  

This is a new second order method, which incorporates Backtracking line search into New Q-Newton's method. The latter is developed in previous work by  T. T. Truong, T. D. To,  H.-T. Nguyen, T. H. Nguyen, H. P. Nguyen and M. Helmy, "A fast and simple modification of quasi-Newton's methods helping to avoid saddle points", arXiv:2006.01512. 

New Q-Newton's method is proven to be able to avoid saddle points, and if converges will converge with quadratic rate of convergence. However, convergence guarantee is not known. 

New Q-Newton's method Backtracking has the same good properties as New Q-Newton's method, and in addition also has good convergence guarantee. 

In particular, it obtains, as far as I know, the best theoretical guarantee, for Morse functions,  for iterative optimization methods in the current literature. Indeed, we have the following: 

Theorem. Let f be a Morse cost function (i.e. all of its critical points have invertible Hessian). Let x_n be the sequence constructed by New Q-Newton's method Backtracking from a random initial point x_0. Then 

either 

i) \lim _{n\rightarrow\infty}||x_n||=\infty

or 

ii) x_n converges to a point x_{\infty} which is a local minimum, and the rate of convergence is quadratic. 




 



 
