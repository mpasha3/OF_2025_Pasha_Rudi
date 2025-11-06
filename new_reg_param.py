import scipy
import numpy as np
import scipy.linalg as la
def tolu(l,A,Q_A,R_A,B,C,Q_C,R_C,D,opt=2):
    AC = np.concatenate(((1-l)*A,l*C))
    BD = np.concatenate(((1-l)*B,l*D))
    Q_AC,R_AC = la.qr(AC,mode='economic')
    BD = np.concatenate(((1-l)*B,l*D))
    def x_star(l):
      return np.linalg.lstsq(R_AC,Q_AC.T@BD,rcond= None)[0]
    def O1(l):
      rhs = la.norm(Q_A)**2*la.norm(R_A@x_star(l) - Q_A.T@B)**2 /(la.norm(Q_A)**2*la.norm(R_A@x_star(l) - Q_A.T@B)**2 + la.norm(Q_C)**2*la.norm(R_C@x_star(l) - Q_C.T@D)**2)
      return l - rhs   
    def O2(l):
      rhs = l* (la.norm(Q_A)**2*la.norm(R_A@x_star(l) - Q_A.T@B)**2 + la.norm(Q_C)**2*la.norm(R_C@x_star(l) - Q_C.T@D)**2) - la.norm(Q_A)**2*la.norm(R_A@x_star(l) - Q_A.T@B)**2 
      return rhs
    if (opt==1): 
      O = O1
    else:
      O = O2
    O_ = lambda l: -O(l)
    return O_(l) 
def gg(A,Q_A,R_A,B,C,Q_C,R_C,D):
    return (scipy.optimize.fminbound(tolu,0,1,[A,Q_A,R_A,B,C,Q_C,R_C,D]))
