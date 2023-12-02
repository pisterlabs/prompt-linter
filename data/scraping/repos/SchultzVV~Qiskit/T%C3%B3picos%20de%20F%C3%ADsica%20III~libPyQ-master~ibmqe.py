import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pTranspose as pT
import discord
import coherence as coh
import entanglement as ent
from distances import fidelity_mm
from states import Werner
from decoherence import werner_pdad
import tomography as tomo
from math import sqrt
import rpvg
import states as st
import gates
import math
import mat_func as mf
import pTrace as ptr
import gell_mann as gm
import rpvg


def werner():
    Nw = 11 # no. of experiments of each configuration
    Nr = 7 # no. of rounds of the experiment
    we = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    Ee = np.zeros(Nw)
    Eerr = np.zeros(Nw)  # for the standard deviation
    Cnle = np.zeros(Nw)
    Cnlerr = np.zeros(Nw)
    Nle = np.zeros(Nw)
    Nlerr = np.zeros(Nw)
    Se = np.zeros(Nw)
    Serr = np.zeros(Nw)
    De = np.zeros(Nw)
    Derr = np.zeros(Nw)
    Fe = np.zeros(Nw)
    Ferr = np.zeros(Nw)
    for j in range(0, Nw):
        sj = str(j)
        Em = 0.0;  E2m = 0.0;  Fm = 0.0;  F2m = 0.0;  Cnlm = 0.0;  Cnl2m = 0.0
        Nlm = 0.0;  Nl2m = 0.0;  Sm = 0.0;  S2m = 0.0;  Dm = 0.0;  D2m = 0.0
        for k in range(0,Nr):
            sk = str(k)
            path1 = '/home/jonas/Dropbox/Research/ibm/bds'
            #path1 = '/Users/jonasmaziero/Dropbox/Research/ibm/bds'
            path2 = '/calc_mauro/dados_plot/dados_plot'
            path = path1 + path2 + sk + '/' + sj + '/'
            if k == 0:
                rhoe = tomo.tomo_2qb(path)
            else:
                rhoe = tomo.tomo_2qb_(path)
            E = 2*ent.negativity(4, pT.pTransposeL(2, 2, rhoe))
            Em += E;  E2m += pow(E,2)
            F = fidelity_mm(4, Werner(j*0.1), rhoe);  Fm += F;  F2m += pow(F,2)
            Cnl = coh.coh_nl(2, 2, rhoe);  Cnlm += Cnl;  Cnl2m += pow(Cnl,2)
            Nl = ent.chsh(rhoe);  Nlm += Nl;  Nl2m += pow(Nl,2)
            S = ent.steering(rhoe);  Sm += S;  S2m += pow(S,2)
            #D = discord.oz_2qb(rhoe);  Dm += D;  D2m += pow(D,2)
            #D = discord.hellinger(2, 2, rhoe);  Dm += D;  D2m += pow(D,2)
        Em = Em/Nr;  E2m = E2m/Nr;  Eerr[j] = sqrt(E2m - pow(Em,2));  Ee[j] = Em
        Fm = Fm/Nr;  F2m = F2m/Nr;  Ferr[j] = sqrt(F2m - pow(Fm,2));  Fe[j] = Fm
        Cnlm = Cnlm/Nr;  Cnl2m = Cnl2m/Nr;  Cnlerr[j] = sqrt(Cnl2m - pow(Cnlm,2));  Cnle[j] = Cnlm
        Nlm = Nlm/Nr;  Nl2m = Nl2m/Nr;  Nlerr[j] = sqrt(Nl2m - pow(Nlm,2));  Nle[j] = Nlm
        Sm = Sm/Nr;  S2m = S2m/Nr;  Serr[j] = sqrt(S2m - pow(Sm,2));  Se[j] = Sm
        Dm = Dm/Nr;  D2m = D2m/Nr;  Derr[j] = sqrt(D2m - pow(Dm,2));  De[j] = Dm
        #F = fidelity_mm(4, Werner(j*0.1), rhoe)
        #Ee = 2*ent.negativity(4, pT.pTransposeL(2, 2, rhoe))
        #Cnle = coh.coh_nl(2, 2, rhoe)
        #Nle[j] = ent.chsh(rhoe)
        #Se[j] = ent.steering(rhoe)
        # De[j] = discord.hellinger(2, 2, rhoe)
        #De[j] = discord.oz_2qb(rhoe)
        #F[j] = fidelity_mm(4, Werner(j*0.1), rhoe)
    Nt = 110
    Et = np.zeros(Nt)
    Nlt = np.zeros(Nt)
    St = np.zeros(Nt)
    Cnlt = np.zeros(Nt)
    wt = np.zeros(Nt)
    Dt = np.zeros(Nt)
    Etd = np.zeros(Nt)
    Nltd = np.zeros(Nt)
    Std = np.zeros(Nt)
    Cnltd = np.zeros(Nt)
    Dtd = np.zeros(Nt)
    p = 0.15
    a = p
    dw = 1.01/Nt
    w = -dw
    for j in range(0, Nt):
        w = w + dw
        if w > 1.01:
            break
        #rho = Werner(w)
        #Et[j] = 2*ent.negativity(4, pT.pTransposeL(2, 2, rho))
        #Cnlt[j] = coh.coh_nl(2, 2, rho)
        #Nlt[j] = ent.chsh(rho)
        #St[j] = ent.steering(rho)
        #Dt[j] = discord.oz_2qb(rho)
        #Dt[j] = discord.hellinger(2, 2, rho)
        rhod = werner_pdad(w, p, a)
        Etd[j] = ent.concurrence(rhod)
        Etd[j] = 2*ent.negativity(4, pT.pTransposeL(2, 2, rhod))
        Cnltd[j] = coh.coh_nl(2, 2, rhod)
        Nltd[j] = ent.chsh(rhod)
        Std[j] = ent.steering(rhod)
        Dtd[j] = discord.oz_2qb(rhod)
        # Dtd[j] = discord.hellinger(2, 2, rhod)
        wt[j] = w
    #plt.errorbar(we, Fe, Ferr, marker='x', label=r'$F$', color='black', markersize=5)
    #plt.plot(we, F, 'x', label=r'$F$', color='black')
    #plt.plot(wt, Cnlt, '.', label='$C$', color='gray')
    #plt.errorbar(we, Cnle, Cnlerr, marker='*', label=r'$C_{e}$', color='gray', markersize=5)
    plt.plot(wt, Cnltd, 'H', label='$C_{d}$', color='gray', markersize=3)
    #plt.plot(we, Cnle, '*', label=r'$C_{e}$', color='gray', markersize=8)
    #plt.plot(wt, Dt, '-', label='D', color='magenta')
    #plt.errorbar(we, De, Derr, marker='o', label=r'$D_{e}$', color='magenta', markersize=5)
    plt.plot(wt, Dtd, 'x', label='$D_{d}$', color='magenta', markersize=3)
    #plt.plot(we, De, 'o', label=r'$D_{e}$', color='magenta', markersize=8)
    #plt.plot(wt, Et, '-.', label='E', color='blue')
    #plt.errorbar(we, Ee, Eerr, marker='s', label=r'$E_{e}$', color='blue', markersize=5)
    plt.plot(wt, Etd, '4', label='$E_{d}$', color='blue', markersize=3)
    #plt.plot(we, Ee, 's', label=r'$E_{e}$', color='blue', markersize=8)
    #plt.errorbar(we, Ee, errE, xerr=None)
    #plt.plot(wt, St, ':', label='$S$', color='red')
    #plt.errorbar(we, Se, Serr, marker='^', label=r'$S_{e}$', color='red', markersize=5)
    plt.plot(wt, Std, 'd', label='$S_{d}$', color='red', markersize=3)
    #plt.plot(we, Se, '^', label=r'$S_{e}$', color='red', markersize=8)
    #plt.plot(wt, Nlt, '--', label='$N$', color='cyan')
    #plt.errorbar(we, Nle, Nlerr, marker='h', label=r'$N_{e}$', color='cyan', markersize=5)
    plt.plot(wt, Nltd, '+', label='$N_{d}$', color='cyan', markersize=3)
    #plt.plot(we, Nle, 'h', label=r'$N_{e}$', color='cyan', markersize=8)
    plt.xlabel('w')
    plt.legend(loc=(0.02,0.5))
    plt.xlim(-0.01,1.02)
    plt.ylim(-0.03,1.02)
    import platform
    if platform.system() == 'Linux':
        plt.savefig('/home/jonas/Dropbox/Research/ibm/bds/calc/qcorrp015.eps',
                    format='eps', dpi=100)
    else:
        plt.savefig('/Users/jonas/Dropbox/Research/ibm/bds/calc/qcorrp015.eps',
                    format='eps', dpi=100)
    plt.show()


def bds_circuit(theta,alpha):
    ket0 = st.cb(2,0); ket1 = st.cb(2,1)
    proj0 = mf.proj(2,ket0); proj1 = mf.proj(2,ket1)
    psi = ket0
    for j in range(1,4):
        psi = np.kron(psi,ket0)
    gate = np.kron(gates.O2(theta/2),gates.O2(alpha/2))
    gate = np.kron(gate,gates.id(4))
    psi = np.dot(gate,psi)
    cn = gates.cnot(4,0,2)
    psi = np.dot(cn,psi)
    cn = gates.cnot(4,1,3)
    psi = np.dot(cn,psi)
    gate = np.kron(gates.id(8),gates.hadamard())
    psi = np.dot(gate,psi)
    cn = gates.cnot(4,3,2)
    psi = np.dot(cn,psi)
    return psi


def dbs_circuit_test():
    ns = 5*10**3
    cxx = np.zeros(ns)
    cyy = np.zeros(ns)
    czz = np.zeros(ns)
    for j in range(0,ns):
        rpv = rpvg.rpv_zhsl(4)
        theta = 2.0*math.acos(math.sqrt(rpv[0]+rpv[1]))
        alpha = 2.0*math.acos(math.sqrt(rpv[0]+rpv[2]))
        psi = bds_circuit(theta,alpha)
        rhor = ptr.pTraceL(4,4,mf.proj(16,psi))
        cm = gm.corr_mat(2,2,rhor)
        cxx[j] = cm[0][0]
        cyy[j] = cm[1][1]
        czz[j] = cm[2][2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cxx, cyy, czz, c = 'b', marker='o', s=1)
    ax.set_xlabel('c_xx')
    ax.set_ylabel('c_yy')
    ax.set_zlabel('c_zz')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.plot([-1,1],[-1,1],[-1,-1], color='b')
    ax.plot([-1,1],[1,-1],[1,1], color='b')
    ax.plot([-1,1],[-1,-1],[-1,1], color='b')
    ax.plot([-1,-1],[-1,1],[-1,1], color='b')
    ax.plot([1,-1],[1,1],[-1,1], color='b')
    ax.plot([1,1],[1,-1],[-1,1], color='b')
    plt.show()


def dbs_circuit_test_angles():
    ns = 10**4
    cxx = np.zeros(ns)
    cyy = np.zeros(ns)
    czz = np.zeros(ns)
    m = -1
    da = 2*math.pi/100
    theta = -da
    for j in range(0,100):
        theta += da
        alpha = -da
        for k in range(0,100):
            alpha += da
            psi = bds_circuit(theta,alpha)
            rhor = ptr.pTraceL(4,4,mf.proj(16,psi))
            cm = gm.corr_mat(2,2,rhor)
            m += 1
            cxx[m] = cm[0][0]
            cyy[m] = cm[1][1]
            czz[m] = cm[2][2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cxx, cyy, czz, c = 'b', marker='o', s=0.75)
    ax.set_xlabel('c_xx')
    ax.set_ylabel('c_yy')
    ax.set_zlabel('c_zz')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.plot([-1,1],[-1,1],[-1,-1], color='b')
    ax.plot([-1,1],[1,-1],[1,1], color='b')
    ax.plot([-1,1],[-1,-1],[-1,1], color='b')
    ax.plot([-1,-1],[-1,1],[-1,1], color='b')
    ax.plot([1,-1],[1,1],[-1,1], color='b')
    ax.plot([1,1],[1,-1],[-1,1], color='b')
    plt.show()


def dbs_rpv_test():
    from states import bell
    ns = 5*10**3
    cxx = np.zeros(ns)
    cyy = np.zeros(ns)
    czz = np.zeros(ns)
    for j in range(0,ns):
        rpv = rpvg.rpv_zhsl(4)
        rhor = rpv[0]*mf.proj(4,bell(0,0)) + rpv[1]*mf.proj(4,bell(0,1))
        rhor += (rpv[2]*mf.proj(4,bell(1,0)) + rpv[3]*mf.proj(4,bell(1,1)))
        cm = gm.corr_mat(2,2,rhor)
        cxx[j] = cm[0][0]
        cyy[j] = cm[1][1]
        czz[j] = cm[2][2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cxx, cyy, czz, c = 'b', marker='o', s=0.5)
    ax.set_xlabel('c_xx')
    ax.set_ylabel('c_yy')
    ax.set_zlabel('c_zz')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.plot([-1,1],[-1,1],[-1,-1], color='b')
    ax.plot([-1,1],[1,-1],[1,1], color='b')
    ax.plot([-1,1],[-1,-1],[-1,1], color='b')
    ax.plot([-1,-1],[-1,1],[-1,1], color='b')
    ax.plot([1,-1],[1,1],[-1,1], color='b')
    ax.plot([1,1],[1,-1],[-1,1], color='b')
    plt.show()


def bds_circuit_(alpha,beta,gamma):
    # ref: arXiv:1912.06105
    ket0 = st.cb(2,0); ket1 = st.cb(2,1)
    proj0 = mf.proj(2,ket0); proj1 = mf.proj(2,ket1)
    psi = ket0
    for j in range(1,4):
        psi = np.kron(psi,ket0)
    gate = np.kron(gates.O2(alpha/2),gates.id(8))
    psi = np.dot(gate,psi)
    cn = gates.cnot(4,0,1)
    psi = np.dot(cn,psi)
    gate = np.kron(gates.O2(beta/2),gates.O2(gamma/2))
    gate = np.kron(gate,gates.id(4))
    psi = np.dot(gate,psi)
    cn = gates.cnot(4,0,2)
    psi = np.dot(cn,psi)
    cn = gates.cnot(4,1,3)
    psi = np.dot(cn,psi)
    gate = np.kron(gates.id(8),gates.hadamard())
    psi = np.dot(gate,psi)
    cn = gates.cnot(4,3,2)
    psi = np.dot(cn,psi)
    return psi

def bds_circuit_angles(p):
    a = np.sqrt(p)
    A = np.zeros((2,2))
    x = np.zeros((2,1))
    b = np.zeros((2,1))
    c = np.zeros((2,1))
    alpha = math.asin(2*(a[0][0]*a[1][1]-a[0][1]*a[1][0]))
    if math.cos(alpha) == 0:
        gamma = 0
        beta = 0
    else:
        A[0][0] = (math.cos(alpha/2)*a[0][0] - math.sin(alpha/2)*a[1][1])/math.cos(alpha)
        A[0][1] = (math.cos(alpha/2)*a[0][1] + math.sin(alpha/2)*a[1][0])/math.cos(alpha)
        A[1][0] = (math.sin(alpha/2)*a[0][1] + math.cos(alpha/2)*a[1][0])/math.cos(alpha)
        A[1][1] = (-math.sin(alpha/2)*a[0][0] + math.cos(alpha/2)*a[1][1])/math.cos(alpha)
        x = np.random.rand(2,1)
        b = np.matmul(np.matmul(A,A.T),x); 
        b /= np.linalg.norm(b,keepdims=True)
        if b[0][0] == 0:
            if b[1][0] < 0:
                b = -b
            beta = 2*math.asin(b[1][0])
        else:
            if b[0][0] < 0:
                b = -b
            beta = 2*math.atan(b[1][0]/b[0][0])
        c = np.matmul(np.matmul(A.T,A),x); 
        c /= np.linalg.norm(c,keepdims=True)
        M = np.matmul(b,c.T) - A
        if np.linalg.norm(M) > 0:
            c = -c
        if c[0][0] == 0:
            gamma = 2*math.asin(c[1][0])
        else:
            gamma = 2*math.atan(c[1][0]/c[0][0])
    return alpha, beta, gamma


def dbs_circuit_test_():
    ns = 5*10**3
    cxx = np.zeros(ns)
    cyy = np.zeros(ns)
    czz = np.zeros(ns)
    p = np.zeros((2,2))
    a = np.zeros(4)
    for j in range(0,ns):
        rpv = rpvg.rpv_zhsl(4)
        a = np.reshape(rpv,(2,2))
        alpha,beta,gamma = bds_circuit_angles(a)
        psi = bds_circuit_(alpha,beta,gamma)
        rhor = ptr.pTraceL(4,4,mf.proj(16,psi))
        cm = gm.corr_mat(2,2,rhor)
        cxx[j] = cm[0][0]
        cyy[j] = cm[1][1]
        czz[j] = cm[2][2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cxx, cyy, czz, c = 'b', marker='o', s=0.75)
    ax.set_xlabel('c_xx')
    ax.set_ylabel('c_yy')
    ax.set_zlabel('c_zz')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.plot([-1,1],[-1,1],[-1,-1], color='b')
    ax.plot([-1,1],[1,-1],[1,1], color='b')
    ax.plot([-1,1],[-1,-1],[-1,1], color='b')
    ax.plot([-1,-1],[-1,1],[-1,1], color='b')
    ax.plot([1,-1],[1,1],[-1,1], color='b')
    ax.plot([1,1],[1,-1],[-1,1], color='b')
    plt.show()


def dbs_circuit_test_angles_():
    ns = 20**3
    cxx = np.zeros(ns)
    cyy = np.zeros(ns)
    czz = np.zeros(ns)
    m = -1
    da = math.pi/20
    alpha = -math.pi/2 - da
    for j in range(0,20):
        alpha += da
        beta = -da
        for k in range(0,20):
            beta += da
            gamma = -da
            for l in range(0,20):
                gamma += da
                psi = bds_circuit_(alpha,beta,gamma)
                rhor = ptr.pTraceL(4,4,mf.proj(16,psi))
                cm = gm.corr_mat(2,2,rhor)
                m += 1
                cxx[m] = cm[0][0]
                cyy[m] = cm[1][1]
                czz[m] = cm[2][2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cxx, cyy, czz, c = 'b', marker='o', s=0.75)
    ax.set_xlabel('c_xx')
    ax.set_ylabel('c_yy')
    ax.set_zlabel('c_zz')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.plot([-1,1],[-1,1],[-1,-1], color='b')
    ax.plot([-1,1],[1,-1],[1,1], color='b')
    ax.plot([-1,1],[-1,-1],[-1,1], color='b')
    ax.plot([-1,-1],[-1,1],[-1,1], color='b')
    ax.plot([1,-1],[1,1],[-1,1], color='b')
    ax.plot([1,1],[1,-1],[-1,1], color='b')
    plt.show()


#dbs_rpv_test()
#dbs_circuit_test()
#dbs_circuit_test_angles()
dbs_circuit_test_()
#dbs_circuit_test_angles_()
