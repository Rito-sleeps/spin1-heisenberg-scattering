#!/usr/bin/env python3
"""
Vanilla Python: sandwich MPS (sMPS) for magnon scattering in the
spin-1 antiferromagnetic Heisenberg chain.

Reference: Milsted et al., Phys. Rev. B 88, 155116 (2013) [arXiv:1207.0691]
Based on evoMPS by Ashley Milsted: https://github.com/amilsted/evoMPS
"""
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as las
import copy, time, sys

# ================================================================
# TENSOR PRIMITIVES (vectorized with einsum for speed)
# ================================================================
def eps_l_noop(x, A1, A2):
    """Left transfer: sum_s A1[s]^H x A2[s] = einsum('sij,jk,skl->il', A1*, x, A2)"""
    return np.einsum('sji,jk,skl->il', A1.conj(), x, A2)

def eps_r_noop(x, A1, A2):
    """Right transfer: sum_s A1[s] x A2[s]^H = einsum('sij,jk,slk->il', A1, x, A2*)"""
    return np.einsum('sij,jk,slk->il', A1, x, A2.conj())

def eps_r_op_1s(x, A1, A2, op):
    """Right transfer with 1-site operator: sum_{s,t} op[s,t] A1[t] x A2[s]^H"""
    # opA1 = sum_t op[s,t] A1[t] -> einsum('st,tij->sij', op, A1)
    opA1 = np.einsum('st,tij->sij', op, A1)
    return np.einsum('sij,jk,slk->il', opA1, x, A2.conj())

def calc_AA(A, Ap1):
    """Two-site tensor: AA[s,t,i,j] = sum_k A[s,i,k] Ap1[t,k,j]"""
    return np.einsum('sik,tkj->stij', A, Ap1)

def calc_C_mat_op_AA(op, AA):
    d = AA.shape[0]*AA.shape[1]
    return (op.reshape(d,d) @ AA.reshape(d, -1)).reshape(AA.shape)

def eps_r_op_2s_C12_AA34(x, C, AA):
    d = C.shape[0]*C.shape[1]
    return eps_r_noop(x, C.reshape(d,C.shape[2],C.shape[3]),
                         AA.reshape(d,AA.shape[2],AA.shape[3]))

def eps_l_op_2s_AA12_C34(x, AA, C):
    d = AA.shape[0]*AA.shape[1]
    return eps_l_noop(x, AA.reshape(d,AA.shape[2],AA.shape[3]),
                         C.reshape(d,C.shape[2],C.shape[3]))

def adot(a, b):
    return np.inner(a.ravel().conj(), b.ravel())

def adot_noconj(a, b):
    return np.inner(a.T.ravel(), b.ravel())

def nullspace_qr(A):
    Q, R = la.qr(np.atleast_2d(A).T)
    return Q[:, R.shape[1]:].conj()

def herm_sqrt_inv(x, zero_tol=1e-15):
    x = np.asarray(x)
    if not np.all(np.isfinite(x)):
        D = x.shape[0]
        return np.eye(D, dtype=x.dtype), np.eye(D, dtype=x.dtype)
    # Symmetrize to ensure Hermitian
    x = 0.5 * (x + x.conj().T)
    ev, EV = la.eigh(x)
    ev_s = np.sqrt(np.maximum(ev, 0.0))
    ev_si = np.where(ev > zero_tol, 1.0/np.where(ev_s > 0, ev_s, 1.0), 0.0)
    ev_s[ev <= zero_tol] = 0
    return EV @ np.diag(ev_s) @ EV.conj().T, EV @ np.diag(ev_si) @ EV.conj().T

# ================================================================
# GAUGE TRANSFORMS (for sandwich CF)
# ================================================================
def herm_fac_with_inv(M, zero_tol=1e-15, lower=True):
    """Factorize Hermitian M. lower=True: M=XX^H. lower=False: M=X^H X.
    Returns X, Xi, rank. Uses EVD (like evoMPS force_evd=True)."""
    ev, EV = la.eigh(M)
    nz = np.count_nonzero(ev > zero_tol)
    ev_sq = np.zeros_like(ev, dtype=M.dtype)
    ev_sq_i = np.zeros_like(ev, dtype=M.dtype)
    if nz > 0:
        ev_sq[-nz:] = np.sqrt(ev[-nz:])
        ev_sq_i[-nz:] = 1.0 / ev_sq[-nz:]
    if lower:
        X  = EV @ np.diag(ev_sq)          # M = X X^H
        Xi = np.diag(ev_sq_i) @ EV.conj().T
    else:
        X  = np.diag(ev_sq) @ EV.conj().T  # M = X^H X
        Xi = EV @ np.diag(ev_sq_i)
    return X, Xi, nz

def restore_RCF_r(A, r, G_n_i, zero_tol=1e-15):
    """Transform A[n] to get r[n-1] ~ identity. Returns r_nm1, G_nm1, G_nm1_i."""
    GGh = G_n_i @ r @ G_n_i.conj().T if G_n_i is not None else r.copy()
    M = eps_r_noop(GGh, A, A)
    X, Xi, nD = herm_fac_with_inv(M, zero_tol=zero_tol)
    G_nm1 = Xi.conj().T; G_nm1_i = X.conj().T
    if G_n_i is None: G_n_i = G_nm1_i
    for s in range(A.shape[0]): A[s] = G_nm1 @ A[s] @ G_n_i
    if nD == A.shape[1]: r_nm1 = np.eye(A.shape[1], dtype=A.dtype)
    else:
        d = np.zeros(A.shape[1], dtype=float); d[-nD:] = 1.0
        r_nm1 = np.diag(d).astype(A.dtype)
    return r_nm1, G_nm1, G_nm1_i

def restore_RCF_l(A, lm1, Gm1):
    """Transform A[n] to get diagonal l[n]. Returns l, G, Gi."""
    x = Gm1.conj().T @ lm1 @ Gm1 if Gm1 is not None else lm1.copy()
    M = eps_l_noop(x, A, A)
    ev, EV = la.eigh(M)
    l = np.diag(ev.astype(A.dtype))
    G_i = EV
    if Gm1 is None: Gm1 = EV.conj().T
    for s in range(A.shape[0]): A[s] = Gm1 @ A[s] @ G_i
    return l, EV.conj().T, G_i

def restore_LCF_l(A, lm1, Gm1, zero_tol=1e-15):
    """Transform A[n] to get l[n] ~ identity. Returns l, G, Gi."""
    GhG = Gm1.conj().T @ lm1 @ Gm1 if Gm1 is not None else lm1.copy()
    M = eps_l_noop(GhG, A, A)
    G, Gi, nD = herm_fac_with_inv(M, zero_tol=zero_tol)
    if Gm1 is None: Gm1 = G
    for s in range(A.shape[0]): A[s] = Gm1 @ A[s] @ Gi
    if nD == A.shape[2]: l = np.eye(A.shape[2], dtype=A.dtype)
    else:
        d = np.zeros(A.shape[2], dtype=float); d[-nD:] = 1.0
        l = np.diag(d).astype(A.dtype)
    return l, G, Gi

def restore_LCF_r(A, r, Gi):
    """Diagonal r step. Returns rm1, Gm1, Gm1_i."""
    x = Gi @ r @ Gi.conj().T if Gi is not None else r.copy()
    M = eps_r_noop(x, A, A)
    ev, EV = la.eigh(M)
    rm1 = np.diag(ev.astype(A.dtype))
    Gm1 = EV.conj().T
    if Gi is None: Gi = EV
    for s in range(A.shape[0]): A[s] = Gm1 @ A[s] @ Gi
    return rm1, Gm1, EV

# ================================================================
# UNIFORM MPS
# ================================================================
class UniformMPS:
    def __init__(self, D, q, ham):
        self.D, self.q, self.L = D, q, 1
        self.ham = ham; self.ham_sites = 2
        self.typ = np.complex128
        self.zero_tol = np.finfo(self.typ).resolution
        self.symm_gauge = True
        self.A  = [np.zeros((q,D,D), dtype=self.typ)]
        self.AA = [None]
        self.l  = [np.eye(D, dtype=self.typ)]
        self.r  = [np.eye(D, dtype=self.typ)]
        self.K  = [np.zeros((D,D), dtype=self.typ)]
        self.K_left = [np.zeros((D,D), dtype=self.typ)]
        self.C  = [None]
        self.h_expect = 0.0; self.eta = np.inf
        self.eta_sq = np.zeros(1)
        self.Vsh = [None]
        self.lL_before_CF = self.l[0].copy()
        self.rL_before_CF = self.r[0].copy()
        for Ak in self.A:
            Ak[:] = np.random.randn(*Ak.shape)+1j*np.random.randn(*Ak.shape)
            Ak /= la.norm(Ak)
        self.update(restore_CF=True)

    # ---------- Transfer-matrix eigenvectors ----------
    def _build_E(self):
        D, A = self.D, self.A[0]; q = A.shape[0]
        E = np.zeros((D*D,D*D), dtype=self.typ)
        for s in range(q): E += np.kron(A[s], A[s].conj())
        return E

    def calc_lr(self):
        D = self.D; E = self._build_E()
        ev, eVL, eVR = la.eig(E, left=True, right=True)
        idx = np.argmax(np.abs(ev)); ev_dom = ev[idx]
        if np.abs(ev_dom) > 1e-15:
            self.A[0] *= 1.0/np.sqrt(np.abs(ev_dom))
        # Re-compute with rescaled A
        E = self._build_E()
        ev, eVL, eVR = la.eig(E, left=True, right=True)
        idx = np.argmax(np.abs(ev))
        l_vec, r_vec = eVL[:,idx], eVR[:,idx]
        for v in [l_vec, r_vec]:
            vm = v.mean()
            if np.abs(vm)>1e-15: v *= np.sqrt(np.conj(vm)/vm)
            if v.mean().real < 0: v *= -1
        l = l_vec.reshape(D,D); r = r_vec.reshape(D,D)
        l = 0.5*(l+l.conj().T); r = 0.5*(r+r.conj().T)
        norm = adot(l,r)
        if norm.real < 0: l = -l; norm = -norm
        sq = np.sqrt(abs(norm))
        if sq>1e-30: l/=sq; r/=sq
        self.l[0]=l; self.r[0]=r
        self.lL_before_CF=l.copy(); self.rL_before_CF=r.copy()

    def restore_CF(self):
        self.calc_lr()
        if self.symm_gauge: self._restore_SCF()

    def _restore_SCF(self):
        """Symmetric canonical form: l=r=diag(Schmidt coefficients)."""
        l, r = np.asarray(self.l[0]), np.asarray(self.r[0])
        zt = self.zero_tol
        # r = X X^H (lower)
        X, _, _ = herm_fac_with_inv(r, zero_tol=zt, lower=True)
        # l = Y^H Y (upper, so lower=False)
        Y, _, _ = herm_fac_with_inv(l, zero_tol=zt, lower=False)
        U, sv, Vh = la.svd(Y @ X)
        Srt = np.sqrt(sv)
        Srti = np.zeros_like(sv)
        nz = sv > zt
        Srti[nz] = 1.0/Srt[nz]
        g   = np.diag(Srti) @ U.conj().T @ Y    # left gauge
        g_i = X @ Vh.conj().T @ np.diag(Srti)    # right gauge
        for s in range(self.q):
            self.A[0][s] = g @ self.A[0][s] @ g_i
        self.l[0] = np.diag(sv); self.r[0] = np.diag(sv)

    def restore_RCF(self, zero_tol=None):
        if zero_tol is None: zero_tol = self.zero_tol
        self.calc_lr()
        r = np.asarray(self.r[0]); l = np.asarray(self.l[0])
        X, Xi, _ = herm_fac_with_inv(r, zero_tol=zero_tol)
        G = Xi.conj().T; Gi = X.conj().T
        for s in range(self.q): self.A[0][s] = G @ self.A[0][s] @ Gi
        self.r[0] = np.eye(self.D, dtype=self.typ)
        self.l[0] = G @ l @ G.conj().T
        return [G], [Gi]

    def restore_LCF(self, zero_tol=None):
        if zero_tol is None: zero_tol = self.zero_tol
        self.calc_lr()
        l = np.asarray(self.l[0]); r = np.asarray(self.r[0])
        X, Xi, _ = herm_fac_with_inv(l, zero_tol=zero_tol, lower=False)
        # l = X^H X. Want Gm1^H l Gm1 = I => Gm1 = X^{-1}
        Gm1 = Xi  # l = X^H X, Xi = inv(X), so Xi^H l Xi = I
        Gm1_i = X
        for s in range(self.q):
            self.A[0][s] = Gm1_i.conj().T @ self.A[0][s] @ Gm1.conj().T
        self.l[0] = np.eye(self.D, dtype=self.typ)
        self.r[0] = Gm1_i.conj().T @ r @ Gm1_i
        return [Gm1_i.conj().T], [Gm1.conj().T]

    # ---------- update ----------
    def update(self, restore_CF=True):
        if restore_CF: self.restore_CF()
        else: self._update_lr_fast()
        self.calc_AA(); self.calc_C(); self.calc_K()

    def _update_lr_fast(self):
        """Recompute l,r via dense eigensolver (no gauge change)."""
        D=self.D; E=self._build_E()
        ev, eVL, eVR = la.eig(E, left=True, right=True)
        idx=np.argmax(np.abs(ev)); ev_dom=ev[idx]
        if np.abs(ev_dom)>1e-15 and abs(np.abs(ev_dom)-1)>1e-14:
            self.A[0] *= 1.0/np.sqrt(np.abs(ev_dom))
            E = self._build_E()
            ev, eVL, eVR = la.eig(E, left=True, right=True)
            idx=np.argmax(np.abs(ev))
        l_vec, r_vec = eVL[:,idx], eVR[:,idx]
        for v in [l_vec, r_vec]:
            vm=v.mean()
            if np.abs(vm)>1e-15: v*=np.sqrt(np.conj(vm)/vm)
            if v.mean().real<0: v*=-1
        l=l_vec.reshape(D,D); r=r_vec.reshape(D,D)
        l=0.5*(l+l.conj().T); r=0.5*(r+r.conj().T)
        norm=adot(l,r)
        if norm.real<0: l=-l; norm=-norm
        sq=np.sqrt(abs(norm))
        if sq>1e-30: l/=sq; r/=sq
        self.l[0]=l; self.r[0]=r

    # ---------- C, K, B ----------
    def calc_AA(self): self.AA[0] = calc_AA(self.A[0], self.A[0])
    def calc_C(self):  self.C[0] = calc_C_mat_op_AA(self.ham, self.AA[0])

    def calc_K(self):
        Hr = eps_r_op_2s_C12_AA34(self.r[0], self.C[0], self.AA[0])
        self.h_expect = adot(self.l[0], Hr)
        QHr = Hr - self.r[0]*self.h_expect
        self.K[0] = self._ppinv(QHr.ravel(), left=False).reshape(self.D,self.D)

    def calc_K_l(self):
        lH = eps_l_op_2s_AA12_C34(self.l[0], self.AA[0], self.C[0])
        h = adot_noconj(lH, self.r[0])
        lHQ = lH - self.l[0]*h
        self.K_left[0] = self._ppinv(lHQ.ravel(), left=True).reshape(self.D,self.D)
        return self.K_left, h

    def _ppinv(self, x_vec, left=False, p=0):
        D=self.D; l=np.asarray(self.l[0]); r=np.asarray(self.r[0]); A=self.A
        def mv(v):
            xm=v.reshape(D,D)
            if left:
                res=eps_l_noop(xm,A[0],A[0]); res-=l*adot(r,xm)
                res*=-np.exp(-1j*p); res+=xm
            else:
                res=eps_r_noop(xm,A[0],A[0]); res-=r*adot(l,xm)
                res*=-np.exp(1j*p); res+=xm
            return res.ravel()
        op=las.LinearOperator((D*D,D*D), matvec=mv, dtype=self.typ)
        y,info=las.bicgstab(op, x_vec, x0=np.ones(D*D,dtype=self.typ),
                            rtol=1e-12, maxiter=4000)
        if info!=0:
            y,info=las.gmres(op, x_vec, rtol=1e-10, maxiter=4000)
        return y

    def calc_Vsh(self, A, r_s):
        D,Dm1,q = A.shape[2],A.shape[1],A.shape[0]
        if q*D-Dm1<=0: return None
        R = np.zeros((D,q,Dm1), dtype=A.dtype)
        for s in range(q): R[:,s,:] = r_s @ A[s].conj().T
        R = R.reshape(q*D, Dm1)
        Vc = nullspace_qr(R.conj().T).T          # (qD-Dm1, qD)
        Vc = Vc.reshape(q*D-Dm1, D, q)
        return np.ascontiguousarray(Vc.T)         # (q, D, qD-Dm1)

    def calc_Vsh_l(self, A, l_s):
        D,Dm1,q = A.shape[2],A.shape[1],A.shape[0]
        if q*Dm1-D<=0: return None
        L = np.zeros((D,q,Dm1), dtype=A.dtype)
        for s in range(q): L[:,s,:] = (l_s@A[s]).conj().T
        L = L.reshape(D, q*Dm1)
        V = nullspace_qr(L)                       # (q*Dm1, q*Dm1-D)
        V = V.reshape(q, Dm1, q*Dm1-D)
        return np.ascontiguousarray(V.conj().transpose(0,2,1))  # (q, q*Dm1-D, Dm1)

    def calc_B(self, set_eta=True):
        l_s, l_si = herm_sqrt_inv(np.asarray(self.l[0]), self.zero_tol)
        r_s, r_si = herm_sqrt_inv(np.asarray(self.r[0]), self.zero_tol)
        A=self.A[0]; q=self.q; D=self.D
        Vsh = self.calc_Vsh(A, r_s)   # (q, D, qD-D)
        self.Vsh[0] = Vsh
        if Vsh is None: self.eta=0; return [np.zeros_like(A)]
        C=self.C[0]; K=self.K[0]
        # x = l_s . [sum_s (C_term + K_term) . r_si . Vsh[s]]
        #   + l_si . [sum_s eps_l(l, A, CmT[s]) . r_s . Vsh[s]]
        x = np.zeros((D, q*D-D), dtype=A.dtype)
        p1 = np.zeros_like(x)
        for s in range(q):
            sub = eps_r_noop(self.r[0], C[s], A) + A[s]@K   # (D,D)
            p1 += sub @ (r_si @ Vsh[s])                      # (D,D)@(D,qD-D)
        x += l_s @ p1
        CmT = C.transpose(1,0,2,3)  # swap phys indices for C[n-1] term
        p2 = np.zeros_like(x)
        for s in range(q):
            sub = eps_l_noop(self.l[0], A, CmT[s])   # (D,D)
            p2 += sub @ (r_s @ Vsh[s])
        x += l_si @ p2
        if set_eta:
            self.eta_sq[0]=adot(x,x).real; self.eta=np.sqrt(self.eta_sq.sum().real)
        B = np.zeros_like(A)
        for s in range(q): B[s] = l_si @ x @ (r_si @ Vsh[s]).conj().T
        return [B]

    def take_step(self, dtau, B=None):
        if B is None: B=self.calc_B()
        self.A[0] += -dtau * B[0]

    def take_step_RK4(self, dtau, B_i=None):
        A0=self.A[0].copy()
        B=self.calc_B() if B_i is None else B_i[:]
        Bf = [B[0].copy()]
        self.A[0]=A0-dtau/2*B[0]; self.update(False)
        B2=self.calc_B(False); self.A[0]=A0-dtau/2*B2[0]; Bf[0]+=2*B2[0]
        self.update(False)
        B3=self.calc_B(False); self.A[0]=A0-dtau*B3[0]; Bf[0]+=2*B3[0]
        self.update(False)
        B4=self.calc_B(False); Bf[0]+=B4[0]
        self.A[0]=A0-dtau/6*Bf[0]

    def expect_1s(self, op, k=0):
        return adot(self.l[0], eps_r_op_1s(self.r[0], self.A[0], self.A[0], op))

    def save_state(self, f):
        np.save(f, {'A':self.A[0],'l':self.l[0],'r':self.r[0]}, allow_pickle=True)
    def load_state(self, f):
        d=np.load(f, allow_pickle=True).item()
        self.A[0]=d['A']; self.l[0]=d['l']; self.r[0]=d['r']
        self.lL_before_CF=self.l[0].copy(); self.rL_before_CF=self.r[0].copy()

# ================================================================
# GROUND STATE FINDER
# ================================================================
def find_ground(uni, tol=1e-6, dtau=0.04, max_steps=10000, verbose=True):
    if verbose: print(f"{'Step':>6s} {'h':>16s} {'eta':>12s}")
    best_h=np.inf; best_A=uni.A[0].copy()
    best_l=uni.l[0].copy(); best_r=uni.r[0].copy()
    cur_dtau = dtau
    for i in range(max_steps):
        try:
            uni.update(restore_CF=(i%8==0))
        except Exception:
            uni.A[0]=best_A.copy(); uni.l[0]=best_l.copy()
            uni.r[0]=best_r.copy(); cur_dtau*=0.5; continue
        B = uni.calc_B()
        h = uni.h_expect.real; eta = uni.eta.real
        if np.isnan(h) or np.isinf(h) or np.isnan(eta):
            uni.A[0]=best_A.copy(); uni.l[0]=best_l.copy()
            uni.r[0]=best_r.copy(); cur_dtau*=0.5; continue
        if h < best_h:
            best_h=h; best_A=uni.A[0].copy()
            best_l=uni.l[0].copy(); best_r=uni.r[0].copy()
        if verbose and i%50==0:
            print(f"{i:6d} {h:16.10f} {eta:12.4e}")
        if eta < tol:
            if verbose: print(f"{i:6d} {h:16.10f} {eta:12.4e}  converged")
            break
        Bn = la.norm(B[0])
        step = min(cur_dtau, 0.5/Bn) if Bn>0 else cur_dtau
        uni.take_step(step, B=B)
    uni.A[0]=best_A; uni.l[0]=best_l; uni.r[0]=best_r
    uni.update(restore_CF=True)
    if verbose: print(f"Final: h = {uni.h_expect.real:.10f}")
    return uni

# ================================================================
# SANDWICH MPS
# ================================================================
class SandwichTDVP:
    def __init__(self, N, uni_ground):
        self.N=N; self.N_centre=N//2; self.typ=np.complex128
        self.zero_tol = np.finfo(self.typ).resolution
        D=uni_ground.D; q=uni_ground.q
        self.uni_l=copy.deepcopy(uni_ground)
        self.uni_r=copy.deepcopy(uni_ground)
        self.D=np.full(N+2, D, dtype=int)
        self.q=np.full(N+2, q, dtype=int)
        self.A=[None]*(N+3); self.l=[None]*(N+3); self.r=[None]*(N+3)
        for n in range(N+3):
            if n<=N+1:
                self.l[n]=np.zeros((D,D),dtype=self.typ)
                self.r[n]=np.zeros((D,D),dtype=self.typ)
            if 1<=n<=N: self.A[n]=uni_ground.A[0].copy()
        for n in range(N+2):
            self.l[n][:]=np.asarray(uni_ground.l[0])
            self.r[n][:]=np.asarray(uni_ground.r[0])
        self.ham=[uni_ground.ham]*(N+1); self.ham_sites=2
        self._AA=[None]*(N+2); self._C=[None]*(N+2)
        self.K=[None]*(N+3); self.K_l=[None]*(N+3)
        for n in range(N+2):
            self._C[n]=np.zeros((q,q,D,D),dtype=self.typ)
            self._AA[n]=np.zeros((q,q,D,D),dtype=self.typ)
        for n in range(self.N_centre, N+2): self.K[n]=np.zeros((D,D),dtype=self.typ)
        for n in range(self.N_centre+1):    self.K_l[n]=np.zeros((D,D),dtype=self.typ)
        self.h_expect=np.zeros(N+1,dtype=self.typ); self.dH_expect=np.nan
        self.eta_sq=np.zeros(N+1,dtype=self.typ); self.eta=np.nan
        self.grown_left=0; self.grown_right=0
        self.uni_l.update(); self.uni_l.calc_B()
        self.eta_sq_uni=self.uni_l.eta_sq.copy()

    def get_A(self,n):
        if n<1: return self.uni_l.A[0]
        elif n>self.N: return self.uni_r.A[0]
        return self.A[n]
    def get_l(self,n):
        if 0<=n<=self.N+1: return self.l[n]
        elif n<0: return np.asarray(self.uni_l.l[0])
        else: return np.asarray(self.uni_r.l[0])
    def get_r(self,n):
        if 0<=n<=self.N+1: return self.r[n]
        elif n>self.N+1: return np.asarray(self.uni_r.r[0])
        else: return np.asarray(self.uni_l.r[0])
    def get_AA(self,n):
        if 0<=n<=self.N: return self._AA[n]
        return self.uni_l.AA[0]
    def get_C(self,n):
        if 0<=n<=self.N: return self._C[n]
        return None

    # ----- l, r, normalisation -----
    def calc_l(self, n_lo=1, n_hi=None):
        if n_hi is None: n_hi=self.N
        self.l[0]=np.asarray(self.uni_l.l[0]).copy()
        for n in range(max(n_lo,1), n_hi+1):
            self.l[n]=eps_l_noop(self.l[n-1], self.A[n], self.A[n])
            # Stabilize: prevent exponential growth by tracking scale
            nrm = np.trace(self.l[n]).real
            if np.isfinite(nrm) and nrm > 0 and abs(nrm) > 1e+5:
                self.l[n] /= nrm / self.l[n].shape[0]
        if n_hi>=self.N:
            self.l[self.N+1]=eps_l_noop(self.l[self.N], self.uni_r.A[0], self.uni_r.A[0])

    def calc_r(self, n_lo=0, n_hi=None):
        if n_hi is None: n_hi=self.N-1
        self.r[self.N]=np.asarray(self.uni_r.r[0]).copy()
        for n in range(min(n_hi,self.N-1), n_lo-1, -1):
            self.r[n]=eps_r_noop(self.r[n+1], self.A[n+1], self.A[n+1])
            nrm = np.trace(self.r[n]).real
            if np.isfinite(nrm) and nrm > 0 and abs(nrm) > 1e+5:
                self.r[n] /= nrm / self.r[n].shape[0]
        self.r[self.N+1]=self.r[self.N].copy()

    def simple_renorm(self, update_lr=True):
        nc=self.N_centre
        norm=adot(self.l[nc-1], self.r[nc-1])
        norm_real = norm.real if np.isfinite(norm) else 1.0
        if norm_real <= 0:
            norm_real = abs(norm) if abs(norm) > 1e-30 else 1.0
        if abs(1 - norm_real) > 1e-15 and np.isfinite(norm_real) and norm_real > 0:
            self.A[nc] = self.A[nc] * (1.0 / np.sqrt(norm_real))
        if update_lr: self.calc_l(n_lo=nc); self.calc_r(n_hi=nc-1)

    def restore_CF(self):
        nc=self.N_centre; D=self.D[0]; zt=self.zero_tol
        self.uni_l.calc_lr(); self.l[0]=np.asarray(self.uni_l.l[0]).copy()
        self.uni_r.calc_lr(); self.r[self.N]=np.asarray(self.uni_r.r[0]).copy()
        # ONR right part: r[n>=nc] -> identity
        uGs,uGis = self.uni_r.restore_RCF(zero_tol=zt)
        Gi = uGs[0]  # This is G for the last site
        self.r[self.N]=np.asarray(self.uni_r.r[0]).copy()
        for n in range(self.N, nc, -1):
            self.r[n-1], Gm1, Gm1_i = restore_RCF_r(self.A[n], self.r[n], Gi, zt)
            Gi = Gm1_i
        self.r[self.N+1]=self.r[self.N].copy()
        for s in range(self.q[nc]): self.A[nc][s] = self.A[nc][s] @ Gi
        # LCF left part: l[n<nc] -> identity
        uGs_l, uGis_l = self.uni_l.restore_LCF(zero_tol=zt)
        Gm1 = uGis_l[0]
        self.l[0]=np.asarray(self.uni_l.l[0]).copy()
        for n in range(1, nc):
            self.l[n], G, G_i = restore_LCF_l(self.A[n], self.l[n-1], Gm1, zt)
            Gm1 = G
        for s in range(self.q[nc]): self.A[nc][s] = Gm1 @ self.A[nc][s]
        # Diagonal steps: r[n<nc] diagonal, l[n>=nc] diagonal
        Ui = np.eye(D, dtype=self.typ)
        for n in range(nc, 0, -1):
            self.r[n-1], Um1, Um1_i = restore_LCF_r(self.A[n], self.r[n], Ui)
            Ui = Um1_i
        U = Um1
        for s in range(self.uni_l.q):
            self.uni_l.A[0][s] = U @ self.uni_l.A[0][s] @ Ui
        self.uni_l.r[0] = U @ np.asarray(self.uni_l.r[0]) @ U.conj().T
        Um1 = np.eye(D, dtype=self.typ)
        for n in range(nc, self.N+1):
            self.l[n], U, Ui = restore_RCF_l(self.A[n], self.l[n-1], Um1)
            Um1 = U
        Um1_i = Ui
        for s in range(self.uni_r.q):
            self.uni_r.A[0][s] = Um1 @ self.uni_r.A[0][s] @ Um1_i
        self.uni_r.l[0] = Um1_i.conj().T @ np.asarray(self.uni_r.l[0]) @ Um1_i
        self.uni_l.lL_before_CF=np.asarray(self.uni_l.l[0]).copy()
        self.uni_l.rL_before_CF=np.asarray(self.uni_l.r[0]).copy()
        self.uni_r.lL_before_CF=np.asarray(self.uni_r.l[0]).copy()
        self.uni_r.rL_before_CF=np.asarray(self.uni_r.r[0]).copy()
        self.r[nc-1]=eps_r_noop(self.r[nc], self.A[nc], self.A[nc])
        self.simple_renorm(update_lr=False)
        self.l[self.N+1]=eps_l_noop(self.l[self.N], self.uni_r.A[0], self.uni_r.A[0])

    # ----- C, K -----
    def calc_C(self):
        for n in range(0, self.N+1):
            self._AA[n]=calc_AA(self.get_A(n), self.get_A(n+1))
            self._C[n]=calc_C_mat_op_AA(self.ham[n], self._AA[n])

    def calc_K(self):
        self.h_expect[:]=0; nc=self.N_centre
        self.uni_r.calc_AA(); self.uni_r.calc_C(); self.uni_r.calc_K()
        self.K[self.N+1]=np.asarray(self.uni_r.K[0]).copy()
        self.uni_l.calc_AA(); self.uni_l.calc_C()
        K_left_l, h_l_uni = self.uni_l.calc_K_l()
        self.K_l[0]=np.asarray(K_left_l[0]).copy()
        for n in range(self.N, nc, -1):
            K_np1=self.K[n+1]; An=self.get_A(n)
            K=eps_r_noop(K_np1,An,An)
            Hr=eps_r_op_2s_C12_AA34(self.get_r(n+1), self._C[n], self._AA[n])
            he=adot(self.get_l(n-1), Hr)
            self.K[n]=K+Hr; self.h_expect[n]=he
        for n in range(1, nc+1):
            K_lm1=self.K_l[n-1]; An=self.get_A(n)
            K=eps_l_noop(K_lm1, An, An)
            Hl=eps_l_op_2s_AA12_C34(self.get_l(n-2), self._AA[n-1], self._C[n-1])
            he=adot_noconj(Hl, self.r[n])
            self.K_l[n]=K+Hl; self.h_expect[n-1]=he
        self.dH_expect=(adot_noconj(self.K_l[nc], self.get_r(nc))
                        +adot(self.get_l(nc-1), self.K[nc])
                        -(self.N+1)*h_l_uni)

    def update(self, restore_CF=True, normalize=True):
        if restore_CF: self.restore_CF()
        else:
            if normalize:
                self.calc_l(n_hi=self.N_centre-1)
                self.calc_r(n_lo=self.N_centre-1)
                self.simple_renorm(update_lr=True)
            else:
                self.calc_l(); self.calc_r()
        self.calc_C(); self.calc_K()

    # ----- TDVP B -----
    def calc_B_n(self, n, set_eta=True):
        nc=self.N_centre
        if n==nc: return self._calc_B_centre(set_eta)
        lnm1 = np.asarray(self.l[n-1])
        rn = np.asarray(self.r[n])
        if not (np.all(np.isfinite(lnm1)) and np.all(np.isfinite(rn))):
            if set_eta: self.eta_sq[n]=0
            return np.zeros_like(self.A[n])
        l_s, l_si = herm_sqrt_inv(lnm1, self.zero_tol)
        r_s, r_si = herm_sqrt_inv(rn, self.zero_tol)
        A=self.A[n]; q=self.q[n]; D=A.shape[2]; Dm1=A.shape[1]
        if n>nc:
            Vsh=self.uni_l.calc_Vsh(A, r_s)
            if Vsh is None: return None
            x = self._calc_x_r(n, Vsh, l_s, l_si, r_s, r_si)
            B=np.empty_like(A)
            for s in range(q): B[s]=l_si @ x @ (r_si @ Vsh[s]).conj().T
        else:
            Vsh=self.uni_l.calc_Vsh_l(A, l_s)
            if Vsh is None: return None
            x = self._calc_x_l(n, Vsh, l_s, l_si, r_s, r_si)
            B=np.empty_like(A)
            for s in range(q): B[s]=l_si @ Vsh[s].conj().T @ x @ r_si
        if set_eta: self.eta_sq[n]=adot(x,x)
        return B

    def _calc_x_r(self, n, Vsh, l_s, l_si, r_s, r_si):
        A=self.A[n]; q=self.q[n]; D=A.shape[2]; Dm1=A.shape[1]
        x=np.zeros((Dm1, q*D-Dm1), dtype=A.dtype)
        C=self.get_C(n); K=self.K[n+1]; Ap1=self.get_A(n+1); rp1=self.get_r(n+1)
        # Precompute r_si @ Vsh[s] and r_s @ Vsh[s] for all s
        riV = np.einsum('ij,sjk->sik', r_si, Vsh)   # (q, D, qD-Dm1)
        rsV = np.einsum('ij,sjk->sik', r_s, Vsh)    # (q, D, qD-Dm1)
        if C is not None or K is not None:
            # Term 1: eps_r(rp1, C[s], Ap1) for each s -> vectorized
            # C[s] shape (q, Dm1, D), treat as (q, Dm1, D) MPS tensor
            # eps_r(rp1, C[s], Ap1) = sum_t C[s,t] rp1 Ap1[t]^H
            if C is not None:
                # Batch: sum_t C[s,t,i,k] rp1[k,l] Ap1[t,j,l]* = einsum for each s
                CrAH = np.einsum('stik,kl,tjl->sij', C, rp1, Ap1.conj())
            else:
                CrAH = np.zeros((q, Dm1, D), dtype=A.dtype)
            if K is not None:
                AK = A @ K  # (q, Dm1, D)
                CrAH = CrAH + AK
            # p = sum_s CrAH[s] @ riV[s] 
            p = np.einsum('sij,sjk->ik', CrAH, riV)
            x += l_s @ p
        Cm1=self.get_C(n-1)
        if Cm1 is not None:
            CmT=Cm1.transpose(1,0,2,3); Am1=self.get_A(n-1); lm2=self.get_l(n-2)
            # sub[s] = eps_l(lm2, Am1, CmT[s]) = sum_t Am1[t]^H lm2 CmT[s,t]
            # = einsum('tji,jk,tskl->sil', Am1*, lm2, CmT) -- but CmT has shape (q,q,D,D)
            # Actually CmT[s] has shape (q, D, D), so eps_l(lm2, Am1, CmT[s]) for each s
            subs = np.einsum('tji,jk,atkl->ail', Am1.conj(), lm2, CmT)  # (q, D, D)
            p2 = np.einsum('sij,sjk->ik', subs, rsV)
            x += l_si @ p2
        return x

    def _calc_x_l(self, n, VshL, l_s, l_si, r_s, r_si):
        A=self.A[n]; q=self.q[n]; D=A.shape[2]; Dm1=A.shape[1]
        x=np.zeros((q*Dm1-D, D), dtype=A.dtype)
        C=self.get_C(n)
        if C is not None:
            Ap1=self.get_A(n+1); rp1=self.get_r(n+1)
            # sub[s] = sum_t C[s,t] rp1 Ap1[t]^H
            subs = np.einsum('stik,kl,tjl->sij', C, rp1, Ap1.conj())  # (q, Dm1, D)
            # p[s] = VshL[s] @ l_s @ subs[s]
            ls_sub = np.einsum('ij,sjk->sik', l_s, subs)  # (q, Dm1, D)
            p = np.einsum('sij,sjk->ik', VshL, ls_sub)  # (qDm1-D, D)
            x += p @ r_si
        Cm1=self.get_C(n-1); K_lm1=self.K_l[n-1]
        lm2=self.get_l(n-2); Am1=self.get_A(n-1)
        sub2 = np.zeros((q, Dm1, D), dtype=A.dtype)
        if Cm1 is not None and lm2 is not None:
            CmT=Cm1.transpose(1,0,2,3)
            sub2 += np.einsum('tji,jk,atkl->ail', Am1.conj(), lm2, CmT)
        if K_lm1 is not None:
            sub2 += np.einsum('ij,sjk->sik', K_lm1, A)
        lsi_sub = np.einsum('ij,sjk->sik', l_si, sub2)
        p2 = np.einsum('sij,sjk->ik', VshL, lsi_sub)
        x += p2 @ r_s
        return x

    def _calc_B_centre(self, set_eta=True):
        nc=self.N_centre; Ac=self.A[nc]; q=self.q[nc]
        rc=np.asarray(self.r[nc]); lcm1=np.asarray(self.l[nc-1])
        # Safe pseudo-inverse
        try:
            rc_i=la.inv(rc)
        except la.LinAlgError:
            rc_i=la.pinv(rc)
        try:
            lcm1_i=la.inv(lcm1)
        except la.LinAlgError:
            lcm1_i=la.pinv(lcm1)
        if not (np.all(np.isfinite(rc_i)) and np.all(np.isfinite(lcm1_i))):
            if set_eta: self.eta_sq[nc]=0
            return np.zeros_like(Ac)
        Acm1=self.get_A(nc-1); Acp1=self.get_A(nc+1)
        rcp1=self.get_r(nc+1); lcm2=self.get_l(nc-2)
        K_l_cm1=self.K_l[nc-1]-lcm1*adot_noconj(self.K_l[nc-1],self.r[nc-1])
        Kcp1=self.K[nc+1]-rc*adot(self.l[nc],self.K[nc+1])
        Cc=self.get_C(nc)-self.h_expect[nc]*self.get_AA(nc)
        Ccm1=self.get_C(nc-1)-self.h_expect[nc-1]*self.get_AA(nc-1)
        # Term 3: Ac[s] @ Kcp1 @ rc_i
        t3 = np.einsum('sij,jk,kl->sil', Ac, Kcp1, rc_i)
        # Term 1: sum_t Cc[s,t] rcp1 Acp1[t]^H rc_i
        t1 = np.einsum('stik,kl,tjl,jm->sim', Cc, rcp1, Acp1.conj(), rc_i)
        # Term 4: K_l_cm1 @ Ac[s]
        t4 = np.einsum('ij,sjk->sik', K_l_cm1, Ac)
        # Term 2: sum_t Acm1[t]^H lcm2 Ccm1[t,s]
        t2 = np.einsum('tji,jk,tskl->sil', Acm1.conj(), lcm2, Ccm1)
        # Combine: Bc = t3 + t1 + lcm1_i @ (t4 + t2)
        Bc = t3 + t1 + np.einsum('ij,sjk->sik', lcm1_i, t4 + t2)
        if set_eta:
            rb=eps_r_noop(rc,Bc,Bc); self.eta_sq[nc]=adot(lcm1,rb)
        return Bc

    def calc_B(self):
        self.eta_sq[:]=0
        B=[None]*(self.N+2)
        for n in range(1,self.N+1):
            B[n]=self.calc_B_n(n)
            if B[n] is None: B[n]=np.zeros_like(self.A[n])
        self.eta=np.sqrt(np.sum(self.eta_sq).real)
        return B

    def take_step(self, dtau, B=None):
        if B is None: B=self.calc_B()
        for n in range(1,self.N+1):
            if B[n] is not None: self.A[n]+=-dtau*B[n]

    def take_step_RK4(self, dtau, B_i=None):
        A0=[None]*(self.N+2)
        for n in range(1,self.N+1): A0[n]=self.A[n].copy()
        B=self.calc_B() if B_i is None else [b for b in B_i]
        Bf=[None]*(self.N+2)
        for n in range(1,self.N+1): Bf[n]=B[n].copy()
        for n in range(1,self.N+1): self.A[n]=A0[n]-dtau/2*B[n]
        self.update(restore_CF=False, normalize=True)
        for n in range(1,self.N+1):
            b2=self.calc_B_n(n,False)
            if b2 is None: b2=np.zeros_like(self.A[n])
            self.A[n]=A0[n]-dtau/2*b2; Bf[n]+=2*b2
        self.update(restore_CF=False, normalize=True)
        for n in range(1,self.N+1):
            b3=self.calc_B_n(n,False)
            if b3 is None: b3=np.zeros_like(self.A[n])
            self.A[n]=A0[n]-dtau*b3; Bf[n]+=2*b3
        self.update(restore_CF=False, normalize=True)
        for n in range(1,self.N+1):
            b4=self.calc_B_n(n,False)
            if b4 is None: b4=np.zeros_like(self.A[n])
            Bf[n]+=b4
        for n in range(1,self.N+1): self.A[n]=A0[n]-dtau/6*Bf[n]

    def expect_1s(self, op, n):
        A=self.get_A(n); r=self.get_r(n); l=self.get_l(n-1)
        return adot(l, eps_r_op_1s(r, A, A, op))

    def apply_op_1s(self, op, n, do_update=True):
        nA=np.zeros_like(self.A[n])
        for s in range(self.q[n]):
            for t in range(self.q[n]): nA[s]+=self.A[n][t]*op[s,t]
        self.A[n]=nA
        if do_update: self.update(restore_CF=False)

    def grow_left(self, m):
        D=self.uni_l.D; q=self.uni_l.q
        oA=[self.A[n] for n in range(self.N+3)]; oN=self.N
        self.N+=m; self.N_centre=self.N//2
        self.D=np.full(self.N+2,D,dtype=int); self.q=np.full(self.N+2,q,dtype=int)
        nA=[None]*(self.N+3); nl=[None]*(self.N+3); nr=[None]*(self.N+3)
        for n in range(self.N+3):
            if n<=self.N+1: nl[n]=np.zeros((D,D),dtype=self.typ); nr[n]=np.zeros((D,D),dtype=self.typ)
            if 1<=n<=self.N: nA[n]=np.zeros((q,D,D),dtype=self.typ)
        for n in range(1,m+1): nA[n][:]=self.uni_l.A[0]
        for n in range(m+1,self.N+1): nA[n][:]=oA[n-m]
        self.A=nA; self.l=nl; self.r=nr
        self._AA=[None]*(self.N+2); self._C=[None]*(self.N+2)
        self.K=[None]*(self.N+3); self.K_l=[None]*(self.N+3)
        for n in range(self.N+2):
            self._C[n]=np.zeros((q,q,D,D),dtype=self.typ)
            self._AA[n]=np.zeros((q,q,D,D),dtype=self.typ)
        for n in range(self.N_centre,self.N+2): self.K[n]=np.zeros((D,D),dtype=self.typ)
        for n in range(self.N_centre+1): self.K_l[n]=np.zeros((D,D),dtype=self.typ)
        self.h_expect=np.zeros(self.N+1,dtype=self.typ)
        self.eta_sq=np.zeros(self.N+1,dtype=self.typ)
        self.ham=[self.uni_l.ham]*m+list(self.ham); self.grown_left+=m

    def grow_right(self, m):
        D=self.uni_r.D; q=self.uni_r.q
        oA=[self.A[n] for n in range(self.N+3)]; oN=self.N
        self.N+=m; self.N_centre=self.N//2
        self.D=np.full(self.N+2,D,dtype=int); self.q=np.full(self.N+2,q,dtype=int)
        nA=[None]*(self.N+3); nl=[None]*(self.N+3); nr=[None]*(self.N+3)
        for n in range(self.N+3):
            if n<=self.N+1: nl[n]=np.zeros((D,D),dtype=self.typ); nr[n]=np.zeros((D,D),dtype=self.typ)
            if 1<=n<=self.N: nA[n]=np.zeros((q,D,D),dtype=self.typ)
        for n in range(1,oN+1): nA[n][:]=oA[n]
        for n in range(oN+1,self.N+1): nA[n][:]=self.uni_r.A[0]
        self.A=nA; self.l=nl; self.r=nr
        self._AA=[None]*(self.N+2); self._C=[None]*(self.N+2)
        self.K=[None]*(self.N+3); self.K_l=[None]*(self.N+3)
        for n in range(self.N+2):
            self._C[n]=np.zeros((q,q,D,D),dtype=self.typ)
            self._AA[n]=np.zeros((q,q,D,D),dtype=self.typ)
        for n in range(self.N_centre,self.N+2): self.K[n]=np.zeros((D,D),dtype=self.typ)
        for n in range(self.N_centre+1): self.K_l[n]=np.zeros((D,D),dtype=self.typ)
        self.h_expect=np.zeros(self.N+1,dtype=self.typ)
        self.eta_sq=np.zeros(self.N+1,dtype=self.typ)
        self.ham=list(self.ham)+[self.uni_r.ham]*m; self.grown_right+=m

# ================================================================
# MAIN
# ================================================================
def main():
    import matplotlib.pyplot as plt

    print("="*70)
    print("Spin-1 Heisenberg AFM: Magnon Scattering via Sandwich MPS")
    print("="*70)

    q=3; D=16
    N=120; dt=0.01; steps=300

    sq2=np.sqrt(0.5)
    Sx=sq2*np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=complex)
    Sy=sq2*1j*np.array([[0,1,0],[-1,0,1],[0,-1,0]],dtype=complex)
    Sz=np.array([[1,0,0],[0,0,0],[0,0,-1]],dtype=complex)
    Sp=Sx+1j*Sy; Sm=Sx-1j*Sy
    ham=(np.kron(Sx,Sx)+np.kron(Sy,Sy)+np.kron(Sz,Sz)).reshape(q,q,q,q)

    print(f"\n--- Ground state (D={D}) ---")
    gf=f"heis_ground_D{D}.npy"
    uni=UniformMPS(D,q,ham)
    try:
        uni.load_state(gf); uni.update()
        print(f"Loaded: h={uni.h_expect.real:.10f}")
    except:
        print("Computing ground state...")
        uni=find_ground(uni, tol=1e-5, dtau=0.04, max_steps=5000)
        uni.save_state(gf)
    print(f"h = {uni.h_expect.real:.10f} (exact ≈ -1.401484)")

    print(f"\n--- Sandwich (N={N}) ---")
    sim=SandwichTDVP(N, uni)

    mid=N//2
    sim.apply_op_1s(Sp, mid-15-5, do_update=False)
    sim.apply_op_1s(Sm, mid-15+5, do_update=False)
    sim.apply_op_1s(Sm, mid+15-5, do_update=False)
    sim.apply_op_1s(Sp, mid+15+5, do_update=False)
    sim.update(restore_CF=False)
    print("Excitations applied")

    print(f"\n--- Time evolution (dt={dt}, steps={steps}) ---")
    op_data=[]; t0=time.time()
    for i in range(steps+1):
        if i>0:
            sim.take_step_RK4(dt*1j)

        # Use CF restoration every 4 steps for stability, otherwise just normalize
        rcf = (i > 0) and (i % 4 == 0)
        sim.update(restore_CF=False)
        rng=list(range(-10,sim.N+10))
        op_data.append([sim.expect_1s(Sz,n).real for n in rng])
        if i%50==0:
            print(f"  {i:4d}/{steps}  t={i*dt:.2f}  eta={sim.eta.real:.2e}  "
                  f"dE={np.real(sim.dH_expect):.4e}  N={sim.N}  "
                  f"CPU={time.time()-t0:.0f}s")

    print("\n--- Plotting ---")
    op_data=np.array(op_data)
    fig,ax=plt.subplots(figsize=(12,8))
    vmax = np.max(np.abs(op_data[0])) * 0.7
    im=ax.imshow(op_data, origin='lower', interpolation='none', aspect='auto',
                 extent=(-10,sim.N+9,0,steps*dt), cmap='RdBu_r',
                 vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Site',fontsize=14); ax.set_ylabel('Time t',fontsize=14)
    ax.set_title(f'Magnon Scattering: Spin-1 Heisenberg AFM (D={D})',fontsize=16)
    cb=fig.colorbar(im); cb.set_label(r'$\langle S^z \rangle$',fontsize=14)
    plt.tight_layout()
    plt.savefig('magnon_scattering.png', dpi=150, bbox_inches='tight')
    np.save('magnon_data.npy', op_data)
    print("Saved magnon_scattering.png and magnon_data.npy")
    plt.show()

if __name__=='__main__':
    main()
