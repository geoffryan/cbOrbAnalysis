import sys
import math
from pathlib import Path
import h5py as h5
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.spatial
import discopy.util as util
import discopy.geom as geom
import discopy.plot as plot
import corner


def getT(filenames):

    t = []
    for fname in filenames:
        with h5.File(fname, "r") as f:
            t.append(f['Grid/T'][0])

    t = np.array(t)
    return t.min(), t.max()


def loadFile(filename):

    pars = util.loadPars(filename)
    opts = util.loadOpts(filename)
    t, r, phi, z, prim, dat = util.loadCheckpoint(filename)

    cp = np.cos(phi)
    sp = np.sin(phi)

    sig = prim[:, 0]
    P = prim[:, 1]
    vr = prim[:, 2]
    vp = prim[:, 3]

    rjph = dat[0]
    piph = dat[3]
    dV = geom.getDV(dat, opts, pars)

    pot = getPlanetPot(r, phi, z, dat[4])
        
    vgrad_r, vgrad_p, _ = geom.calculateGrad(r, phi, z,
                                             np.stack((vr, vp), axis=-1),
                                             dat, opts, pars)


    return t, r, phi, sig, P, vr, vp, (vgrad_r, vgrad_p), pot, rjph, piph, dV,\
            dat[4], pars, opts


def getPlanetPot(r, phi, z, planets):

    PhiG = np.zeros(r.shape)

    for ip in range(planets.shape[0]):
        ptype = int(planets[ip, 6])
        GM = planets[ip, 0]
        rp = planets[ip, 3]
        phip = planets[ip, 4]
        eps = planets[ip, 5]

        dx = r*np.cos(phi) - rp*np.cos(phip)
        dy = r*np.sin(phi) - rp*np.sin(phip)
        dz = z
        R = np.sqrt(dx*dx + dy*dy + dz*dz)

        if ptype == 0:
            PhiG += -GM/np.sqrt(R*R + eps*eps)
        elif ptype == 1:
            PhiG += -GM / (R - 2*GM)
        elif ptype == 2:
            PhiG += -GM * R
        elif ptype == 3:
            epsS = eps * 2.8
            u = R / epsS
            A = u < 0.5
            B = (u >= 0.5) & (u < 1.0)
            C = u >= 1.0
            PhiG[A] += GM * (16./3.*u**2 - 48./5.*u**4 + 32./5.*u**5
                             - 14./5.)[A] / eps
            PhiG[B] += GM * (1./(15.*u) + 32./3.*u**2 - 16.*u**3 + 48./5.*u**4
                             - 32./15.*u**5 - 3.2)[B] / eps
            PhiG[C] += -GM/R[C]
        else:
            print("Unknown Planet Type")

    return PhiG


def getPlanetGacc(r, phi, z, planets):

    gr = np.zeros(r.shape)
    gp = np.zeros(r.shape)
    gz = np.zeros(r.shape)

    for ip in range(planets.shape[0]):
        ptype = int(planets[ip, 6])
        GM = planets[ip, 0]
        rp = planets[ip, 3]
        phip = planets[ip, 4]
        eps = planets[ip, 5]

        cosphi = np.cos(phi)
        sinphi = np.sin(phi)

        dx = r*cosphi - rp*np.cos(phip)
        dy = r*sinphi - rp*np.sin(phip)
        dz = z
        Rc = np.sqrt(dx*dx + dy*dy)
        R = np.sqrt(dx*dx + dy*dy + dz*dz)

        if ptype == 0:
            g = -GM*R/np.power(R*R + eps*eps, 1.5)
        elif ptype == 1:
            g = -GM / (R - 2*GM)**2
        elif ptype == 2:
            g = -GM * np.ones(r.shape)
        elif ptype == 3:
            epsS = eps * 2.8
            u = R / epsS
            A = u < 0.5
            B = (u >= 0.5) & (u < 1.0)
            C = u >= 1.0
            g = np.empty(r.shape)

            g[A] = -GM * (32./3.*u - 192./5.*u**3 + 32.*u**4)[A] / eps**2
            g[B] = -GM * (-1./(15.*u**2) + 64./3.*u - 48.*u**2 + 192./5.*u**3
                             - 32./3.*u**4)[B] / eps**2
            g[C] = -GM/R[C]**2
        else:
            g = np.zeros(r.shape)

        gx = g * dx/R
        gy = g * dy/R
        gz += g * dz/R

        gr += gx*cosphi + gy*sinphi
        gp += Rc * (-gx*sinphi + gy*cosphi)

    return gr, gp, gz


def calcOrb(r, phi, vr, vp, pot, GM=1.0):

    aB = 1.0
    OmB = math.sqrt(GM / aB**3)

    cosp = np.cos(phi)
    sinp = np.sin(phi)
    x = r*cosp
    y = r*sinp

    vx = cosp*vr - r*sinp*vp
    vy = sinp*vr + r*cosp*vp

    v2 = vr*vr + r*r*vp*vp
    j = r*r*vp
    en = 0.5*v2 - GM/r
    en2 = 0.5*v2 + pot

    p = j**2 / GM

    e = np.sqrt(1 + 2*p*en/GM)
    ex = (v2*x - r*vr*vx) / GM - cosp
    ey = (v2*y - r*vr*vy) / GM - sinp
    phip = np.arctan2(ey, ex)

    a = 1.0 / (-2 * en / GM)

    CJ = (r*OmB)**2 - 2*pot - v2

    return x, y, vx, vy, a, p, e, phip, j, en, en2, CJ


def integrate(ta, tb, pts, vxfunc_a, vyfunc_a, vxfunc_b=None, vyfunc_b=None,
              N=10):

    t = np.linspace(ta, tb, N+1)

    x = pts[:, 0].copy()
    y = pts[:, 1].copy()

    def getv(t, x, y):
        vx_a = vxfunc_a(x, y)
        vy_a = vyfunc_a(x, y)
        vx_b = vxfunc_b(x, y) if vxfunc_b is not None else vx_a
        vy_b = vyfunc_b(x, y) if vyfunc_b is not None else vy_a
        vx = ((tb-t) * vx_a + (t-ta) * vx_b) / (tb-ta)
        vy = ((tb-t) * vy_a + (t-ta) * vy_b) / (tb-ta)
        return vx, vy

    """
    vx, vy = getv(ta, x, y)
    print("V", vx, vy)
    x += (tb-ta)*vx
    y += (tb-ta)*vy
    """

    for i in range(N):

        dt = t[i+1] - t[i]
        vx1, vy1 = getv(t[i], x, y)
        vx2, vy2 = getv(t[i] + 0.5*dt, x + 0.5*dt*vx1, y + 0.5*dt*vy1)
        vx3, vy3 = getv(t[i] + 0.5*dt, x + 0.5*dt*vx2, y + 0.5*dt*vy2)
        vx4, vy4 = getv(t[i] + dt, x + dt*vx3, y + dt*vy3)

        x += dt * (vx1 + 2*vx2 + 2*vx3 + vx4) / 6.0
        y += dt * (vy1 + 2*vy2 + 2*vy3 + vy4) / 6.0

    return np.stack((x, y), axis=-1)


def runTheThing(filenames, Rmax=5.0, fixedPts=False,
                h5name='particleTracks.h5'):

    PHI = np.linspace(0.0, 2*np.pi, 5, endpoint=False)

    X1 = 2 * np.cos(PHI)
    X2 = 3 * np.cos(PHI + np.pi)
    Y1 = 2 * np.sin(PHI)
    Y2 = 3 * np.sin(PHI + np.pi)
    pts0 = np.stack((np.concatenate((X1, X2)), np.concatenate((Y1, Y2))),
                    axis=-1)

    # pts0 = np.stack((np.linspace(1.0, 3.0, 5), np.zeros(5)), axis=-1)

    jgrid = np.linspace(0.0, np.sqrt(Rmax), 111)
    engrid = np.linspace(-1.0, 0.5, 101)
    agrid = np.linspace(0.0, 5.0, 111)
    egrid = np.linspace(0.0, 1.0, 101)

    Nt = len(filenames)
    Npts = pts0.shape[0]
    tmin, tmax = getT(filenames)

    t_i = np.empty(Nt)

    Xe = np.linspace(-Rmax, Rmax, 513)
    Ye = np.linspace(-Rmax, Rmax, 511)
    X, Y = np.meshgrid(0.5*(Xe[1:]+Xe[:-1]), 0.5*(Ye[1:]+Ye[:-1]))
    PHI = np.linspace(0.0, 2*np.pi, 300)

    track = np.empty((Nt, Npts, 2))
    track1 = np.empty((Nt, Npts, 2))
    track2 = np.empty((Nt, Npts, 2))
    track3 = np.empty((Nt, Npts, 2))
    track4 = np.empty((Nt, Npts, 2))

    with h5.File(h5name, "w") as f:
        f.create_group('tracks')
        f.create_group('hist')
        f.create_dataset('tracks/t', shape=(Nt), dtype=float)
        f.create_dataset('tracks/xy', shape=(Npts, Nt, 2), dtype=float)
        f.create_dataset('tracks/xy1', shape=(Npts, Nt, 2), dtype=float)
        f.create_dataset('tracks/xy2', shape=(Npts, Nt, 2), dtype=float)
        f.create_dataset('tracks/xy3', shape=(Npts, Nt, 2), dtype=float)
        f.create_dataset('tracks/Sig', shape=(Npts, Nt), dtype=float)
        f.create_dataset('tracks/Pi', shape=(Npts, Nt), dtype=float)
        f.create_dataset('tracks/vx', shape=(Npts, Nt), dtype=float)
        f.create_dataset('tracks/vy', shape=(Npts, Nt), dtype=float)
        f.create_dataset('tracks/a', shape=(Npts, Nt), dtype=float)
        f.create_dataset('tracks/e', shape=(Npts, Nt), dtype=float)
        f.create_dataset('tracks/p', shape=(Npts, Nt), dtype=float)
        f.create_dataset('tracks/j', shape=(Npts, Nt), dtype=float)
        f.create_dataset('tracks/en', shape=(Npts, Nt), dtype=float)
        f.create_dataset('tracks/en2', shape=(Npts, Nt), dtype=float)
        f.create_dataset('tracks/CJ', shape=(Npts, Nt), dtype=float)
        f.create_dataset('tracks/pot', shape=(Npts, Nt), dtype=float)
        f.create_dataset('tracks/phip', shape=(Npts, Nt), dtype=float)
        f.create_dataset('tracks/vort', shape=(Npts, Nt), dtype=float)
        f.create_dataset('hist/jgrid', data=jgrid)
        f.create_dataset('hist/engrid', data=engrid)
        f.create_dataset('hist/agrid', data=agrid)
        f.create_dataset('hist/egrid', data=egrid)
        f.create_dataset('hist/dMdjden', shape=(len(jgrid)-1, len(engrid)-1),
                         dtype=float)
        f.create_dataset('hist/dMdjden2', shape=(len(jgrid)-1, len(engrid)-1),
                         dtype=float)
        f.create_dataset('hist/dMdade', shape=(len(agrid)-1, len(egrid)-1),
                         dtype=float)
        f['tracks/t'][:] = 0.0
        f['hist/dMdjden'][: , :] = 0.0
        f['hist/dMdjden2'][: , :] = 0.0
        f['hist/dMdade'][: , :] = 0.0

    for i, filename in enumerate(filenames):
        print("Loading", filename)
        t, r, phi, sig, Pi, vr, vp, vgrad, pot, rjph, piph, dV, plDat,\
                pars, opts = loadFile(filename)

        x, y, vx, vy, a, p, e, phip, j, en, en2, CJ\
                = calcOrb(r, phi, vr, vp, pot)

        dvrdr = vgrad[0][:, 0]
        dvrdp = vgrad[1][:, 0]
        dvpdr = vgrad[0][:, 1]
        dvpdp = vgrad[1][:, 1]

        inside = r < Rmax

        tri = scipy.spatial.Delaunay(np.stack((x[inside], y[inside]), axis=-1))
        t_i[i] = t-tmin

        if i == 0:
            pts = pts0.copy()
            vxfunc = scipy.interpolate.CloughTocher2DInterpolator(tri,
                                                                  vx[inside])
            vyfunc = scipy.interpolate.CloughTocher2DInterpolator(tri,
                                                                  vy[inside])
            track[0, :, :] = pts
            track1[0, :, :] = pts
            track2[0, :, :] = pts
            track3[0, :, :] = pts
            pts1 = pts
            pts2 = pts
            pts3 = pts

        else:
            newvxfunc = scipy.interpolate.CloughTocher2DInterpolator(tri,
                                                                  vx[inside])
            newvyfunc = scipy.interpolate.CloughTocher2DInterpolator(tri,
                                                                  vy[inside])

            if not fixedPts:
                print("Integrating...")
                pts = integrate(t_i[i-1], t_i[i], track[i-1],
                                vxfunc, vyfunc, newvxfunc, newvyfunc)
                pts1 = integrate(t_i[i-1], t_i[i], track1[i-1],
                                 vxfunc, vyfunc)
                pts2 = integrate(t_i[i-1], t_i[i], track2[i-1],
                                 newvxfunc, newvyfunc)
                pts3 = integrate(t_i[i-1], t_i[i], track3[i-1],
                                 vxfunc, vyfunc, newvxfunc, newvyfunc, N=1)

                track[i, :, :] = pts
                track1[i, :, :] = pts1
                track2[i, :, :] = pts2
                track3[i, :, :] = pts3

            vxfunc = newvxfunc
            vyfunc = newvyfunc

        curl_v = r * dvpdr + 2 * vp - dvrdp/r
        vort = curl_v / sig

        print("Interpolating...") 

        sigf = scipy.interpolate.CloughTocher2DInterpolator(tri, sig[inside])
        sig_i = sigf(pts[:, 0], pts[:, 1])
        vx_i = vxfunc(pts[:, 0], pts[:, 1])
        vy_i = vyfunc(pts[:, 0], pts[:, 1])
        Pif = scipy.interpolate.CloughTocher2DInterpolator(tri, Pi[inside])
        Pi_i = Pif(pts[:, 0], pts[:, 1])
        af = scipy.interpolate.CloughTocher2DInterpolator(tri, a[inside])
        a_i = af(pts[:, 0], pts[:, 1])
        pf = scipy.interpolate.CloughTocher2DInterpolator(tri, p[inside])
        p_i = pf(pts[:, 0], pts[:, 1])
        ef = scipy.interpolate.CloughTocher2DInterpolator(tri, e[inside])
        e_i = ef(pts[:, 0], pts[:, 1])
        jf = scipy.interpolate.CloughTocher2DInterpolator(tri, j[inside])
        j_i = jf(pts[:, 0], pts[:, 1])
        enf = scipy.interpolate.CloughTocher2DInterpolator(tri, en[inside])
        en_i = enf(pts[:, 0], pts[:, 1])
        en2f = scipy.interpolate.CloughTocher2DInterpolator(tri, en2[inside])
        en2_i = en2f(pts[:, 0], pts[:, 1])
        CJf = scipy.interpolate.CloughTocher2DInterpolator(tri, CJ[inside])
        CJ_i = CJf(pts[:, 0], pts[:, 1])
        potf = scipy.interpolate.CloughTocher2DInterpolator(tri, pot[inside])
        pot_i = potf(pts[:, 0], pts[:, 1])
        phipf = scipy.interpolate.CloughTocher2DInterpolator(tri, phip[inside])
        phip_i = phipf(pts[:, 0], pts[:, 1])
        vortf = scipy.interpolate.CloughTocher2DInterpolator(tri, vort[inside])
        vort_i = vortf(pts[:, 0], pts[:, 1])

        print("Binning...")

        dM = sig*dV

        cav = (r > 1.0) & (r < Rmax)

        dMdjden, _, _ = np.histogram2d(j[cav], en[cav], bins=(jgrid, engrid),
                                       density=True, weights=dM[cav])
        dMdjden2, _, _ = np.histogram2d(j[cav], en2[cav], bins=(jgrid, engrid),
                                        density=True, weights=dM[cav])
        dMdade, _, _ = np.histogram2d(a[cav], e[cav], bins=(agrid, egrid),
                                      density=True, weights=dM[cav])

        print("Writing to", h5name)
        with h5.File(h5name, 'r+') as f:
            f['tracks/t'][i] = t
            f['tracks/xy'][:, i, :] = pts
            f['tracks/xy1'][:, i, :] = pts1
            f['tracks/xy2'][:, i, :] = pts2
            f['tracks/xy3'][:, i, :] = pts3
            f['tracks/Sig'][:, i] = sig_i
            f['tracks/Pi'][:, i] = Pi_i
            f['tracks/vx'][:, i] = vx_i
            f['tracks/vy'][:, i] = vy_i
            f['tracks/a'][:, i] = a_i
            f['tracks/e'][:, i] = e_i
            f['tracks/p'][:, i] = p_i
            f['tracks/j'][:, i] = j_i
            f['tracks/en'][:, i] = en_i
            f['tracks/en2'][:, i] = en2_i
            f['tracks/CJ'][:, i] = CJ_i
            f['tracks/pot'][:, i] = pot_i
            f['tracks/phip'][:, i] = phip_i
            f['tracks/vort'][:, i] = vort_i

            f['hist/dMdjden'][:, :] = (i*f['hist/dMdjden'][:, :] + dMdjden
                                       ) / (i + 1)
            f['hist/dMdjden2'][:, :] = (i*f['hist/dMdjden2'][:, :] + dMdjden2
                                        ) / (i + 1)
            f['hist/dMdade'][:, :] = (i*f['hist/dMdade'][:, :] + dMdade
                                      ) / (i + 1)

    return h5name


def plotTheThing(h5name, Rmax=5.0):

    plotDir = Path("plots")
    if not plotDir.exists():
        plotDir.mkdir()

    with h5.File(h5name, "r") as f:
        t = f['tracks/t'][...]
        xy = f['tracks/xy'][...]
        xy1 = f['tracks/xy1'][...]
        xy2 = f['tracks/xy2'][...]
        xy3 = f['tracks/xy3'][...]
        Sig = f['tracks/Sig'][...]
        Pi = f['tracks/Pi'][...]
        vx = f['tracks/vx'][...]
        vy = f['tracks/vy'][...]
        a = f['tracks/a'][...]
        e = f['tracks/e'][...]
        p = f['tracks/p'][...]
        j = f['tracks/j'][...]
        en = f['tracks/en'][...]
        en2 = f['tracks/en2'][...]
        CJ = f['tracks/CJ'][...]
        pot = f['tracks/pot'][...]
        vort = f['tracks/vort'][...]
        phip = f['tracks/phip'][...]

        jgrid = f['hist/jgrid'][...]
        engrid = f['hist/engrid'][...]
        agrid = f['hist/agrid'][...]
        egrid = f['hist/egrid'][...]

        dMdjden = f['hist/dMdjden'][...]
        dMdjden2 = f['hist/dMdjden2'][...]
        dMdade = f['hist/dMdade'][...]


    Npts, Nt = Sig.shape
    tmin = t.min()
    tmax = t.max()
    
    t0 = t - tmin
    NT = len(t[t>0])

    fig, ax = plt.subplots(3, 4, figsize=(12, 9))
    figSig, axSig = plt.subplots(1, 1, figsize=(12, 9))

    ax[0, 0].pcolormesh(jgrid, engrid, dMdjden.T,
                        norm=mpl.colors.LogNorm(),
                        rasterized=True)
    ax[0, 1].pcolormesh(jgrid, engrid, dMdjden2.T,
                        norm=mpl.colors.LogNorm(),
                        rasterized=True)
    ax[0, 2].pcolormesh(agrid, egrid, dMdade.T,
                        norm=mpl.colors.LogNorm(),
                        rasterized=True)

    colors = ['C{0:d}'.format(i) for i in range(10)]

    for pt in range(Npts):

        # rell = p_i[i, j] / (1 + e_i[i, j]*np.cos(PHI-phip_i[i, j]))

        # l = axSig.plot(pts[j, 0], pts[j, 1], marker='.')
        c = colors[pt]
        
        ax[0, 0].plot(j[pt, :NT], en[pt, :NT], color=c, lw=1)
        ax[0, 1].plot(j[pt, :NT], en2[pt, :NT], color=c, lw=1)
        ax[0, 2].plot(a[pt, :NT], e[pt, :NT], color=c, lw=1)
        ax[0, 3].plot(t0[:NT], Sig[pt, :NT], color=c)
        ax[1, 0].plot(t0[:NT], j[pt, :NT], color=c)
        ax[1, 1].plot(t0[:NT], en[pt, :NT], color=c)
        ax[1, 2].plot(t0[:NT], en2[pt, :NT], color=c)
        ax[1, 3].plot(t0[:NT], vort[pt, :NT], color=c)
        ax[2, 0].plot(t0[:NT], a[pt, :NT], color=c)
        ax[2, 1].plot(t0[:NT], e[pt, :NT], color=c)
        ax[2, 2].plot(t0[:NT], phip[pt, :NT], color=c)
        ax[2, 3].plot(t0[:NT], CJ[pt, :NT], color=c)

    ax[0, 0].set(xlim=(jgrid[0], jgrid[-1]), xlabel=r'$v_\phi$',
                 ylim=(engrid[0], engrid[-1]), ylabel=r'$\varepsilon_1$')
    ax[0, 1].set(xlim=(jgrid[0], jgrid[-1]), xlabel=r'$v_\phi$',
                 ylim=(engrid[0], engrid[-1]), ylabel=r'$\varepsilon$')
    ax[0, 2].set(xlim=(agrid[0], agrid[-1]), xlabel=r'$a$',
                 ylim=(egrid[0], egrid[-1]), ylabel=r'e')

    ax[0, 3].set(xlim=(0.0, tmax-tmin), ylabel=r'$\Sigma$')

    ax[1, 0].set(xlim=(0.0, tmax-tmin), ylabel=r'$v_\phi$',
                 ylim=(jgrid[0], jgrid[-1]))
    ax[1, 1].set(xlim=(0.0, tmax-tmin), ylabel=r'$\varepsilon_1$',
                 ylim=(engrid[0], engrid[-1]))
    ax[1, 2].set(xlim=(0.0, tmax-tmin), ylabel=r'$\varepsilon$',
                 ylim=(engrid[0], engrid[-1]))
    ax[1, 3].set(xlim=(0.0, tmax-tmin), ylabel=r'$\nabla \times v / \Sigma$')
    ax[2, 0].set(xlim=(0.0, tmax-tmin), ylabel=r'$a$',
                 xlabel=r'$t\ (\Omega_b^{-1})$', ylim=(agrid[0], agrid[-1]))
    ax[2, 1].set(xlim=(0.0, tmax-tmin), ylabel=r'$e$',
                 xlabel=r'$t\ (\Omega_b^{-1})$', ylim=(egrid[0], egrid[-1]))
    ax[2, 2].set(xlim=(0.0, tmax-tmin), ylabel=r'$\phi_p$',
                 xlabel=r'$t\ (\Omega_b^{-1})$')
    ax[2, 3].set(xlim=(0.0, tmax-tmin), ylabel=r'$C_J$',
                 xlabel=r'$t\ (\Omega_b^{-1})$')


    fig.tight_layout()

    figname = plotDir / "timeseries_full.pdf"
    print("Saving", figname)
    fig.savefig(figname)
    plt.close(fig)


if __name__ == "__main__":

    filenames = [Path(x) for x in sys.argv[1:]]

    Rmax = 5.0
    fixedPts = False

    if len(filenames) > 1:
        h5name = runTheThing(filenames, Rmax=Rmax, fixedPts=fixedPts)
        plotTheThing(h5name)
    else:
        plotTheThing(filenames[0])
