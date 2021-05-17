import sys
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
    vr = prim[:, 2]
    vp = prim[:, 3]

    rjph = dat[0]
    piph = dat[3]
    dV = geom.getDV(dat, opts, pars)

    return t, r, phi, sig, vr, vp, rjph, piph, dV, pars, opts


def calcOrb(r, phi, vr, vp, GM=1.0):


    cosp = np.cos(phi)
    sinp = np.sin(phi)
    x = r*cosp
    y = r*sinp

    vx = cosp*vr - r*sinp*vp
    vy = sinp*vr + r*cosp*vp

    v2 = vr*vr + r*r*vp*vp
    j = r*r*vp
    en = 0.5*v2 - GM/r

    p = j**2 / GM

    e = np.sqrt(1 + 2*p*en/GM)
    ex = (v2*x - r*vr*vx) / GM - cosp
    ey = (v2*y - r*vr*vy) / GM - sinp
    phip = np.arctan2(ey, ex)

    a = 1.0 / (-2 * en / GM)

    return x, y, vx, vy, a, p, e, phip, j, en


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


def runTheThing(filenames, Rmax=5.0, fixedPts=False):

    plotDir = Path("plots")
    if not plotDir.exists():
        plotDir.mkdir()

    pts0 = np.stack((np.linspace(1.0, 3.0, 5), np.zeros(5)), axis=-1)

    jgrid = np.linspace(0.0, np.sqrt(Rmax), 111)
    engrid = np.linspace(-1.0, 0.5, 101)
    agrid = np.linspace(0.0, 5.0, 111)
    egrid = np.linspace(0.0, 1.0, 101)

    Nt = len(filenames)
    Npts = pts0.shape[0]
    tmin, tmax = getT(filenames)

    t_i = np.empty(Nt)
    sig_i = np.empty((Nt, Npts))
    en_i = np.empty((Nt, Npts))
    j_i = np.empty((Nt, Npts))
    a_i = np.empty((Nt, Npts))
    p_i = np.empty((Nt, Npts))
    e_i = np.empty((Nt, Npts))
    phip_i = np.empty((Nt, Npts))

    Xe = np.linspace(-Rmax, Rmax, 513)
    Ye = np.linspace(-Rmax, Rmax, 511)
    X, Y = np.meshgrid(0.5*(Xe[1:]+Xe[:-1]), 0.5*(Ye[1:]+Ye[:-1]))
    PHI = np.linspace(0.0, 2*np.pi, 300)

    track = np.empty((Nt, Npts, 2))
    track1 = np.empty((Nt, Npts, 2))
    track2 = np.empty((Nt, Npts, 2))
    track3 = np.empty((Nt, Npts, 2))
    track4 = np.empty((Nt, Npts, 2))

    for i, filename in enumerate(filenames):
        print("Loading", filename)
        t, r, phi, sig, vr, vp, rjph, piph, dV, pars, opts = loadFile(filename)

        x, y, vx, vy, a, p, e, phip, j, en = calcOrb(r, phi, vr, vp)

        inside = r < Rmax

        tri = scipy.spatial.Delaunay(np.stack((x[inside], y[inside]), axis=-1))
        t_i[i] = t-tmin

        if i == 0:
            pts = pts0.copy()
            vxfunc = scipy.interpolate.CloughTocher2DInterpolator(tri,
                                                                  vx[inside])
            vyfunc = scipy.interpolate.CloughTocher2DInterpolator(tri,
                                                                  vy[inside])
            track[i, :, :] = pts
            track1[i, :, :] = pts
            track2[i, :, :] = pts
            track3[i, :, :] = pts
            track4[i, :, :] = pts
        else:
            newvxfunc = scipy.interpolate.CloughTocher2DInterpolator(tri,
                                                                  vx[inside])
            newvyfunc = scipy.interpolate.CloughTocher2DInterpolator(tri,
                                                                  vy[inside])

            if not fixedPts:
                print("Integrating...")
                newpts1 = integrate(t_i[i-1], t_i[i], track1[i-1],
                                    vxfunc, vyfunc)
                newpts2 = integrate(t_i[i-1], t_i[i], track2[i-1],
                                    newvxfunc, newvyfunc)
                newpts3 = integrate(t_i[i-1], t_i[i], track3[i-1],
                                    vxfunc, vyfunc, newvxfunc, newvyfunc)
                newpts4 = integrate(t_i[i-1], t_i[i], track4[i-1],
                                    vxfunc, vyfunc, newvxfunc, newvyfunc, N=1)

                pts = (newpts1 + newpts2 + newpts3 + newpts4)/4

                track[i, :, :] = pts
                track1[i, :, :] = newpts1
                track2[i, :, :] = newpts2
                track3[i, :, :] = newpts3
                track4[i, :, :] = newpts4

            vxfunc = newvxfunc
            vyfunc = newvyfunc

        
        print("Binning...")

        sigf = scipy.interpolate.CloughTocher2DInterpolator(tri, sig[inside])
        sig_i[i, :] = sigf(pts[:, 0], pts[:, 1])
        af = scipy.interpolate.CloughTocher2DInterpolator(tri, a[inside])
        a_i[i, :] = af(pts[:, 0], pts[:, 1])
        pf = scipy.interpolate.CloughTocher2DInterpolator(tri, p[inside])
        p_i[i, :] = pf(pts[:, 0], pts[:, 1])
        ef = scipy.interpolate.CloughTocher2DInterpolator(tri, e[inside])
        e_i[i, :] = ef(pts[:, 0], pts[:, 1])
        jf = scipy.interpolate.CloughTocher2DInterpolator(tri, j[inside])
        j_i[i, :] = jf(pts[:, 0], pts[:, 1])
        enf = scipy.interpolate.CloughTocher2DInterpolator(tri, en[inside])
        en_i[i, :] = enf(pts[:, 0], pts[:, 1])
        phipf = scipy.interpolate.CloughTocher2DInterpolator(tri, phip[inside])
        phip_i[i, :] = phipf(pts[:, 0], pts[:, 1])

        dM = sig*dV

        cav = (r > 1.0) & (r < Rmax)

        dMdjden, _, _ = np.histogram2d(j[cav], en[cav], bins=(jgrid, engrid),
                                       density=True, weights=dM[cav])
        dMdade, _, _ = np.histogram2d(a[cav], e[cav], bins=(agrid, egrid),
                                      density=True, weights=dM[cav])

        fig, ax = plt.subplots(3, 3, figsize=(12, 9))
        figSig, axSig = plt.subplots(1, 1, figsize=(12, 9))

        plot.plotZSlice(fig, ax[0, 0], rjph, piph, r, sig, np.array([0.0]),
                        r"$\Sigma$", pars, opts, log=True, rmax=Rmax)
        plot.plotZSlice(fig, axSig, rjph, piph, r, sig, np.array([0.0]),
                        r"$\Sigma$", pars, opts, log=True, rmax=Rmax)

        ax[0, 1].pcolormesh(jgrid, engrid, dMdjden.T,
                            norm=mpl.colors.LogNorm())
        ax[0, 2].pcolormesh(agrid, egrid, dMdade.T,
                            norm=mpl.colors.LogNorm())

        ax[0, 1].set(xlim=(jgrid[0], jgrid[-1]), xlabel=r'$v_\phi$',
                     ylim=(engrid[0], engrid[-1]), ylabel=r'$\varepsilon$')
        ax[0, 2].set(xlim=(agrid[0], agrid[-1]), xlabel=r'$a$',
                     ylim=(egrid[0], egrid[-1]), ylabel=r'e')

        for j in range(Npts):

            rell = p_i[i, j] / (1 + e_i[i, j]*np.cos(PHI-phip_i[i, j]))

            axSig.plot((rell * np.cos(PHI))[rell>0],
                          (rell * np.sin(PHI))[rell>0], lw=1, color='w')
            l = axSig.plot(pts[j, 0], pts[j, 1], marker='.')
            c = l[0].get_color()
            
            if not fixedPts:
                axSig.plot(track[:i+1, j, 0], track[:i+1, j, 1],
                              marker='', ls='-', lw=0.5, color=c)
                axSig.plot(track1[:i+1, j, 0], track[:i+1, j, 1],
                              marker='', ls=':', lw=0.5, color=c)
                axSig.plot(track2[:i+1, j, 0], track[:i+1, j, 1],
                              marker='', ls=':', lw=0.5, color=c)
                axSig.plot(track3[:i+1, j, 0], track[:i+1, j, 1],
                              marker='', ls=':', lw=0.5, color=c)
                axSig.plot(track4[:i+1, j, 0], track[:i+1, j, 1],
                              marker='', ls=':', lw=0.5, color=c)

            ax[0, 0].plot((rell * np.cos(PHI))[rell>0],
                          (rell * np.sin(PHI))[rell>0], lw=1, color='w')
            ax[0, 0].plot(pts[j, 0], pts[j, 1], marker='.', color=c)
            
            if not fixedPts:
                ax[0, 0].plot(track[:i+1, j, 0], track[:i+1, j, 1],
                              marker='', ls='-', lw=0.5, color=c)
                ax[0, 0].plot(track1[:i+1, j, 0], track[:i+1, j, 1],
                              marker='', ls=':', lw=0.5, color=c)
                ax[0, 0].plot(track2[:i+1, j, 0], track[:i+1, j, 1],
                              marker='', ls=':', lw=0.5, color=c)
                ax[0, 0].plot(track3[:i+1, j, 0], track[:i+1, j, 1],
                              marker='', ls=':', lw=0.5, color=c)
                ax[0, 0].plot(track4[:i+1, j, 0], track[:i+1, j, 1],
                              marker='', ls=':', lw=0.5, color=c)

            ax[0, 1].plot(j_i[:i+1, j], en_i[:i+1, j], color=c)
            ax[0, 1].plot(j_i[i, j], en_i[i, j], marker='.', color=c)
            ax[0, 2].plot(a_i[:i+1, j], e_i[:i+1, j], color=c)
            ax[0, 2].plot(a_i[i, j], e_i[i, j], marker='.', color=c)
            ax[1, 0].plot(t_i[:i+1], sig_i[:i+1, j], color=c)
            ax[1, 0].plot(t_i[i], sig_i[i, j], marker='o', color=c)
            ax[1, 1].plot(t_i[:i+1], j_i[:i+1, j], color=c)
            ax[1, 1].plot(t_i[i], j_i[i, j], marker='o', color=c)
            ax[1, 2].plot(t_i[:i+1], en_i[:i+1, j], color=c)
            ax[1, 2].plot(t_i[i], en_i[i, j], marker='o', color=c)
            ax[2, 0].plot(t_i[:i+1], a_i[:i+1, j], color=c)
            ax[2, 0].plot(t_i[i], a_i[i, j], marker='o', color=c)
            ax[2, 1].plot(t_i[:i+1], e_i[:i+1, j], color=c)
            ax[2, 1].plot(t_i[i], e_i[i, j], marker='o', color=c)
            ax[2, 2].plot(t_i[:i+1], phip_i[:i+1, j], color=c)
            ax[2, 2].plot(t_i[i], phip_i[i, j], marker='o', color=c)


        ax[1, 0].set(xlim=(0.0, tmax-tmin), ylabel=r'$\Sigma$')
        ax[1, 1].set(xlim=(0.0, tmax-tmin), ylabel=r'$v_\phi$')
        ax[1, 2].set(xlim=(0.0, tmax-tmin), ylabel=r'$\varepsilon$')
        ax[2, 0].set(xlim=(0.0, tmax-tmin), ylabel=r'$a$',
                     xlabel=r'$t\ (\Omega_b^{-1})$')
        ax[2, 1].set(xlim=(0.0, tmax-tmin), ylabel=r'$e$',
                     xlabel=r'$t\ (\Omega_b^{-1})$')
        ax[2, 2].set(xlim=(0.0, tmax-tmin), ylabel=r'$\phi_p$',
                     xlabel=r'$t\ (\Omega_b^{-1})$')


        fig.tight_layout()

        figname = plotDir / "frame_{0:04d}.png".format(i)
        print("Saving", figname)
        fig.savefig(figname)
        plt.close(fig)

        figSig.tight_layout()

        figname = plotDir / "frameSigma_{0:04d}.png".format(i)
        print("Saving", figname)
        figSig.savefig(figname)
        plt.close(figSig)




if __name__ == "__main__":

    filenames = [Path(x) for x in sys.argv[1:]]


    runTheThing(filenames)
