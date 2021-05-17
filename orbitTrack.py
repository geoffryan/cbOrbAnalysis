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


def runTheThing(filenames, Rmax=5.0):

    plotDir = Path("plots")
    if not plotDir.exists():
        plotDir.mkdir()

    pts = np.stack((np.linspace(1.0, 3.0, 5), np.zeros(5)), axis=-1)

    jgrid = np.linspace(0.0, np.sqrt(Rmax), 111)
    engrid = np.linspace(-1.0, 0.5, 101)
    agrid = np.linspace(0.0, 5.0, 111)
    egrid = np.linspace(0.0, 1.0, 101)

    Nt = len(filenames)
    Npts = pts.shape[0]
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

    for i, filename in enumerate(filenames):
        t, r, phi, sig, vr, vp, rjph, piph, dV, pars, opts = loadFile(filename)

        x, y, vx, vy, a, p, e, phip, j, en = calcOrb(r, phi, vr, vp)

        tri = scipy.spatial.Delaunay(np.stack((x, y), axis=-1))

        t_i[i] = t-tmin
        sigf = scipy.interpolate.CloughTocher2DInterpolator(tri, sig)
        sig_i[i, :] = sigf(pts[:, 0], pts[:, 1])
        af = scipy.interpolate.CloughTocher2DInterpolator(tri, a)
        a_i[i, :] = af(pts[:, 0], pts[:, 1])
        pf = scipy.interpolate.CloughTocher2DInterpolator(tri, p)
        p_i[i, :] = pf(pts[:, 0], pts[:, 1])
        ef = scipy.interpolate.CloughTocher2DInterpolator(tri, e)
        e_i[i, :] = ef(pts[:, 0], pts[:, 1])
        jf = scipy.interpolate.CloughTocher2DInterpolator(tri, j)
        j_i[i, :] = jf(pts[:, 0], pts[:, 1])
        enf = scipy.interpolate.CloughTocher2DInterpolator(tri, en)
        en_i[i, :] = enf(pts[:, 0], pts[:, 1])
        phipf = scipy.interpolate.CloughTocher2DInterpolator(tri, phip)
        phip_i[i, :] = phipf(pts[:, 0], pts[:, 1])

        dM = sig*dV

        cav = (r > 1.0) & (r < Rmax)

        dMdjden, _, _ = np.histogram2d(j[cav], en[cav], bins=(jgrid, engrid),
                                       density=True, weights=dM[cav])
        dMdade, _, _ = np.histogram2d(a[cav], e[cav], bins=(agrid, egrid),
                                      density=True, weights=dM[cav])

        fig, ax = plt.subplots(3, 3, figsize=(12, 9))

        plot.plotZSlice(fig, ax[0, 0], rjph, piph, r, sig, np.array([0.0]),
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

            ax[0, 0].plot((rell * np.cos(PHI))[rell>0],
                          (rell * np.sin(PHI))[rell>0], lw=1, color='w')
            l = ax[0, 0].plot(pts[j, 0], pts[j, 1], marker='.')

            ax[0, 1].plot(j_i[i, j], en_i[i, j], marker='.',
                          color=l[0].get_color())
            ax[0, 2].plot(a_i[i, j], e_i[i, j], marker='.',
                          color=l[0].get_color())
            ax[1, 0].plot(t_i[:i+1], sig_i[:i+1, j])
            ax[1, 0].plot(t_i[i], sig_i[i, j], marker='o',
                          color=l[0].get_color())
            ax[1, 1].plot(t_i[:i+1], j_i[:i+1, j],
                          color=l[0].get_color())
            ax[1, 1].plot(t_i[i], j_i[i, j], marker='o',
                          color=l[0].get_color())
            ax[1, 2].plot(t_i[:i+1], en_i[:i+1, j],
                          color=l[0].get_color())
            ax[1, 2].plot(t_i[i], en_i[i, j], marker='o',
                          color=l[0].get_color())
            ax[2, 0].plot(t_i[:i+1], a_i[:i+1, j],
                          color=l[0].get_color())
            ax[2, 0].plot(t_i[i], a_i[i, j], marker='o',
                          color=l[0].get_color())
            ax[2, 1].plot(t_i[:i+1], e_i[:i+1, j],
                          color=l[0].get_color())
            ax[2, 1].plot(t_i[i], e_i[i, j], marker='o',
                          color=l[0].get_color())
            ax[2, 2].plot(t_i[:i+1], phip_i[:i+1, j],
                          color=l[0].get_color())
            ax[2, 2].plot(t_i[i], phip_i[i, j], marker='o',
                          color=l[0].get_color())


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




if __name__ == "__main__":

    filenames = [Path(x) for x in sys.argv[1:]]


    runTheThing(filenames)
