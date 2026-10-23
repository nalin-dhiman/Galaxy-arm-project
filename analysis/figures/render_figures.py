#!/usr/bin/env python3
"""Rebuild all nine paper figures from compact, attributed derived plot arrays.

Requires Python3, NumPy and Matplotlib only. No model fitting, raw-data download,
serialized object loading, workspace imports or network access occurs.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,os,tempfile
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"galaxy_portable_figure_fonts"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.text import Text

HERE=Path(__file__).resolve().parent
OUT=HERE/"output"
BLUE="#12628b";ORANGE="#c16a1b";CYAN="#25d0d0";GRAY="#dedede";PURPLE="#73528e"
RECORDS=[];QA=[];CACHE=None;STATS=None

class Column(np.ndarray):
    def __new__(cls,value):return np.asarray(value).view(cls)
    def to_numpy(self,dtype=None):return np.asarray(self,dtype=dtype)

class Rows:
    def __init__(self,table):self.table=table
    def __getitem__(self,key):
        mask,column=key
        return self.table[column][mask]

class Table:
    """Tiny NumPy presentation adapter; no dataframe library is required."""
    def __init__(self,data):self.data={key:np.asarray(value).view(Column) for key,value in data.items()}
    def __len__(self):return len(next(iter(self.data.values())))
    def __getitem__(self,key):
        if isinstance(key,str):return self.data[key]
        return Table({name:value[key] for name,value in self.data.items()})
    def __getattr__(self,key):
        if key=="loc":return Rows(self)
        return self.data[key]
    def copy(self):return Table({name:value.copy() for name,value in self.data.items()})

def readjson(name):return STATS[name]

def readcsv(name):
    if name=="redshift_predictions":
        return Table({"sample_group":np.full(len(CACHE["redshift_z"]),"strong_spiral"),"spec_z":CACHE["redshift_z"],
            "PGI_predicted_catalog_redshift":CACHE["redshift_PGI"],"PGIA_predicted_catalog_redshift":CACHE["redshift_PGIA"],
            "PGI_missing_predictors":CACHE["redshift_missing"]})
    if name=="btfr_predictions":
        mapping={"reference_mpc":"reference","reference_error_mpc":"reference_error","distance_mpc":"distance",
            "model_interval_90_lower_mpc":"lower90","model_interval_90_upper_mpc":"upper90",
            "outside_velocity_calibration_range":"extrapolation"}
        return Table({name:CACHE["btfr_"+key] for name,key in mapping.items()})
    raise ValueError("Unknown public plot table")

def load_example(identifier):
    meta=STATS["example_metadata"][identifier];prefix=meta["prefix"]
    return {**meta,"rgb":CACHE[prefix+"_rgb"],"valid":CACHE[prefix+"_valid"],"votes":CACHE[prefix+"_votes"],
        "extent":CACHE[prefix+"_extent"],"result":{"probability":CACHE[prefix+"_raw_score"]}}

class CachedHalo:
    def density(self,r,z):return CACHE["halo_density"]
    def pressure(self,r,z):return CACHE["halo_pressure"]
    def local_dispersion_squared(self,r,z):return CACHE["halo_sigma2"]
    def equatorial_circular_velocity(self,r):return CACHE["halo_velocity"]


def qa_figure(fig,name):
    """Record actual rendered text collisions; human raster review follows."""
    fig.canvas.draw();renderer=fig.canvas.get_renderer();texts=[]
    # Matplotlib retains Text objects for unused ticks beyond the visible view.
    # Axis.draw excludes them; do not flag their undrawn boxes as collisions.
    undrawn=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            low,high=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not low-1e-10<=tick.get_loc()<=high+1e-10:
                    undrawn.update([id(tick.label1),id(tick.label2)])
    for artist in fig.findobj(match=Text):
        if id(artist) not in undrawn and artist.get_visible() and artist.get_text().strip():
            box=artist.get_window_extent(renderer)
            if box.width>0 and box.height>0:texts.append((artist.get_text(),box))
    collisions=[]
    for i,(a,ba) in enumerate(texts):
        for b,bb in texts[i+1:]:
            ix=min(ba.x1,bb.x1)-max(ba.x0,bb.x0);iy=min(ba.y1,bb.y1)-max(ba.y0,bb.y0)
            if ix>1.5 and iy>1.5:
                collisions.append({"first":a,"second":b,"intersection_px":[round(ix,2),round(iy,2)]})
    outside=[]
    for text,box in texts:
        if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5:
            outside.append(text)
    QA.append({"figure":name,"text_artists":len(texts),"text_bbox_collisions":collisions,"text_outside_nominal_canvas":outside})

def save(fig,name,caption,*,scope="main",credits="Original analysis and visualization",sources=(),ids=()):
    qa_figure(fig,name)
    for ext in ["pdf","png"]:fig.savefig(OUT/(name+"."+ext),dpi=240,bbox_inches="tight",pad_inches=.06)
    plt.close(fig)
    RECORDS.append({"name":name,"scope":scope,"pdf":name+".pdf","png":name+".png",
                    "caption":caption,"credits":credits,"source_urls":list(sources),"fixed_example_ids":list(ids)})

def standard(ncols=2,width=7.4,height=3.9,bottom=.27,wspace=.43):
    fig,axes=plt.subplots(1,ncols,figsize=(width,height),squeeze=False)
    fig.subplots_adjust(left=.10,right=.98,top=.90,bottom=bottom,wspace=wspace)
    for ax in axes.flat:ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
    return fig,axes.flat

def legend(fig,handles,labels,ncol=2,y=.015):
    return fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,y),ncol=ncol,
                      frameon=False,columnspacing=1.6,handlelength=2.5,fontsize=8.3)

def pipeline():
    fig=plt.figure(figsize=(7.4,5.0));ax=fig.add_axes([.02,.03,.96,.94]);ax.set(xlim=(0,10),ylim=(0,7));ax.axis("off")
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.10,rounding_size=.10",facecolor=color,edgecolor="#607580",lw=.8))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=9.0,linespacing=1.4)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=12,color="#526d7c",lw=1.1))
    box(.1,5.45,4.2,1.1,"Original resolved-spiral cohort\n160 galaxies; 64 sky groups", "#e5eff4")
    box(5.7,5.45,4.2,1.1,"Fresh spatial/population cohort\n99 galaxies; 39 new sky groups", "#f5eadc")
    box(.1,3.7,2.0,1.0,"TRAIN\n91 / 38 groups","#e5eff4")
    box(2.55,3.7,2.0,1.0,"CALIBRATION\n40 / 13 groups","#e5eff4")
    box(5.0,3.7,2.0,1.0,"ORIGINAL TEST\n29 / 13 groups","#e5eff4")
    box(7.45,3.7,2.45,1.0,"FRESH STRONG\n59 / 30 groups","#f5eadc")
    arrow((2.2,5.4),(1.1,4.8));arrow((2.2,5.4),(3.5,4.8));arrow((4.2,5.7),(6,4.8));arrow((7.8,5.4),(8.6,4.8))
    box(.1,1.8,4.45,1.1,"Native grz + variance + local PSF\n44 fixed intensity/shape/color features\nHuman centers/star masks supplied", "#edf2ed")
    box(5.0,1.8,4.9,1.1,"Frozen redshift ablations\nPhotometry + shared observing controls\nPaired comparison: basic images vs + arms", "#edf2ed")
    arrow((1.1,3.6),(2.3,3.0));arrow((3.5,3.6),(3.5,3.0));arrow((6,3.6),(7.2,3.0));arrow((8.6,3.6),(8.6,3.0));arrow((4.65,2.35),(4.95,2.35))
    ax.text(5,.90,"Human polygons supervise training and score held-out agreement; never distance predictors.",ha="center",fontsize=8.6)
    ax.text(5,.36,"Redshift is a catalogue label. Angular structure does not supply an absolute distance ruler.",ha="center",fontsize=8.6)
    save(fig,"pipeline_splits","Data separation and conditional inference. The original 160 selected resolved spirals occupy 64 nonoverlapping ten-degree sky groups, split into 91 training, 40 calibration and 29 test galaxies. The separate fresh population cohort has 99 identities in 39 previously unused groups; its prespecified primary strong-spiral stratum contains 59 galaxies in 30 groups. Frozen image classifiers use native grz science, inverse variance, local PSFs and supplied human center/foreground marks. Human arm polygons are evaluation references and never redshift predictors. The 99-object cohort is a same-survey spatial/population validation, not an independent telescope survey.")

def arm_examples():
    # These identities are fixed entries of the pre-scoring external gallery:
    # first two strong, first bar-proxy, first unknown. No score-based choices.
    protocol=readjson("external_protocol")
    ids=[protocol["gallery_ids"][i] for i in [0,1,4,6]]
    objects=[load_example(i) for i in ids]
    fig=plt.figure(figsize=(7.4,10.4));gs=fig.add_gridspec(4,3,left=.09,right=.96,top=.95,bottom=.13,wspace=.08,hspace=.22)
    cmap=plt.get_cmap("magma").copy();cmap.set_bad(GRAY)
    refmap=matplotlib.colors.ListedColormap(["#fafafa","#167b64"]);refmap.set_bad(GRAY)
    axes=[]
    for row,item in enumerate(objects):
        for col in range(3):
            ax=fig.add_subplot(gs[row,col]);axes.append(ax)
            if col==0:ax.imshow(item["rgb"],origin="lower",extent=item["extent"],interpolation="nearest")
            elif col==1:ax.imshow(np.where(item["valid"],item["result"]["probability"],np.nan),origin="lower",extent=item["extent"],cmap=cmap,vmin=0,vmax=1,interpolation="nearest")
            elif item["annotated"]:ax.imshow(np.where(item["valid"],item["votes"]>=3,np.nan),origin="lower",extent=item["extent"],cmap=refmap,vmin=0,vmax=1,interpolation="nearest")
            else:
                ax.set_facecolor(GRAY);ax.text(.5,.5,"Spatial arm task\nnot supplied\n\nTRUTH UNKNOWN",ha="center",va="center",transform=ax.transAxes,fontsize=8.6)
            ax.set(xlim=(-28,28),ylim=(-28,28),xticks=[-20,0,20],yticks=[-20,0,20])
            ax.tick_params(labelsize=7,length=2)
            if row==0:ax.set_title(["Native grz display","Raw classifier score","Human consensus region"][col],fontsize=10,pad=9)
            if col==0:ax.set_ylabel(item["identifier"]+"\nOffset (arcsec)",fontsize=8.3)
            else:ax.set_yticklabels([])
            if row==3:ax.set_xlabel("Image-frame offset (arcsec)",fontsize=8.1)
            else:ax.set_xticklabels([])
    cax=fig.add_axes([.29,.077,.40,.012]);fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,1),cmap=cmap),cax=cax,orientation="horizontal",ticks=[0,.5,1],label="Raw score; not physical arm probability")
    legend(fig,[Patch(color=GRAY),Patch(facecolor="#fafafa",edgecolor=".5"),Patch(color="#167b64")],["No comparable pixels","Unmarked reference pixels","At least 3 human votes"],ncol=3,y=.014)
    fig.text(.5,.004,"Legacy Surveys / D. Lang (Perimeter Institute); human marks: SDSS-IV / Galaxy Zoo 3D volunteers",ha="center",fontsize=7.4)
    save(fig,"arm_region_examples","Fixed galaxy examples from the pre-scoring gallery: the first two strong spirals, first weak/bar-proxy object and first unasked control. Left: an asinh grz display made from native DR9 science planes. Middle: the frozen TRAIN-only branch's raw classifier score, without probability calibration. Right: independently supplied regions with at least three Galaxy Zoo 3D votes, mapped through celestial WCS. Gray denotes pixels outside the common analysis/reference footprint or excluded foreground regions. White denotes valid unmarked reference pixels, not established physical negatives. No spatial task means unknown truth, not a negative label; the first object's lack of a two-class consensus is retained. Native local PSFs and variance maps are used by the frozen reader, and the science images are not homogenized for display. Scores indicate annotation agreement under a selected population, not physical arm certainty. Visible image credit: Legacy Surveys / D. Lang (Perimeter Institute); human annotations: SDSS-IV and Galaxy Zoo 3D volunteers.",credits="Legacy Surveys / D. Lang (Perimeter Institute); SDSS-IV / Galaxy Zoo 3D volunteers; original derived score visualization",sources=["https://www.legacysurvey.org/acknowledgment/","https://www.sdss4.org/dr17/manga/manga-data/manga-gz3d/","https://arxiv.org/abs/2108.02065"],ids=ids)

def arm_validation():
    old=readjson("arm_original");fresh=readjson("arm_fresh")
    fig,axes=standard(width=7.8,height=3.9,wspace=.72);a,b=list(axes)
    keys=["raw_g_AP","circular_profile_AP","brightness_radius_baseline_AP"]
    labels=["g brightness","Circular radial profile","Learned brightness\n+ radius"]
    for y,k in enumerate(keys):
        r=old["sky_group_cluster_bootstrap"][k];a.plot(r["bootstrap_95_interval"],[y,y],color=BLUE,lw=2);a.scatter(r["mean_difference"],y,color=BLUE,s=24)
    a.axvline(0,color=".5",lw=.8);a.set(yticks=range(3),yticklabels=labels,xlabel="Full model minus baseline AP",title="(a) Original test: 29 galaxies");a.set_ylim(2.6,-.6)
    strata=["strong_spiral","weak_lowbar","weak_bar_proxy"]
    for y,k in enumerate(strata):
        r=fresh["arm_region_agreement"][k]["3"];ci=r["average_precision_group_bootstrap"]
        b.plot([ci["ci95_low"],ci["ci95_high"]],[y,y],color=ORANGE,lw=2);b.scatter(ci["mean"],y,color=ORANGE,s=24)
    b.text(.5,3,"AP undefined",ha="center",va="center",fontsize=8.5,color=".4")
    b.set(xlim=(0,1),ylim=(3.6,-.6),yticks=range(4),yticklabels=["Strong (57)","Weak / low bar (10)","Weak / bar proxy (12)","Unasked (16)"],xlabel="Galaxy-macro AP",title="(b) Fresh population")
    legend(fig,[Line2D([0],[0],color=BLUE,marker="o"),Line2D([0],[0],color=ORANGE,marker="o")],["13 groups; equal-group paired gain","Whole-group bootstrap; galaxy-macro mean"],ncol=1,y=.008)
    save(fig,"arm_region_validation","Arm-region agreement and baseline control. (a) On the original 29-galaxy test in 13 sky groups, paired differences against g brightness, a circular radial profile and the training-selected learned brightness/radius baseline average sky-group means equally; bars are the archived 95 percent whole-group bootstrap intervals. This estimand differs from the ordinary galaxy-macro AP difference. (b) In the fresh same-survey cohort, AP is averaged over galaxies, with whole-sky-group bootstrap uncertainty. Strong spirals give AP 0.689 on 57 evaluable objects (58 images from 59 fixed identities), weak/low-bar objects 0.453 on 10 and weak/bar-proxy objects 0.628 on 12. Sixteen unasked controls have no AP ground truth. Human vote regions are incomplete projected annotations, not exact physical-arm truth.")

def grouped_effects():
    s=readjson("redshift_statistics")
    fig,axes=standard(width=7.6,height=4.1,wspace=.78,bottom=.28);a,b=list(axes)
    rows=[s["cohorts"]["original_test"]["z"],s["cohorts"]["fresh_strong"]["z"]]
    for ax in [a,b]:ax.axvspan(-1,1,color="#e1edf3");ax.axvline(0,color=".45",lw=.8);ax.set(xlim=(-1.65,1.9),ylim=(1.65,-.6),xlabel="Paired MAE(z) gain, units of 0.001")
    for y,r in enumerate(rows):a.plot(1000*np.array(r["bootstrap95"]),[y,y],color=BLUE,lw=2);a.scatter(1000*r["equal_group_improvement"],y,color=BLUE,s=28)
    a.set(yticks=[0,1],yticklabels=["Original: 29 / 13 groups","Fresh: 59 / 30 groups"],title="(a) Frozen PGI versus + arms")
    for y,key in enumerate(["L5_minus_L5A","L3_minus_L3A"]):
        r=s["controlled_baselines"]["paired_contrasts"][key];b.plot(1000*np.array(r["bootstrap95"]),[y,y],color=ORANGE,lw=2);b.scatter(1000*r["equal_group_improvement"],y,color=ORANGE,s=28,marker="s")
    b.set(yticks=[0,1],yticklabels=["ugriz morphology + arms","grz morphology + arms"],title="(b) Exploratory controls")
    legend(fig,[Line2D([0],[0],color=".3",lw=2),Patch(facecolor="#e1edf3")],["95% whole-sky-group bootstrap","Retrospective margin: +/- 0.001"],ncol=1,y=.007)
    save(fig,"grouped_redshift_effects","Incremental arm information in redshift prediction, with positive values favoring arm summaries. Each galaxy's paired absolute-error difference is averaged within its sky group and groups receive equal weight. (a) Frozen PGI versus PGIA gives -0.000471, 95 percent interval [-0.001391, 0.000367], on the original 29 galaxies/13 groups; the fresh 59/30 result is 0.000440 [-0.000555, 0.001452]. Cohorts are not pooled. (b) New literature-motivated controls fit only the original 91 training galaxies at fixed estimator settings, adding angular radius, axis ratio, observed surface brightness and Sersic n before testing raw arm summaries. The ugriz and explicit-grz gains are 0.000584 [-0.000331, 0.001658] and 0.000142 [-0.000802, 0.001195]. These added comparisons are retrospective and exploratory, not exact reproductions of the published Morpho-z pipeline. Shading is an explicitly retrospective practical margin, not demonstrated equivalence. All intervals use 100000 whole-group bootstrap draws.")

def redshifts():
    f=readcsv("redshift_predictions");f=f[f.sample_group=="strong_spiral"].copy()
    fig,axes=standard(width=7.4,height=3.9,wspace=.42);a,b=list(axes)
    limit=max(f.spec_z.max(),f.PGI_predicted_catalog_redshift.max(),f.PGIA_predicted_catalog_redshift.max())*1.05
    a.plot([0,limit],[0,limit],color=".55",lw=.8,ls=":")
    for family,color,marker,style in [("PGI",BLUE,"o","-"),("PGIA",ORANGE,"^","--")]:
        pred=f[family+"_predicted_catalog_redshift"].to_numpy();a.scatter(f.spec_z,pred,color=color,marker=marker,s=19,alpha=.72,linewidths=0)
        err=np.sort(np.abs(pred-f.spec_z));b.step(err,np.arange(1,len(err)+1)/len(err),where="post",color=color,ls=style,lw=1.6)
    missing=f.PGI_missing_predictors>0
    if missing.any():a.scatter(f.loc[missing,"spec_z"],f.loc[missing,"PGI_predicted_catalog_redshift"],s=64,facecolors="none",edgecolors="black",linewidths=.8)
    a.set(xlim=(0,limit),ylim=(0,limit),xlabel="Inherited catalogue redshift",ylabel="Frozen redshift estimate",title="(a) All 59 fresh strong spirals")
    b.set(xlabel="Absolute redshift error",ylabel="Fraction of galaxies",ylim=(0,1.03),title="(b) Error distribution")
    legend(fig,[Line2D([0],[0],color=BLUE,marker="o"),Line2D([0],[0],color=ORANGE,marker="^",ls="--"),Line2D([0],[0],marker="o",color="black",markerfacecolor="none",ls="")],["PGI: basic images","PGIA: + raw arm summaries","Missing-image row retained"],ncol=2,y=.009)
    save(fig,"redshift_predictions","Frozen predictions and absolute-error empirical distributions for all 59 fixed fresh strong-spiral identities, including the single imputed missing-image row (open black outline). No input-support screening removes points from these primary displays. Galaxy-weighted MAE(z) is 0.01277 for PGI and 0.01253 for PGIA; these point summaries differ from the equal-sky-group paired estimand in the contrast figure. The target is inherited heliocentric NSA catalogue redshift, not an independently measured geometric distance. No universal morphology gain or distance-coverage claim follows.")

def geometry():
    rows=STATS["geometry_metadata"]
    fig,axes=standard(width=7.4,height=4.4,bottom=.28,wspace=.24);axes=list(axes)
    for ax,row in zip(axes,rows):
        prefix=row["prefix"];raw=CACHE[prefix+"_data"];valid=CACHE[prefix+"_valid"]
        lo,hi=np.quantile(raw[valid],[.02,.995]);data=np.arcsinh(np.maximum((raw-lo)/(hi-lo),0)*5)
        ax.imshow(data,origin="lower",extent=CACHE[prefix+"_extent"],cmap="gray",interpolation="nearest")
        for i in range(row["references"]):ax.plot(*CACHE[prefix+f"_ref{i}"].T,color=CYAN,lw=1.6)
        for i in range(row["predictions"]):ax.plot(*CACHE[prefix+f"_pred{i}"].T,color="#efa620",lw=1.05,ls="--")
        ax.set_title("Subject "+row["id"],fontsize=10)
        ax.set(xlabel="Image-frame offset (arcsec)",ylabel="Image-frame offset (arcsec)")
    legend(fig,[Line2D([0],[0],color=CYAN,lw=2),Line2D([0],[0],color="#efa620",ls="--")],["Independent human-fit centerlines","Fixed image-only templates"],ncol=1,y=.036)
    fig.text(.5,.008,"SDSS imaging; Galaxy Builder volunteers and Lingard et al. (2020, 2021)",ha="center",fontsize=7.6)
    record=next(r for r in PUBLIC_FIGURES["figures"] if r["name"]=="centerline_validation")
    save(fig,"centerline_validation",record["caption"],credits=record["credits"],sources=record["source_urls"],ids=[r["id"] for r in rows])


def finite_halo():
    # Exact supplied positive-equilibrium parameters of the archived validation.
    h=CachedHalo()
    r=np.geomspace(.1,230,400);rho=h.density(r,0);pressure=h.pressure(r,0)
    fig,axes=standard(ncols=3,width=9.0,height=3.9,wspace=.47,bottom=.27);a,b,c=list(axes)
    a.plot(r,rho/3e7,color=BLUE);a.plot(r,pressure/(3e7*1e4),color=ORANGE,ls="--")
    a.set(xscale="log",yscale="log",ylim=(1e-8,2),xlabel="Equatorial radius (kpc)",ylabel="Normalized density / pressure",title="(a) Free-surface closure")
    b.plot(r,np.where(rho>0,np.sqrt(h.local_dispersion_squared(r,0)),np.nan),color=PURPLE)
    b.axhline(100,color=".5",lw=.8,ls=":");b.set(xscale="log",xlabel="Equatorial radius (kpc)",ylabel="Local 1-D dispersion (km/s)",title="(b) Dispersion is not constant")
    c.plot(r,h.equatorial_circular_velocity(r),color=BLUE);c.set(xscale="log",xlabel="Equatorial radius (kpc)",ylabel="Circular speed (km/s)",title="(c) Total equilibrium force")
    legend(fig,[Line2D([0],[0],color=BLUE),Line2D([0],[0],color=ORANGE,ls="--"),Line2D([0],[0],color=".5",ls=":")],["Density / central density","Pressure / (central density x velocity scale squared)","Distribution velocity scale: 100 km/s"],ncol=1,y=.005)
    save(fig,"finite_halo_profiles","A supplied-parameter equilibrium illustrates the lowered-isothermal closure: central halo density 3e7 solar masses per cubic kpc, distribution velocity scale 100 km/s, central binding W0=5, and a Miyamoto-Nagai baryonic component with mass 1e10 solar masses, a=2 kpc and b=0.4 kpc. Density and pressure vanish at the same physical free surface; the numerical sphere extends through vacuum. Local one-dimensional dispersion is sqrt(P/rho), not the distribution scale. The rotation curve is a forward prediction from these stated physical parameters, not a fitted distance ruler. This established ergodic equilibrium can also be collisionless; static moments identify no viscosity or dark interaction rate. Curves are evaluated with the archived validation resolution, without refitting observed galaxies.",scope="supplement")

def btfr():
    f=readcsv("btfr_predictions")
    fig,axes=standard(width=7.4,height=3.9,wspace=.45);a,b=list(axes)
    extrap=f.outside_velocity_calibration_range.to_numpy(bool)
    for use,color,marker in [(~extrap,BLUE,"o"),(extrap,ORANGE,"^")]:
        a.errorbar(f.loc[use,"reference_mpc"],f.loc[use,"distance_mpc"],
            yerr=[f.loc[use,"distance_mpc"]-f.loc[use,"model_interval_90_lower_mpc"],f.loc[use,"model_interval_90_upper_mpc"]-f.loc[use,"distance_mpc"]],
            xerr=f.loc[use,"reference_error_mpc"],fmt=marker,color=color,markersize=3.4,elinewidth=.7,alpha=.7)
    lim=max(f.reference_mpc.max(),f.model_interval_90_upper_mpc.max())*1.05
    a.plot([0,lim],[0,lim],ls=":",color=".4",lw=.9);a.set(xlabel="Anchor reference distance (Mpc)",ylabel="LOO BTFR distance (Mpc)",title="(a) Same 30 local calibrators",xlim=(0,lim),ylim=(0,lim))
    err=(f.distance_mpc/f.reference_mpc-1).to_numpy();b.hist(err,bins=np.linspace(min(-.7,err.min()-.05),max(.9,err.max()+.05),10),color=BLUE,edgecolor="white");b.axvline(0,color=".4",ls=":",lw=.9)
    b.set(xlabel="Fractional distance residual",ylabel="Galaxies",title="(b) Conditional LOO residuals")
    b.yaxis.set_major_locator(MaxNLocator(4,integer=True))
    legend(fig,[Line2D([0],[0],color=BLUE,marker="o",ls=""),Line2D([0],[0],color=ORANGE,marker="^",ls=""),Line2D([0],[0],color=".4",ls="-")],["Within calibration speed range","Speed extrapolation (2 objects)","90% model interval; reference x-error"],ncol=2,y=.005)
    save(fig,"btfr_validation","Conditional baryonic Tully-Fisher leave-one-galaxy-out distance checks on the same 30 local calibrators. Each point is predicted from 29 others using measured HI flat speed and a calibrated baryonic-flux proxy; horizontal error bars show reference-distance uncertainty and vertical bars are 90 percent model intervals, not independently calibrated coverage guarantees. Median absolute fractional error is 17.9 percent, reference-point coverage 27/30, and median interval width is 82.2 percent of predicted distance. Two speed extrapolations are retained and marked. Shared anchor systematics, sample selection and common mass-to-light assumptions remain unmodeled. These are kinematic calibration checks, not image-only or viscosity-based distances.",scope="supplement",credits="SPARC/Lelli et al. published data; original derived uncertainty analysis",sources=["https://astroweb.case.edu/SPARC/","https://astroweb.case.edu/SPARC/BTFR_Lelli2019.mrt"])

def consensus():
    old=readjson("arm_original");fresh=readjson("arm_fresh")
    fig,axes=standard(width=7.4,height=3.9,wspace=.60);a,b=list(axes)
    for y,t in enumerate(["3","5"]):
        r=fresh["arm_region_agreement"]["strong_spiral"][t];ci=r["average_precision_group_bootstrap"]
        a.plot([ci["ci95_low"],ci["ci95_high"]],[y,y],color=BLUE,lw=2);a.scatter(ci["mean"],y,color=BLUE,s=28)
    a.set(xlim=(0,1),ylim=(1.6,-.6),yticks=[0,1],yticklabels=["3 votes: 57 objects","5 votes: 53 objects"],xlabel="Galaxy-macro AP",title="(a) Fresh strong spirals")
    ids=["strong_spiral","weak_lowbar","weak_bar_proxy","low_vote_unknown"]
    for y,k in enumerate(ids):
        r=fresh["arm_region_agreement"][k]["3"];b.scatter(r["candidate_area_fraction_median"],y,color=ORANGE,s=28)
    b.set(xlim=(0,.3),ylim=(3.6,-.6),yticks=range(4),yticklabels=["Strong","Weak / low bar","Weak / bar proxy","Unasked truth"],xlabel="Median candidate area fraction",title="(b) Descriptive candidate occupancy")
    legend(fig,[Line2D([0],[0],color=BLUE,marker="o"),Line2D([0],[0],color=ORANGE,marker="o",ls="")],["95% whole-group bootstrap","Fixed displayed threshold; not a false-positive rate"],ncol=1,y=.004)
    save(fig,"arm_consensus_sensitivity","Sensitivity to incomplete arm references and unknown controls. (a) On the fresh strong cohort, the fixed classifier's macro AP is 0.689 on 57 evaluable galaxies at the prespecified three-vote consensus and 0.487 on 53 at five votes. The evaluated reference prevalence and set of galaxies change, so this is a reference-definition sensitivity, not directly comparable independent accuracy experiments. (b) Candidate-area medians at the frozen displayed threshold are descriptive. Sixteen unasked controls have no spatial arm truth; nonzero occupancy in them cannot establish a measured real-survey false-positive rate. All fixed identities, reference failures and coverage gaps remain recorded.",scope="supplement")

def latex_escape(text):
    mapping={"&":r"\&","%":r"\%","_":r"\_","#":r"\#"}
    return "".join(mapping.get(c,c) for c in text)

def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    global CACHE,STATS,OUT,PUBLIC_FIGURES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"output",help="Figure output directory; defaults to ./output beside this script")
    args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((HERE/"provenance.json").read_text())
    if file_sha(HERE/"publication_plot_data.npz")!=provenance["npz_sha256"] or file_sha(HERE/"publication_plot_data.json")!=provenance["json_sha256"]:
        raise RuntimeError("Public plot data hashes do not match the export manifest")
    CACHE=np.load(HERE/"publication_plot_data.npz",allow_pickle=False)
    STATS=json.loads((HERE/"publication_plot_data.json").read_text())
    PUBLIC_FIGURES=json.loads((HERE/"figures.json").read_text())
    if any(CACHE[k].dtype.hasobject for k in CACHE.files):raise RuntimeError("Object arrays are prohibited")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.0,"axes.titlesize":10,
        "axes.labelsize":9.0,"xtick.labelsize":8.0,"ytick.labelsize":8.0,"axes.spines.top":False,
        "axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42,"figure.dpi":120})
    for function in [pipeline,arm_examples,arm_validation,grouped_effects,redshifts,geometry,finite_halo,btfr,consensus]:
        function();print("rendered",function.__name__,flush=True)
    (OUT/"figure_qa.json").write_text(json.dumps(QA,indent=2)+"\n")
    metadata={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"figures":RECORDS,
        "renderer":"render_figures.py","renderer_sha256":file_sha(Path(__file__)),
        "plot_data_sha256":provenance["npz_sha256"],"plot_statistics_sha256":provenance["json_sha256"],
        "source_scope":"Previously derived public plotting arrays; no refitting or raw-survey processing in this renderer"}
    (OUT/"figures.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"figures":len(RECORDS),"data_hashes_verified":True,
        "text_collisions":{r["figure"]:len(r["text_bbox_collisions"]) for r in QA}},indent=2))

if __name__=="__main__":main()
