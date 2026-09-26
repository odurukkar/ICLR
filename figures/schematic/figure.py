"""FigMirror style transfer from TabM (ICLR 2025); original TempoPB data.
Self-contained: edit the DATA SECTOR, then run this file. No repository imports.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.text import Text
import numpy as np

# === DATA SECTOR (edit here) ===
FIGURE = "schematic"
DATA = json.loads("{\"counts\":[\"12/32\",\"3/32\"],\"rules\":[\"Equal Shares\\npayments\",\"Equal Shares\\npayments\",\"Greedy fill\\nto budget\"],\"endowment_equation\":\"$\\\\beta_i\\\\propto(1+w^\\\\top f_i)_+,\\\\ \\\\sum_i\\\\beta_i=b$\",\"direct_equation\":\"$s(p)=v^\\\\top\\\\varphi(p,\\\\mathcal{H})$\",\"containment\":[\"$w=0\\\\Rightarrow\\\\mathrm{MES};\\\\quad w=\\\\lambda e_1\\\\Rightarrow\\\\mathrm{RES}(\\\\lambda)$\",\"$v=e_1\\\\Rightarrow\\\\mathrm{greedy\\\\!-\\\\!count};\\\\quad v=e_2\\\\Rightarrow\\\\mathrm{greedy\\\\!-\\\\!cost}$\"]}")
# === END DATA SECTOR ===

# Palette inspired by TabM Figures 1, 3 and 4; restrained line weights.
BLUE, ORANGE, GREEN = "#80b1d3", "#f9b160", "#7cca72"
PALE_BLUE, PALE_GREEN = "#cfe4f7", "#c4ecc1"
INK, GRID = "#222222", "#d6d6d6"  # Near-black axes and light dashed guides.
TIE = "#f4f1de"  # Exact-zero effects retain a separate color.
NEG, POS = "#3b6fb6", "#c45a3d"  # Negative and positive effect colors.
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.4, "ytick.labelsize": 8,
    "legend.fontsize": 7.3, "legend.frameon": False,
    "axes.edgecolor": INK, "axes.linewidth": 0.55,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.unicode_minus": False, "figure.dpi": 180,
    "savefig.dpi": 400, "savefig.bbox": None,
})
TARGETS = ("0.0", "0.85", "1.0")
KERNELS = ("static-floor", "payment-only", "payment+completion")
CITIES = ("Poland/Katowice", "Poland/Krakow")
SEEDS = (1, 2, 42)

def style_axis(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.55)
        ax.spines[s].set_color(INK)
    ax.tick_params(length=2.0, width=0.5, direction="out", pad=2)
    ax.grid(True, axis="both", color=GRID, linestyle=(0, (2, 2)), linewidth=0.45)
    ax.set_axisbelow(True)

def text_floor(fig) -> dict:
    """Check actual rendered text bounds, including legends and figure-level labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in fig.axes:
        if ax.axison:
            texts.extend(ax.get_xticklabels() + ax.get_yticklabels())
            texts.extend([ax.title, ax._left_title, ax._right_title,
                          ax.xaxis.label, ax.yaxis.label])
        texts.extend(ax.texts)
        if ax.get_legend():
            texts.extend(ax.get_legend().get_texts())
            texts.append(ax.get_legend().get_title())
    for leg in fig.legends:
        texts.extend(leg.get_texts()); texts.append(leg.get_title())
    seen = set(); boxes = []
    for t in texts:
        if id(t) in seen or not t.get_visible() or not t.get_text().strip():
            continue
        seen.add(id(t))
        b = Text.get_window_extent(t, renderer)
        boxes.append((t.get_text(), b))
    clipping = [s for s,b in boxes if b.x0 < 0 or b.y0 < 0 or
                b.x1 > fig.bbox.width or b.y1 > fig.bbox.height]
    overlap = []
    for i,(sa,a) in enumerate(boxes):
        for sb,b in boxes[i+1:]:
            if a.overlaps(b):
                overlap.append([sa,sb])
    result = {"scope":"rendered text bounds and pairwise text overlap only",
              "text_count":len(boxes), "clipped":clipping, "overlaps":overlap,
              "passed":not clipping and not overlap}
    if not result["passed"]:
        raise AssertionError(json.dumps(result, ensure_ascii=False))
    return result

def schematic():
    plt.rcParams.update({"font.family":"sans-serif",
                         "font.sans-serif":["Arial", "DejaVu Sans"]})
    fig = plt.figure(figsize=(5.5, 2.05))
    ax = fig.add_axes([0.012,0.025,0.976,0.955]); ax.set(xlim=(0,1),ylim=(0,1))
    ax.axis("off")
    centers = (0.217,0.58,0.862)
    for x,label in zip(centers, ("Learned object","Allocation rule","Observed changes")):
        ax.text(x,.935,label,ha="center",va="center",fontsize=6.8)
    rows = (.77,.57,.37)
    fills = (PALE_BLUE, PALE_GREEN, "#f5f5f5")  # Distinct mechanism lanes and a neutral baseline.
    blocks = ((.018,.398),(.47,.22),(.744,.238))
    for y,fill in zip(rows,fills):
        height = .174 if y == rows[0] else .154
        for col,(x,w) in enumerate(blocks):
            ax.add_patch(FancyBboxPatch((x,y-height/2),w,height,
                boxstyle="round,pad=0.004,rounding_size=0.018",
                fc=fill if col<2 else "white",ec=INK,lw=.55))
        for x0,x1 in ((.422,.463),(.696,.737)):
            ax.add_patch(FancyArrowPatch((x0,y),(x1,y),arrowstyle="-|>",
                         mutation_scale=6,lw=.55,color=INK))
    ax.text(centers[0],.823,"Endowment map",ha="center",va="center",fontsize=6.8)
    ax.text(centers[0],.742,DATA["endowment_equation"],ha="center",va="center",fontsize=7.7)
    ax.text(centers[0],.595,"Project priority",ha="center",va="center",fontsize=7.0)
    ax.text(centers[0],.54,"Learned purchase order",ha="center",va="center",fontsize=7.0)
    ax.text(centers[0],.416,"Direct score",ha="center",va="center",fontsize=7.0)
    ax.text(centers[0],.355,DATA["direct_equation"],ha="center",va="center",fontsize=8.0)
    for y,label in zip(rows,DATA["rules"]):
        ax.text(centers[1],y,label,ha="center",va="center",fontsize=7.0,linespacing=1.1)
    for y,count in zip(rows[:2], DATA["counts"]):
        ax.text(centers[2],y+.033,count+" series",ha="center",va="center",fontsize=7.4)
        ax.text(centers[2],y-.034,"Projected support",ha="center",va="center",fontsize=6.9)
    ax.text(centers[2],rows[2]+.053,"Not guaranteed",ha="center",va="center",fontsize=7.0)
    ax.text(centers[2],rows[2]-.031,"Not tested on\nprojected support",ha="center",
            va="center",fontsize=6.9,linespacing=1.05)
    ax.text(.018,.238,"Contains",ha="left",va="center",fontsize=6.8)
    ax.text(.165,.238,DATA["containment"][0],ha="left",va="center",fontsize=7.2)
    ax.text(.165,.175,DATA["containment"][1],ha="left",va="center",fontsize=7.2)
    ax.text(.5,.087,"Replay: fixed weights and features; histories and later scores may differ.",
            ha="center",va="center",fontsize=6.5,color="#555555")
    ax.text(.5,.033,"Fitted-class differences do not identify the causal effect of payment.",
            ha="center",va="center",fontsize=6.5,color="#555555")
    return fig, {"lanes":DATA["counts"],"equations":[DATA["endowment_equation"],
                         DATA["direct_equation"]]+DATA["containment"]}

def coverage():
    table={(str(float(r["floor"])),r["kernel"]):r for r in DATA["rows"]}
    assert len(table)==12
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.20),sharey=True)
    fig.subplots_adjust(left=.185,right=.985,bottom=.275,top=.86,wspace=.14)
    specs=(("exclusion_diff_vs_direct","exclusion_ci_lo","exclusion_ci_hi",
            "(a) Exclusion",[-.1,-.05,0],(-.14,.025)),
           ("diff_vs_direct","ci_lo","ci_hi",
            "(b) Worst-group CSD",[-.025,0,.025,.05,.075,.1],(-.043,.107)))
    plotted=[]
    for ax,(field,lo,hi,title,ticks,limits) in zip(axes,specs):
        style_axis(ax)
        for i,(target,color) in enumerate(zip(TARGETS,(BLUE,ORANGE,GREEN))):
            ys=np.arange(3)[::-1]+(-.22,0,.22)[i]
            vals=np.array([float(table[(target,k)][field]) for k in KERNELS])
            lows=np.array([float(table[(target,k)][lo]) for k in KERNELS])
            highs=np.array([float(table[(target,k)][hi]) for k in KERNELS])
            assert np.all(lows<=vals) and np.all(vals<=highs)
            ax.errorbar(vals,ys,xerr=[vals-lows,highs-vals],fmt="D",ms=3.6,
                        mfc=color,mec=INK,mew=.45,ecolor=color,elinewidth=1.0,
                        capsize=1.7,capthick=.6,zorder=4)
            plotted.extend([[field,target,k,float(v),float(l),float(h)]
                             for k,v,l,h in zip(KERNELS,vals,lows,highs)])
        ax.axvline(0,color=INK,lw=.7,zorder=2)
        ax.set(yticks=[2,1,0],yticklabels=["Static floor","Payment","Payment\n+ completion"],
               ylim=(-.45,2.48),xlim=limits,xticks=ticks,xlabel="Change versus direct fill")
        ax.set_title(title,loc="left",pad=6)
    handles=[Line2D([],[],marker="D",ls="",mfc=c,mec=INK,mew=.45,ms=3.8,label=l)
             for c,l in zip((BLUE,ORANGE,GREEN),("Unpenalized","0.85","1.00"))]
    fig.legend(handles=handles,ncol=3,loc="lower center",bbox_to_anchor=(.58,.005),
               title="Welfare target",title_fontsize=7.4,handletextpad=.3,
               columnspacing=.9,borderpad=.2,labelspacing=.25)
    return fig, {"contrasts":plotted}

def external():
    effects=DATA["effects"]; intervals=DATA["intervals"]
    assert DATA["decision"]=="falsified" and len(effects)==96 and len(intervals)==6
    assert len({(int(r["seed"]),r["series"]) for r in effects})==96
    primary=[r for r in effects if int(r["seed"])==42]
    assert len(primary)==32
    ordered=sorted(primary,key=lambda r:(float(r["delta_csd"]),r["series"]))
    order=[r["series"] for r in ordered]
    lookup={(r["series"],int(r["seed"])):float(r["delta_csd"]) for r in effects}
    matrix=np.array([[lookup[(s,seed)] for seed in SEEDS] for s in order])
    limit=float(np.max(np.abs(matrix)))
    assert limit==0.19971371186431283
    fig=plt.figure(figsize=(5.5,2.95))
    ax=fig.add_axes([.128,.24,.235,.65])
    heat=fig.add_axes([.444,.24,.108,.65])
    cbax=fig.add_axes([.563,.24,.018,.65])
    trade=fig.add_axes([.716,.24,.267,.65])
    style_axis(ax); style_axis(trade)
    colors={CITIES[0]:BLUE,CITIES[1]:ORANGE}
    shapes={CITIES[0]:"o",CITIES[1]:"s"}
    intervals_plotted=[]
    for ci,city in enumerate(CITIES):
        for seed,offset in zip(SEEDS,(-.16,0,.16)):
            r=next(r for r in intervals if r["city"]==city and int(r["seed"])==seed)
            v,lo,hi=[float(r[k]) for k in ("difference","ci_low","ci_high")]
            face={1:"white",2:TIE,42:colors[city]}[seed]
            ax.errorbar(v,1-ci+offset,xerr=[[v-lo],[hi-v]],fmt=shapes[city],
                        ms=4,mfc=face,mec=INK,mew=.5,ecolor=colors[city],
                        elinewidth=1.05,capsize=1.7,capthick=.55,zorder=4)
            intervals_plotted.append([city,seed,v,lo,hi])
    ax.axvline(0,color=INK,lw=.7)
    ax.axvline(-.005,color="#777777",lw=.7,ls=(0,(3,2)))
    ax.set(xlim=(-.045,.021),ylim=(-.38,1.4),yticks=[1,0],
           yticklabels=["Katowice","Krakow"],xticks=[-.04,-.02,0,.02])
    ax.set_xlabel(r"$\Delta$ CSD",labelpad=3)
    ax.set_title("(a) City intervals",loc="left",pad=7,fontsize=8.2)
    cmap=LinearSegmentedColormap.from_list("csd",(NEG,"white",POS))
    cmap=cmap.with_extremes(bad=TIE)
    im=heat.imshow(np.ma.masked_where(matrix==0,matrix),aspect="auto",
                    interpolation="nearest",cmap=cmap,
                    norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
    heat.set(xticks=[0,1,2],xticklabels=["1","2","42"],yticks=[],xlabel="Seed")
    heat.set_title("(b) Series effects",loc="left",pad=7,fontsize=8.2)
    heat.tick_params(length=0,pad=3)
    for s in heat.spines.values(): s.set_visible(False)
    for j,r in enumerate(ordered):
        heat.add_patch(Rectangle((-.78,j-.5),.16,1,fc=colors[r["city"]],
                                 ec="none",clip_on=False))
        for k,value in enumerate(matrix[j]):
            if value:
                heat.text(k,j,"+" if value>0 else "-",
                          ha="center",va="center",fontsize=5.5,
                          color="white" if abs(value)>.13 else INK)
    cb=fig.colorbar(im,cax=cbax,ticks=[-.1,0,.1])
    cb.ax.tick_params(labelsize=6.6,length=1.5,width=.4,pad=2)
    cb.outline.set_linewidth(.4)
    # Place the colorbar label below the scale to avoid adjacent labels.
    cb.ax.set_xlabel(r"$\Delta$ CSD",fontsize=6.8,labelpad=5)
    for city in CITIES:
        rows=[r for r in primary if r["city"]==city]
        trade.scatter([float(r["delta_csd"]) for r in rows],
                      [float(r["delta_exclusion"]) for r in rows],
                      s=16,c=colors[city],marker=shapes[city],edgecolors=INK,
                      linewidths=.5,label=city.rsplit("/",1)[-1],zorder=4)
    trade.axvline(0,color=INK,lw=.7);trade.axhline(0,color=INK,lw=.7)
    trade.set(xlim=(-.183,.115),ylim=(-.02,.09),xticks=[-.1,0,.1],
              yticks=[-.02,0,.02,.04,.06,.08],
              xlabel=r"$\Delta$ CSD",ylabel=r"$\Delta$ exclusion")
    trade.set_title("(c) Seed-42 tradeoff",loc="left",pad=7,fontsize=8.2)
    trade.yaxis.labelpad=2
    for name,offset in {"Czyżyny":(3,-10),"Podlesie":(4,-10),"Bronowice":(-5,6)}.items():
        r=next(r for r in primary if r["series"].endswith("/"+name))
        trade.annotate(name,(float(r["delta_csd"]),float(r["delta_exclusion"])),
                       xytext=offset,textcoords="offset points",
                       ha="left" if offset[0]>0 else "right",
                       va="top" if offset[1]<0 else "bottom",fontsize=6.7,
                       arrowprops={"arrowstyle":"-","lw":.45,"color":INK})
    seed_handles=[Line2D([],[],marker="o",ls="",mfc=f,mec=INK,ms=3.5,label=str(s))
                  for s,f in [(1,"white"),(2,TIE),(42,INK)]]
    fig.legend(handles=seed_handles,title="Seed",title_fontsize=7.3,ncol=3,
               loc="lower left",bbox_to_anchor=(.07,.015),handletextpad=.2,
               columnspacing=.65,borderpad=.15,labelspacing=.2)
    city_handles=[Line2D([],[],marker=shapes[c],ls="",mfc=colors[c],mec=INK,
                        ms=3.6,label=c.rsplit("/",1)[-1]) for c in CITIES]
    fig.legend(handles=city_handles,ncol=2,loc="lower right",bbox_to_anchor=(.99,.048),
               handletextpad=.2,columnspacing=.7,borderpad=.1)
    fig.text(.43,.12,"Beige cells: exact ties",fontsize=6.7,ha="left")
    fig.text(.5,.003,"Differences: learned endowment minus MES",ha="center",fontsize=7)
    return fig, {"intervals":intervals_plotted,"order":order,"matrix":matrix.tolist(),
                 "points":[[r["series"],r["city"],float(r["delta_csd"]),
                             float(r["delta_exclusion"])] for r in primary],
                 "zero_primary":sum(float(r["delta_csd"])==0 for r in primary),
                 "heatmap_limit":limit,"threshold":-.005}

def main() -> None:
    fig,plotted={"schematic":schematic,"coverage":coverage,"external":external}[FIGURE]()
    report=text_floor(fig)
    folder=Path(__file__).resolve().parent
    suffix=Path(__file__).stem.removeprefix("figure_")
    stem="figure" if suffix=="figure" else "img_"+suffix
    fig.savefig(folder/(stem+".png"),dpi=400)
    fig.savefig(folder/(stem+".pdf"),metadata={"CreationDate":None,"ModDate":None})
    report_path=folder/("floor_selfcheck_final.txt" if stem=="figure" else
                        "floor_selfcheck_"+suffix+".txt")
    report_path.write_text(json.dumps(report,indent=2)+"\n")
    (folder/"plotted_data.json").write_text(json.dumps(plotted,ensure_ascii=False,indent=2)+"\n")
    plt.close(fig)
    print(FIGURE+": floor passed; "+stem+".png/.pdf")
if __name__=="__main__":
    main()
