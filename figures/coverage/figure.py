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
FIGURE = "coverage"
DATA = json.loads("{\"rows\":[{\"floor\":\"0.0\",\"kernel\":\"direct\",\"worst_csd\":\"0.12284201495051265\",\"mean_csd\":\"0.0024390046663263563\",\"welfare\":\"586114.0\",\"exclusion\":\"0.1470062083207223\",\"spent\":\"182489181.0\",\"budget_utilization\":\"0.997099328924935\",\"welfare_ratio_vs_direct\":\"1.0\",\"diff_vs_direct\":\"0.0\",\"ci_lo\":\"0.0\",\"ci_hi\":\"0.0\",\"p_two_sided\":\"1.0\",\"wins_vs_direct\":\"0\",\"n\":\"18\",\"exclusion_diff_vs_direct\":\"0.0\",\"exclusion_ci_lo\":\"0.0\",\"exclusion_ci_hi\":\"0.0\",\"exclusion_p_two_sided\":\"1.0\",\"exclusion_wins_vs_direct\":\"0\"},{\"floor\":\"0.0\",\"kernel\":\"static-floor\",\"worst_csd\":\"0.11645627739212576\",\"mean_csd\":\"0.010666653570433223\",\"welfare\":\"719168.0\",\"exclusion\":\"0.093514885304777\",\"spent\":\"181880677.0\",\"budget_utilization\":\"0.9937745349468847\",\"welfare_ratio_vs_direct\":\"1.2270104450670005\",\"diff_vs_direct\":\"-0.006385737558386892\",\"ci_lo\":\"-0.03035136099009801\",\"ci_hi\":\"0.017787173995645524\",\"p_two_sided\":\"0.6122894287109375\",\"wins_vs_direct\":\"9\",\"n\":\"18\",\"exclusion_diff_vs_direct\":\"-0.05349132301594531\",\"exclusion_ci_lo\":\"-0.07732043316427882\",\"exclusion_ci_hi\":\"-0.03191044270200268\",\"exclusion_p_two_sided\":\"4.57763671875e-05\",\"exclusion_wins_vs_direct\":\"16\"},{\"floor\":\"0.0\",\"kernel\":\"payment+completion\",\"worst_csd\":\"0.1282197018070305\",\"mean_csd\":\"0.02128781074959427\",\"welfare\":\"1168671.0\",\"exclusion\":\"0.04271770601627977\",\"spent\":\"180650923.0\",\"budget_utilization\":\"0.987055304352372\",\"welfare_ratio_vs_direct\":\"1.9939312147466193\",\"diff_vs_direct\":\"0.00537768685651784\",\"ci_lo\":\"-0.02269681625606068\",\"ci_hi\":\"0.03409134091009465\",\"p_two_sided\":\"0.7207870483398438\",\"wins_vs_direct\":\"9\",\"n\":\"18\",\"exclusion_diff_vs_direct\":\"-0.10428850230444256\",\"exclusion_ci_lo\":\"-0.1313266259443973\",\"exclusion_ci_hi\":\"-0.07734990642939375\",\"exclusion_p_two_sided\":\"1.52587890625e-05\",\"exclusion_wins_vs_direct\":\"17\"},{\"floor\":\"0.0\",\"kernel\":\"payment-only\",\"worst_csd\":\"0.11402625759138887\",\"mean_csd\":\"0.014675463069687374\",\"welfare\":\"1021566.0\",\"exclusion\":\"0.05302672416921917\",\"spent\":\"152976410.0\",\"budget_utilization\":\"0.8358450343001195\",\"welfare_ratio_vs_direct\":\"1.742947617698946\",\"diff_vs_direct\":\"-0.008815757359123783\",\"ci_lo\":\"-0.03652738946347208\",\"ci_hi\":\"0.018219390500565723\",\"p_two_sided\":\"0.5453414916992188\",\"wins_vs_direct\":\"11\",\"n\":\"18\",\"exclusion_diff_vs_direct\":\"-0.09397948415150313\",\"exclusion_ci_lo\":\"-0.11934902912169573\",\"exclusion_ci_hi\":\"-0.06800426576810324\",\"exclusion_p_two_sided\":\"2.288818359375e-05\",\"exclusion_wins_vs_direct\":\"17\"},{\"floor\":\"0.85\",\"kernel\":\"direct\",\"worst_csd\":\"0.11525794725360286\",\"mean_csd\":\"0.010097473016641425\",\"welfare\":\"1198311.0\",\"exclusion\":\"0.044989635662301776\",\"spent\":\"179819440.0\",\"budget_utilization\":\"0.9825121794571351\",\"welfare_ratio_vs_direct\":\"1.0\",\"diff_vs_direct\":\"0.0\",\"ci_lo\":\"0.0\",\"ci_hi\":\"0.0\",\"p_two_sided\":\"1.0\",\"wins_vs_direct\":\"0\",\"n\":\"18\",\"exclusion_diff_vs_direct\":\"0.0\",\"exclusion_ci_lo\":\"0.0\",\"exclusion_ci_hi\":\"0.0\",\"exclusion_p_two_sided\":\"1.0\",\"exclusion_wins_vs_direct\":\"0\"},{\"floor\":\"0.85\",\"kernel\":\"static-floor\",\"worst_csd\":\"0.13395761107068216\",\"mean_csd\":\"0.01935479947232579\",\"welfare\":\"1218887.0\",\"exclusion\":\"0.04338334628658546\",\"spent\":\"179655832.0\",\"budget_utilization\":\"0.9816182446709039\",\"welfare_ratio_vs_direct\":\"1.0171708346163892\",\"diff_vs_direct\":\"0.018699663817079312\",\"ci_lo\":\"0.003929596499955022\",\"ci_hi\":\"0.036219191604341366\",\"p_two_sided\":\"0.02972412109375\",\"wins_vs_direct\":\"4\",\"n\":\"18\",\"exclusion_diff_vs_direct\":\"-0.001606289375716318\",\"exclusion_ci_lo\":\"-0.0032096076597615017\",\"exclusion_ci_hi\":\"-0.00014340586256445426\",\"exclusion_p_two_sided\":\"0.064697265625\",\"exclusion_wins_vs_direct\":\"9\"},{\"floor\":\"0.85\",\"kernel\":\"payment+completion\",\"worst_csd\":\"0.18355506239208846\",\"mean_csd\":\"0.03538932523785418\",\"welfare\":\"1273668.0\",\"exclusion\":\"0.04303838989463842\",\"spent\":\"180123237.0\",\"budget_utilization\":\"0.984172090379906\",\"welfare_ratio_vs_direct\":\"1.0628860120619772\",\"diff_vs_direct\":\"0.06829711513848563\",\"ci_lo\":\"0.04058192319392198\",\"ci_hi\":\"0.09712910309356602\",\"p_two_sided\":\"0.00020599365234375\",\"wins_vs_direct\":\"2\",\"n\":\"18\",\"exclusion_diff_vs_direct\":\"-0.0019512457676633546\",\"exclusion_ci_lo\":\"-0.005034808828854539\",\"exclusion_ci_hi\":\"0.0010681618013033442\",\"exclusion_p_two_sided\":\"0.240386962890625\",\"exclusion_wins_vs_direct\":\"12\"},{\"floor\":\"0.85\",\"kernel\":\"payment-only\",\"worst_csd\":\"0.16048544204708992\",\"mean_csd\":\"0.027550904849958948\",\"welfare\":\"1132666.0\",\"exclusion\":\"0.05342146627346392\",\"spent\":\"132234278.0\",\"budget_utilization\":\"0.7225124751624223\",\"welfare_ratio_vs_direct\":\"0.9452187286939701\",\"diff_vs_direct\":\"0.045227494793487054\",\"ci_lo\":\"0.017541110451134236\",\"ci_hi\":\"0.0743515005044121\",\"p_two_sided\":\"0.0068359375\",\"wins_vs_direct\":\"5\",\"n\":\"18\",\"exclusion_diff_vs_direct\":\"0.008431830611162145\",\"exclusion_ci_lo\":\"0.005008581763071386\",\"exclusion_ci_hi\":\"0.012012820633105669\",\"exclusion_p_two_sided\":\"0.00029754638671875\",\"exclusion_wins_vs_direct\":\"3\"},{\"floor\":\"1.0\",\"kernel\":\"direct\",\"worst_csd\":\"0.17280131051580602\",\"mean_csd\":\"0.031998238333563134\",\"welfare\":\"1303057.0\",\"exclusion\":\"0.04472175782546373\",\"spent\":\"178800252.0\",\"budget_utilization\":\"0.9769434566140623\",\"welfare_ratio_vs_direct\":\"1.0\",\"diff_vs_direct\":\"0.0\",\"ci_lo\":\"0.0\",\"ci_hi\":\"0.0\",\"p_two_sided\":\"1.0\",\"wins_vs_direct\":\"0\",\"n\":\"18\",\"exclusion_diff_vs_direct\":\"0.0\",\"exclusion_ci_lo\":\"0.0\",\"exclusion_ci_hi\":\"0.0\",\"exclusion_p_two_sided\":\"1.0\",\"exclusion_wins_vs_direct\":\"0\"},{\"floor\":\"1.0\",\"kernel\":\"static-floor\",\"worst_csd\":\"0.17987228004995118\",\"mean_csd\":\"0.0339500283462427\",\"welfare\":\"1302820.0\",\"exclusion\":\"0.04485632922360676\",\"spent\":\"178561602.0\",\"budget_utilization\":\"0.9756395012039718\",\"welfare_ratio_vs_direct\":\"0.9998181200054947\",\"diff_vs_direct\":\"0.007070969534145167\",\"ci_lo\":\"0.00013470514313479868\",\"ci_hi\":\"0.01742560118525445\",\"p_two_sided\":\"0.109375\",\"wins_vs_direct\":\"3\",\"n\":\"18\",\"exclusion_diff_vs_direct\":\"0.00013457139814302684\",\"exclusion_ci_lo\":\"-0.0002527879910575172\",\"exclusion_ci_hi\":\"0.0005689850358406147\",\"exclusion_p_two_sided\":\"0.578125\",\"exclusion_wins_vs_direct\":\"3\"},{\"floor\":\"1.0\",\"kernel\":\"payment+completion\",\"worst_csd\":\"0.19788295265808992\",\"mean_csd\":\"0.041227594458618004\",\"welfare\":\"1289024.0\",\"exclusion\":\"0.04305305070178723\",\"spent\":\"180111126.0\",\"budget_utilization\":\"0.9841059173064863\",\"welfare_ratio_vs_direct\":\"0.9892307090173339\",\"diff_vs_direct\":\"0.02508164214228389\",\"ci_lo\":\"0.006110634694265958\",\"ci_hi\":\"0.04565947661329655\",\"p_two_sided\":\"0.0218505859375\",\"wins_vs_direct\":\"4\",\"n\":\"18\",\"exclusion_diff_vs_direct\":\"-0.0016687071236765045\",\"exclusion_ci_lo\":\"-0.006362839452739079\",\"exclusion_ci_hi\":\"0.0026550204035142816\",\"exclusion_p_two_sided\":\"0.5133209228515625\",\"exclusion_wins_vs_direct\":\"10\"},{\"floor\":\"1.0\",\"kernel\":\"payment-only\",\"worst_csd\":\"0.17856230764452016\",\"mean_csd\":\"0.03558838433508174\",\"welfare\":\"1130853.0\",\"exclusion\":\"0.05698684268101634\",\"spent\":\"123689137.0\",\"budget_utilization\":\"0.6758228341109402\",\"welfare_ratio_vs_direct\":\"0.8678461494777281\",\"diff_vs_direct\":\"0.005760997128714162\",\"ci_lo\":\"-0.009964922472066042\",\"ci_hi\":\"0.02367879021839295\",\"p_two_sided\":\"0.5448074340820312\",\"wins_vs_direct\":\"7\",\"n\":\"18\",\"exclusion_diff_vs_direct\":\"0.01226508485555261\",\"exclusion_ci_lo\":\"0.008554367101605988\",\"exclusion_ci_hi\":\"0.0165218860162078\",\"exclusion_p_two_sided\":\"2.288818359375e-05\",\"exclusion_wins_vs_direct\":\"1\"}],\"sha256\":\"cb5b67916099f9747189c199cc2587b5d042192e4652a1bfe7e351d3d83b9ec1\"}")
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
    fig = plt.figure(figsize=(5.5, 2.24))
    ax = fig.add_axes([0.012,0.025,0.976,0.955]); ax.set(xlim=(0,1),ylim=(0,1))
    ax.axis("off")
    centers = (0.217,0.58,0.862)
    for x,label in zip(centers, ("Learned object","Allocation rule","Observed changes")):
        ax.text(x,.962,label,ha="center",va="center",fontsize=8.1)
    rows = (.77,.57,.37)
    fills = (PALE_BLUE, PALE_GREEN, "#f5f5f5")  # Distinct mechanism lanes and a neutral baseline.
    blocks = ((.018,.398),(.47,.22),(.744,.238))
    for y,fill in zip(rows,fills):
        for col,(x,w) in enumerate(blocks):
            ax.add_patch(FancyBboxPatch((x,y-.082),w,.164,
                boxstyle="round,pad=0.004,rounding_size=0.018",
                fc=fill if col<2 else "white",ec=INK,lw=.55))
        for x0,x1 in ((.422,.463),(.696,.737)):
            ax.add_patch(FancyArrowPatch((x0,y),(x1,y),arrowstyle="-|>",
                         mutation_scale=6,lw=.55,color=INK))
    ax.text(centers[0],.827,"Endowment map",ha="center",va="center",fontsize=7.7)
    ax.text(centers[0],.742,DATA["endowment_equation"],ha="center",va="center",fontsize=8.6)
    ax.text(centers[0],.595,"Project priority",ha="center",va="center",fontsize=7.7)
    ax.text(centers[0],.54,"Learned purchase order",ha="center",va="center",fontsize=7.2)
    ax.text(centers[0],.416,"Direct score",ha="center",va="center",fontsize=7.7)
    ax.text(centers[0],.355,DATA["direct_equation"],ha="center",va="center",fontsize=8.6)
    for y,label in zip(rows,DATA["rules"]):
        ax.text(centers[1],y,label,ha="center",va="center",fontsize=7.4,linespacing=1.2)
    for y,count in zip(rows[:2], DATA["counts"]):
        ax.text(centers[2],y+.033,count+" series",ha="center",va="center",fontsize=8.8)
        ax.text(centers[2],y-.034,"Projected support",ha="center",va="center",fontsize=6.9)
    ax.text(centers[2],rows[2]+.053,"Not guaranteed",ha="center",va="center",fontsize=7.2)
    ax.text(centers[2],rows[2]-.031,"Not tested on\nprojected support",ha="center",
            va="center",fontsize=6.9,linespacing=1.05)
    ax.text(.018,.238,"Contains",ha="left",va="center",fontsize=7.1)
    ax.text(.165,.238,DATA["containment"][0],ha="left",va="center",fontsize=8.0)
    ax.text(.165,.175,DATA["containment"][1],ha="left",va="center",fontsize=8.0)
    ax.plot([.018,.982],[.126,.126],color="#777777",lw=.45)
    ax.text(.5,.087,"Replay: fixed weights and features; histories and later scores may differ.",
            ha="center",va="center",fontsize=7.0)
    ax.text(.5,.033,"Fitted-class differences do not identify the causal effect of payment.",
            ha="center",va="center",fontsize=7.0)
    return fig, {"lanes":DATA["counts"],"equations":[DATA["endowment_equation"],
                         DATA["direct_equation"]]+DATA["containment"]}

def coverage():
    table={(str(float(r["floor"])),r["kernel"]):r for r in DATA["rows"]}
    assert len(table)==12
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.20),sharey=True)
    fig.subplots_adjust(left=.185,right=.985,bottom=.17,top=.91,wspace=.14)
    specs=(("exclusion_diff_vs_direct","exclusion_ci_lo","exclusion_ci_hi",
            "(a) Exclusion",[-.15,-.1,-.05,0],(-.165,.025)),
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
    axes[0].legend(handles=handles,ncol=1,loc="upper left",
                   title="Welfare target",title_fontsize=7.0,fontsize=7.0,
                   frameon=True,fancybox=False,edgecolor=GRID,facecolor="white",
                   handletextpad=.25,handlelength=1.1,borderpad=.25,labelspacing=.2)
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
