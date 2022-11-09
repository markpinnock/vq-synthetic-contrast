library(car)

source("load_funcs.R")
source("intensity_funcs.R")

# dfs = load_quant_models("contrast")
# models = dfs[[3]]
# gt = dfs[[2]]
# dfs = dfs[[1]]

dfs = load_hyper_models("contrast")
models = dfs[[3]]
gt = dfs[[2]]
dfs = dfs[[1]]

data = intensity_data(dfs, gt, models)

cme = data[[1]]
nge = data[[2]]

# run_anova("Aorta", cme)
# run_anova("Cortex", cme)
# run_anova("Medulla", cme)
# run_anova("Tumour", cme)
# run_anova("Aorta", nge)
# run_anova("Cortex", nge)
# run_anova("Medulla", nge)
# run_anova("Tumour", nge)

run_kruskal("Aorta", cme)
run_kruskal("Cortex", cme)
run_kruskal("Medulla", cme)
run_kruskal("Tumour", cme)
run_kruskal("Aorta", nge)
run_kruskal("Cortex", nge)
run_kruskal("Medulla", nge)
run_kruskal("Tumour", nge)

# gt_unetbase = cme$model == "gt" | cme$model == "unet-base"
# gt_unetphase = cme$model == "gt" | cme$model == "unet-phase"
# gt_pix2pix = cme$model == "gt" | cme$model == "pix2pix"
# gt_cyclegan = cme$model == "gt" | cme$model == "cyclegan"
# unetbase_unetphase = cme$model == "unet-base" | cme$model == "unet-phase"
# unetbase_pix2pix = cme$model == "unet-base" | cme$model == "pix2pix"
# unetbase_cyclegan = cme$model == "unet-base" | cme$model == "cyclegan"
# unetphase_pix2pix = cme$model == "unet-phase" | cme$model == "pix2pix"
# unetphase_cyclegan = cme$model == "unet-phase" | cme$model == "cyclegan"
# pix2pix_cyclegan = cme$model == "pix2pix" | cme$model == "cyclegan"

gt_p2p = cme$model == "gt" | cme$model == "p2p-full"
gt_p2ppatch = cme$model == "gt" | cme$model == "p2p-patch"
gt_hp2p = cme$model == "gt" | cme$model == "hp2p-full"
gt_hp2ppatch = cme$model == "gt" | cme$model == "hp2p-patch"
p2p_p2ppatch = cme$model == "p2p-full" | cme$model == "p2p-patch"
p2p_hp2p = cme$model == "p2p-full" | cme$model == "hp2p-full"
p2p_hp2ppatch = cme$model == "p2p-full" | cme$model == "hp2p-patch"
p2ppatch_hp2p = cme$model == "p2p-patch" | cme$model == "hp2p-full"
p2ppatch_hp2ppatch = cme$model == "p2p-patch" | cme$model == "hp2p-patch"
hp2p_hp2ppatch = cme$model == "hp2p-full" | cme$model == "hp2p-patch"

ROI = "Tumour"
f = formula(paste(ROI, "model", sep = " ~ "))

# wilcox.test(f, data = cme[gt_unetbase,])
# wilcox.test(f, data = cme[gt_unetphase,])
# wilcox.test(f, data = cme[gt_pix2pix,])
# wilcox.test(f, data = cme[gt_cyclegan,])
# wilcox.test(f, data = cme[unetbase_unetphase,])
# wilcox.test(f, data = cme[unetbase_pix2pix,])
# wilcox.test(f, data = cme[unetbase_cyclegan,])
# wilcox.test(f, data = cme[unetphase_pix2pix,])
# wilcox.test(f, data = cme[unetphase_cyclegan,])
# wilcox.test(f, data = cme[pix2pix_cyclegan,])

wilcox.test(f, data = nge[gt_p2p,])
wilcox.test(f, data = nge[gt_p2ppatch,])
wilcox.test(f, data = nge[gt_hp2p,])
wilcox.test(f, data = nge[gt_hp2ppatch,])
wilcox.test(f, data = nge[p2p_p2ppatch,])
wilcox.test(f, data = nge[p2p_hp2p,])
wilcox.test(f, data = nge[p2p_hp2ppatch,])
wilcox.test(f, data = nge[p2ppatch_hp2p,])
wilcox.test(f, data = nge[p2ppatch_hp2ppatch,])
wilcox.test(f, data = nge[hp2p_hp2ppatch,])

