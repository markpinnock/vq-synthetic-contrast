source("load_funcs.R")
source("segmentation_funcs.R")

dfs <- load_hyper_segs()
cme <- dfs[[1]]
nge <- dfs[[2]]

run_kruskal(nge)

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

# wilcox.test(Dice ~ model, data = cme[gt_unetbase,])
# wilcox.test(Dice ~ model, data = cme[gt_unetphase,])
# wilcox.test(Dice ~ model, data = cme[gt_pix2pix,])
# wilcox.test(Dice ~ model, data = cme[gt_cyclegan,])
# wilcox.test(Dice ~ model, data = cme[unetbase_unetphase,])
# wilcox.test(Dice ~ model, data = cme[unetbase_pix2pix,])
# wilcox.test(Dice ~ model, data = cme[unetbase_cyclegan,])
# wilcox.test(Dice ~ model, data = cme[unetphase_pix2pix,])
# wilcox.test(Dice ~ model, data = cme[unetphase_cyclegan,])
# wilcox.test(Dice ~ model, data = cme[pix2pix_cyclegan,])

wilcox.test(Dice ~ model, data = cme[gt_p2p,])
wilcox.test(Dice ~ model, data = cme[gt_p2ppatch,])
wilcox.test(Dice ~ model, data = cme[gt_hp2p,])
wilcox.test(Dice ~ model, data = cme[gt_hp2ppatch,])
wilcox.test(Dice ~ model, data = cme[p2p_p2ppatch,])
wilcox.test(Dice ~ model, data = cme[p2p_hp2p,])
wilcox.test(Dice ~ model, data = cme[p2p_hp2ppatch,])
wilcox.test(Dice ~ model, data = cme[p2ppatch_hp2p,])
wilcox.test(Dice ~ model, data = cme[p2ppatch_hp2ppatch,])
wilcox.test(Dice ~ model, data = cme[hp2p_hp2ppatch,])
