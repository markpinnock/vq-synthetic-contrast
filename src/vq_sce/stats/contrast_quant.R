library("emmeans")
library("ggplot2")

source("load_funcs.R")
source("contrast_funcs.R")

dfs = load_quant_models("contrast")
models = dfs[[3]]
gt = dfs[[2]]
dfs = dfs[[1]]

data = contrast_data(dfs, gt, models)

cme = data[[1]]
nge = data[[2]]

run_levene("Aorta", c("unet-base", "unet-phase", "pix2pix", "cyclegan"), cme)
run_levene("Cortex", c("unet-base", "unet-phase", "pix2pix", "cyclegan"), cme)
run_levene("Medulla", c("unet-base", "unet-phase", "pix2pix", "cyclegan"), cme)
run_levene("Tumour", c("unet-base", "unet-phase", "pix2pix", "cyclegan"), cme)
run_lm("Aorta", "unet-base", cme)
run_lm("Aorta", "unet-phase", cme)
run_lm("Aorta", "pix2pix", cme)
run_lm("Aorta", "cyclegan", cme)
run_lm("Cortex", "unet-base", cme)
run_lm("Cortex", "unet-phase", cme)
run_lm("Cortex", "pix2pix", cme)
run_lm("Cortex", "cyclegan", cme)
run_lm("Medulla", "unet-base", cme)
run_lm("Medulla", "unet-phase", cme)
run_lm("Medulla", "pix2pix", cme)
run_lm("Medulla", "cyclegan", cme)
run_lm("Tumour", "unet-base", nge)
run_lm("Tumour", "unet-phase", cme)
run_lm("Tumour", "pix2pix", nge)
run_lm("Tumour", "cyclegan", nge)

run_ancova("Aorta", cme, interact = FALSE, pairwise = TRUE)
run_ancova("Cortex", cme, interact = FALSE, pairwise = TRUE)
run_ancova("Medulla", cme, interact = FALSE, pairwise = TRUE)
run_ancova("Tumour", cme, interact = FALSE, pairwise = TRUE)
