library("emmeans")
library("ggplot2")

source("load_funcs.R")
source("contrast_funcs.R")

dfs = load_hyper_models("contrast")
models = dfs[[3]]
gt = dfs[[2]]
dfs = dfs[[1]]

data = contrast_data(dfs, gt, models)

cme = data[[1]]
nge = data[[2]]

p2p_p2ppatch = cme$model == "p2p-full" | cme$model == "p2p-patch"
hp2p_hp2ppatch = cme$model == "hp2p-full" | cme$model == "hp2p-patch"
p2p_hp2p = cme$model == "p2p-full" | cme$model == "hp2p-full"
p2ppatch_hp2ppatch = cme$model == "p2p-patch" | cme$model == "hp2p-patch"

run_levene("Aorta", c("p2p-full", "p2p-patch", "hp2p-full", "hp2p-patch"), cme)
run_levene("Cortex", c("p2p-full", "p2p-patch", "hp2p-full", "hp2p-patch"), nge)
run_levene("Medulla", c("p2p-full", "p2p-patch", "hp2p-full", "hp2p-patch"), nge)
run_levene("Tumour", c("p2p-full", "p2p-patch", "hp2p-full", "hp2p-patch"), nge)
run_lm("Aorta", "p2p-full", cme)
run_lm("Aorta", "p2p-patch", cme)
run_lm("Aorta", "hp2p-full", cme)
run_lm("Aorta", "hp2p-patch", cme)
run_lm("Cortex", "p2p-full", cme)
run_lm("Cortex", "p2p-patch", cme)
run_lm("Cortex", "hp2p-full", cme)
run_lm("Cortex", "hp2p-patch", cme)
run_lm("Medulla", "p2p-full", cme)
run_lm("Medulla", "p2p-patch", cme)
run_lm("Medulla", "hp2p-full", cme)
run_lm("Medulla", "hp2p-patch", cme)
run_lm("Tumour", "p2p-full", cme)
run_lm("Tumour", "p2p-patch", cme)
run_lm("Tumour", "hp2p-full", cme)
run_lm("Tumour", "hp2p-patch", cme)

run_ancova("Aorta", nge, FALSE, TRUE)
run_ancova("Cortex", nge, FALSE, TRUE)
run_ancova("Medulla", nge, FALSE, TRUE)
run_ancova("Tumour", nge, FALSE, TRUE)
