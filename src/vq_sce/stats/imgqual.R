library(car)

source("load_funcs.R")
source("imgqual_funcs.R")

# dfs = load_quant_models("imgqual")
# models = dfs[[2]]
# dfs = dfs[[1]]

dfs = load_hyper_models("imgqual")
models = dfs[[2]]
dfs = dfs[[1]]

data = imgqual_data(dfs, models)

cme = data[[1]]
nge = data[[2]]

# run_anova("MSE", cme)
# run_anova("pSNR", cme)
# run_anova("SSIM", cme)
# run_anova("MSE", nge)
# run_anova("pSNR", nge)
# run_anova("SSIM", nge)

run_kruskal("MSE", cme)
run_kruskal("pSNR", cme)
run_kruskal("SSIM", cme)
run_kruskal("MSE", nge)
run_kruskal("pSNR", nge)
run_kruskal("SSIM", nge)

p2p_p2ppatch = cme$model == "p2p-full" | cme$model == "p2p-patch"
hp2p_hp2ppatch = cme$model == "hp2p-full" | cme$model == "hp2p-patch"
p2p_hp2p = cme$model == "p2p-full" | cme$model == "hp2p-full"
p2ppatch_hp2ppatch = cme$model == "p2p-patch" | cme$model == "hp2p-patch"

wilcox.test(MSE ~ model, data = cme[p2p_p2ppatch,])
wilcox.test(MSE ~ model, data = cme[hp2p_hp2ppatch,])
wilcox.test(MSE ~ model, data = cme[p2p_hp2p,])
wilcox.test(MSE ~ model, data = cme[p2ppatch_hp2ppatch,])

wilcox.test(pSNR ~ model, data = cme[p2p_p2ppatch,])
wilcox.test(pSNR ~ model, data = cme[hp2p_hp2ppatch,])
wilcox.test(pSNR ~ model, data = cme[p2p_hp2p,])
wilcox.test(pSNR ~ model, data = cme[p2ppatch_hp2ppatch,])

wilcox.test(MSE ~ model, data = nge[p2p_p2ppatch,])
wilcox.test(MSE ~ model, data = nge[hp2p_hp2ppatch,])
wilcox.test(MSE ~ model, data = nge[p2p_hp2p,])
wilcox.test(MSE ~ model, data = nge[p2ppatch_hp2ppatch,])

wilcox.test(pSNR ~ model, data = nge[p2p_p2ppatch,])
wilcox.test(pSNR ~ model, data = nge[hp2p_hp2ppatch,])
wilcox.test(pSNR ~ model, data = nge[p2p_hp2p,])
wilcox.test(pSNR ~ model, data = nge[p2ppatch_hp2ppatch,])
