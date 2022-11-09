imgqual_data <- function(dfs, models) {
  cmeMSE = c()
  cmepSNR = c()
  cmeSSIM = c()
  ngeMSE = c()
  ngepSNR = c()
  ngeSSIM = c()
  
  for (df in dfs) {
    cmeMSE = c(cmeMSE, df$MSE)
    cmepSNR = c(cmepSNR, df$pSNR)
    cmeSSIM = c(cmeSSIM, df$SSIM)
    ngeMSE = c(ngeMSE, df$MSE.1)
    ngepSNR = c(ngepSNR, df$pSNR.1)
    ngeSSIM = c(ngeSSIM, df$SSIM.1)
  }
  
  cme <- data.frame("MSE" = cmeMSE, "pSNR" = cmepSNR, "SSIM" = cmeSSIM, "model" = models)
  nge <- data.frame("MSE" = ngeMSE, "pSNR" = ngepSNR, "SSIM" = ngeSSIM, "model" = models)
  
  return(list(cme, nge))
}

run_anova <- function(metric, df) {
  f = formula(paste(metric, "model", sep = " ~ "))
  par(mfrow=c(1, 1))
  boxplot(f, data = df)
  res_aov <- aov(f, data = df)
  print(summary(res_aov))
  print(shapiro.test(res_aov$residuals))
  print(leveneTest(f, data = df))
  par(mfrow=c(2, 2))
  plot(res_aov)
}

run_kruskal <- function(metric, df) {
  f = formula(paste(metric, "model", sep = " ~ "))
  par(mfrow=c(1, 1))
  boxplot(f, data = df)
  res_kruskal <- kruskal.test(f, data = df)
  print(res_kruskal)
}