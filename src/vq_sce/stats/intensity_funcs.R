intensity_data <- function(dfs, gt, models) {
  cmeAo = c()
  cmeCo = c()
  cmeMd = c()
  cmeTu = c()
  ngeAo = c()
  ngeCo = c()
  ngeMd = c()
  ngeTu = c()
  
  for (df in dfs) {
    cmeAo = c(cmeAo, df$Aorta)
    cmeCo = c(cmeCo, df$Cortex)
    cmeMd = c(cmeMd, df$Medulla)
    cmeTu = c(cmeTu, df$Tumour)
    ngeAo = c(ngeAo, df$Aorta.1)
    ngeCo = c(ngeCo, df$Cortex.1)
    ngeMd = c(ngeMd, df$Medulla.1)
    ngeTu = c(ngeTu, df$Tumour.1)
  }
  
  cmeAo = c(cmeAo, gt$Aorta.1)
  cmeCo = c(cmeCo, gt$Cortex.1)
  cmeMd = c(cmeMd, gt$Medulla.1)
  cmeTu = c(cmeTu, gt$Tumour.1)
  ngeAo = c(ngeAo, gt$Aorta.2)
  ngeCo = c(ngeCo, gt$Cortex.2)
  ngeMd = c(ngeMd, gt$Medulla.2)
  ngeTu = c(ngeTu, gt$Tumour.2)

  models = unlist(list(models, rep(factor("gt"), length(cmeAo) - length(models))))

  cme <- data.frame(
    "Aorta" = cmeAo,
    "Cortex" = cmeCo,
    "Medulla" = cmeMd,
    "Tumour" = cmeTu,
    "model" = models
  )
  
  nge <- data.frame(
    "Aorta" = ngeAo,
    "Cortex" = ngeCo,
    "Medulla" = ngeMd,
    "Tumour" = ngeTu,
    "model" = models
  )
  
  return(list(cme, nge))
}

run_anova <- function(ROI, df) {
  f = formula(paste(ROI, "model", sep = " ~ "))
  par(mfrow=c(1, 1))
  boxplot(f, data = df)
  res_aov <- aov(f, data = df)
  print(summary(res_aov))
  print(shapiro.test(res_aov$residuals))
  print(leveneTest(f, data = df))
  par(mfrow=c(2, 2))
  plot(res_aov)
}

run_kruskal <- function(ROI, df) {
  f = formula(paste(ROI, "model", sep = " ~ "))
  par(mfrow=c(1, 1))
  boxplot(f, data = df)
  res_kruskal <- kruskal.test(f, data = df)
  print(res_kruskal)
}