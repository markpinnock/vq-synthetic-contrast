contrast_data <- function(dfs, gt, models) {
  cmeAoMean = c()
  cmeAoDiff = c()
  cmeCoMean = c()
  cmeCoDiff = c()
  cmeMdMean = c()
  cmeMdDiff = c()
  cmeTuMean = c()
  cmeTuDiff = c()
  ngeAoMean = c()
  ngeAoDiff = c()
  ngeCoMean = c()
  ngeCoDiff = c()
  ngeMdMean = c()
  ngeMdDiff = c()
  ngeTuMean = c()
  ngeTuDiff = c()
  
  for (df in dfs) {
    cmeAoMean = c(cmeAoMean, (df$Aorta + gt$Aorta.1) / 2)
    cmeAoDiff = c(cmeAoDiff, df$Aorta - gt$Aorta.1)
    cmeCoMean = c(cmeCoMean, (df$Cortex + gt$Cortex.1) / 2)
    cmeCoDiff = c(cmeCoDiff, df$Cortex - gt$Cortex.1)
    cmeMdMean = c(cmeMdMean, (df$Medulla + gt$Medulla.1) / 2)
    cmeMdDiff = c(cmeMdDiff, df$Medulla - gt$Medulla.1)
    cmeTuMean = c(cmeTuMean, (df$Tumour + gt$Tumour.1) / 2)
    cmeTuDiff = c(cmeTuDiff, df$Tumour - gt$Tumour.1)
    ngeAoMean = c(ngeAoMean, (df$Aorta.1 + gt$Aorta.2) / 2)
    ngeAoDiff = c(ngeAoDiff, df$Aorta.1 - gt$Aorta.2)
    ngeCoMean = c(ngeCoMean, (df$Cortex.1 + gt$Cortex.2) / 2)
    ngeCoDiff = c(ngeCoDiff, df$Cortex.1 - gt$Cortex.2)
    ngeMdMean = c(ngeMdMean, (df$Medulla.1 + gt$Medulla.2) / 2)
    ngeMdDiff = c(ngeMdDiff, df$Medulla.1 - gt$Medulla.2)
    ngeTuMean = c(ngeTuMean, (df$Tumour.1 + gt$Tumour.2) / 2)
    ngeTuDiff = c(ngeTuDiff, df$Tumour.1 - gt$Tumour.2)
  }
  
  cme <- data.frame(
    "AortaDiff" = cmeAoDiff,
    "AortaMean" = cmeAoMean,
    "CortexDiff" = cmeCoDiff,
    "CortexMean" = cmeCoMean,
    "MedullaDiff" = cmeMdDiff,
    "MedullaMean" = cmeMdMean,
    "TumourDiff" = cmeTuDiff,
    "TumourMean" = cmeTuMean,
    "model" = models
    )

  nge <- data.frame(
    "AortaDiff" = ngeAoDiff,
    "AortaMean" = ngeAoMean,
    "CortexDiff" = ngeCoDiff,
    "CortexMean" = ngeCoMean,
    "MedullaDiff" = ngeMdDiff,
    "MedullaMean" = ngeMdMean,
    "TumourDiff" = ngeTuDiff,
    "TumourMean" = ngeTuMean,
    "model" = models
  )
  
  return(list(cme, nge))
}

run_lm <- function(ROI, model, df) {
  x <- df[df$model == model, paste(ROI, "Mean", sep = '')]
  xcentred <- x - mean(x, na.rm = TRUE)
  y <- df[df$model == model, paste(ROI, "Diff", sep = '')]
  model_df <- data.frame('y' = y, 'xcentred' = xcentred, 'x' = x)
  model_df <- na.omit(model_df)

  res_lm <- lm(y ~ xcentred, data = model_df)
  par(mfrow=c(1, 1))
  print(ggplot(model_df, aes(x, y)) + geom_point() + geom_smooth(method = 'lm'))
  print(summary(res_lm))
  meanBias <- mean(model_df$y, na.rm = TRUE)
  LoA <- sd(res_lm$residuals, na.rm = TRUE) * 1.96
  print(paste(ROI, " ", model, " Mean bias: ", meanBias, ", LoA: ", LoA, sep = ''))
  print(shapiro.test(res_lm$residuals))
  par(mfrow=c(2, 2))
  plot(res_lm)
}

run_ancova <- function(ROI, df, interact, pairwise) {
  x <- df[, paste(ROI, "Mean", sep = '')]
  x <- x - mean(x, na.rm = TRUE)
  y <- df[, paste(ROI, "Diff", sep = '')]
  new_df <- data.frame('y' = y, 'x' = x, 'g' = df$model)
  new_df <- na.omit(new_df)

  if (interact) {
    f <- formula("y ~ x * g")
  } else {
    f <- formula("y ~ x + g")
  }

  # par(mfrow=c(1, 1))
  # boxplot(f, data = df)
  res_lm <- lm(f, data = new_df)
  print(summary(res_lm))
  print(shapiro.test(res_lm$residuals))
  par(mfrow=c(2, 2))
  plot(res_lm)

  if (pairwise) {
    print(emmeans(res_lm, pairwise ~ g, adjust = "none"))
  }
}

run_levene <- function(ROI, models, df) {
  pooled_x <- df[, paste(ROI, "Mean", sep = '')]
  pooled_x <- pooled_x - mean(pooled_x, na.rm = TRUE)
  pooled_y <- df[, paste(ROI, "Diff", sep = '')]
  pooled_df <- data.frame('y' = pooled_y, 'x' = pooled_x, "model" = df[, "model"])
  pooled_df <- na.omit(pooled_df)
  grand_mean <- mean(pooled_df$x)
  pooled_lm <- lm(pooled_df$y ~ pooled_df$x, data = pooled_df)
  common_m <- summary(pooled_lm)$coefficients[2, 1]

  adj_ys <- c()
  
  for (model in models) {
    x <- df[df$model == model, paste(ROI, "Mean", sep = '')]
    x <- x - mean(x, na.rm = TRUE)
    y <- df[df$model == model, paste(ROI, "Diff", sep = '')]
    model_df <- data.frame('y' = y, 'x' = x)
    model_df <- na.omit(model_df)
    model_lm <- lm(model_df$y ~ model_df$x, data = model_df)

    adj_y <- model_df$y - common_m * (model_df$x - grand_mean)
    adj_ys <- c(adj_ys, adj_y)
    MSE <- mean(adj_y * adj_y)
    dof <- length(adj_y)

    print(paste(ROI, model, ": adjusted mean ", round(mean(adj_y)), ", MSE ", round(MSE), ", LoA ", round(sd(model_lm$residuals) * 1.96)))

    for (model_comp in models) {
      x_comp <- df[df$model == model_comp, paste(ROI, "Mean", sep = '')]
      x_comp <- x_comp - mean(x_comp, na.rm = TRUE)
      y_comp <- df[df$model == model_comp, paste(ROI, "Diff", sep = '')]
      comp_df <- data.frame('y' = y_comp, 'x' = x_comp)
      comp_df <- na.omit(comp_df)

      adj_y_comp <- comp_df$y - common_m * (comp_df$x - grand_mean)
      MSE_comp <- mean(adj_y_comp * adj_y_comp)
      dof_comp <- length(adj_y_comp)

      if (MSE > MSE_comp) {
        F_stat <- MSE / MSE_comp
      } else {
        F_stat <- MSE_comp / MSE
      }

      if (MSE > MSE_comp) {
        p_value <- pf(F_stat, dof, dof_comp, lower.tail = FALSE)
      } else {
        p_value <- pf(F_stat, dof_comp, dof, lower.tail = FALSE)
      }

      print(paste(model, model_comp, F_stat, p_value))
    }
  }

}