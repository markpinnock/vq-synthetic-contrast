plot_ba <- function(df, ROI, models) {
  par(mfrow = c(2, 2))
  subplots <- list(c(1, 1), c(1, 2), c(2, 1), c(2, 2))

  for (i in 1:4) {
    x <- df[df$model == models[i], paste(ROI, "Mean", sep = '')]
    y <- df[df$model == models[i], paste(ROI, "Diff", sep = '')]
    model_df <- data.frame('y' = y, 'x' = x)
    model_df <- na.omit(model_df)

    res_lm <- lm(y ~ x, data = model_df)
    b <-summary(lm(AortaMean ~ AortaDiff, data = cme))$coefficients[1, 1]
    m <-summary(lm(AortaMean ~ AortaDiff, data = cme))$coefficients[2, 1]
    LoA <- sd(summary(lm(AortaMean ~ AortaDiff, data = cme))$residuals) * 1.96

    par(mfg = subplots[i])
    plot(model_df$x, model_df$y)
    abline(a = b, b = m)
    abline(a = b + LoA, b = m)
    abline(a = b - LoA, b = m)
  }
}