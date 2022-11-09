run_kruskal <- function(df) {
  par(mfrow=c(1, 1))
  boxplot(Dice ~ model, data = df)
  res_kruskal <- kruskal.test(Dice ~ model, data = df)
  print(res_kruskal)
}