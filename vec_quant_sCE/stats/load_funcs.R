load_quant_models <- function(expt_type) {
  unetBase <- read.csv(file.path('../', paste(expt_type, "unetbase.csv", sep = '_')), skip = 1)
  unetBase$model <- "unet-base"
  unetPhase <- read.csv(file.path('../', paste(expt_type, "unetphase.csv", sep = '_')), skip = 1)
  unetPhase$model <- "unet-phase"
  pix2pix <- read.csv(file.path('../', paste(expt_type, "p2ppatch.csv", sep = '_')), skip = 1)
  pix2pix$model <- "pix2pix"
  cycleGAN <- read.csv(file.path('../', paste(expt_type, "cyclegan.csv", sep = '_')), skip = 1)
  cycleGAN$model <- "cyclegan"

  models <- as.factor(c(unetBase$model, unetPhase$model, pix2pix$model, cycleGAN$model))
  models <- relevel(relevel(relevel(models, "pix2pix"), "unet-phase"), "unet-base")

  if (expt_type == "contrast") {
    gt <- read.csv(file.path('..', paste(expt_type, "gt.csv", sep = '_')), skip = 1)
    return(list(list(unetBase, unetPhase, pix2pix, cycleGAN), gt, models))
  } else {
    return(list(list(unetBase, unetPhase, pix2pix, cycleGAN), models))
  }
}

load_hyper_models <- function(expt_type) {
  p2pFull <- read.csv(file.path('..', paste(expt_type, "p2p.csv", sep = '_')), skip = 1)
  p2pFull$model <- "p2p-full"
  p2pPatch <- read.csv(file.path('..', paste(expt_type, "p2ppatch.csv", sep = '_')), skip = 1)
  p2pPatch$model <- "p2p-patch"
  hp2pFull <- read.csv(file.path('..', paste(expt_type, "hyperp2p.csv", sep = '_')), skip = 1)
  hp2pFull$model <- "hp2p-full"
  hp2pPatch <- read.csv(file.path('..', paste(expt_type, "hyperp2ppatch.csv", sep = '_')), skip = 1)
  hp2pPatch$model <- "hp2p-patch"

  models <- as.factor(c(p2pFull$model, p2pPatch$model, hp2pFull$model, hp2pPatch$model))
  models <- relevel(relevel(relevel(models, "hp2p-full"), "p2p-patch"), "p2p-full")

  if (expt_type == "contrast") {
    gt <- read.csv(file.path('..', paste(expt_type, "gt.csv", sep = '_')), skip = 1)
    return(list(list(p2pFull, p2pPatch, hp2pFull, hp2pPatch), gt, models))
  } else {
    return(list(list(p2pFull, p2pPatch, hp2pFull, hp2pPatch), models))
  }
}

load_quant_segs <- function() {
  ACVC <- read.csv("../seg_ACVC.csv")
  HQm <- read.csv("../seg_HQm.csv")

  unetBase <- rep("unet-base", length(HQm$UNetACVC_AP))
  unetPhase <- rep("unet-phase", length(HQm$UNetT_save1000_AP))
  pix2pix <- rep("pix2pix", length(HQm$X2_save170_patch_AP))
  cycleGAN <- rep("cyclegan", length(HQm$CycleGANT_save880_AP))
  models <- as.factor(c(rep("gt", length(ACVC$AC)), unetBase, unetPhase, pix2pix, cycleGAN))
  models <- relevel(relevel(relevel(relevel(models, "pix2pix"), "unet-phase"), "unet-base"), "gt")

  CME <- data.frame(
    "Dice" = c(ACVC$AC, HQm$UNetACVC_AP, HQm$UNetT_save1000_AP, HQm$X2_save170_patch_AP, HQm$CycleGANT_save880_AP),
    "model" = models)
  NGE <- data.frame(
    "Dice" = c(ACVC$VC, HQm$UNetACVC_VP, HQm$UNetT_save1000_VP, HQm$X2_save170_patch_VP, HQm$CycleGANT_save880_VP),
    "model" = models)

  return(list(CME, NGE))
}

load_hyper_segs <- function() {
  ACVC <- read.csv("../seg_ACVC.csv")
  HQm <- read.csv("../seg_HQm.csv")
  
  p2pFull <- rep("p2p-full", length(HQm$X2_save230_AP))
  p2pPatch <- rep("p2p-patch", length(HQm$X2_save170_patch_AP))
  hp2pFull <- rep("hp2p-full", length(HQm$H2_save280_AP))
  hp2pPatch <- rep("hp2p-patch", length(HQm$H2_save300_patch_AP))
  models <- as.factor(c(rep("gt", length(ACVC$AC)), p2pFull, p2pPatch, hp2pFull, hp2pPatch))
  models <- relevel(relevel(relevel(relevel(models, "hp2p-full"), "p2p-patch"), "p2p-full"), "gt")

  CME <- data.frame(
    "Dice" = c(ACVC$AC, HQm$X2_save230_AP, HQm$X2_save170_patch_AP, HQm$H2_save280_AP, HQm$H2_save300_patch_AP),
    "model" = models)
  NGE <- data.frame(
    "Dice" = c(ACVC$VC, HQm$X2_save230_VP, HQm$X2_save170_patch_VP, HQm$H2_save280_VP, HQm$H2_save300_patch_VP),
    "model" = models)
  
  return(list(CME, NGE))
}