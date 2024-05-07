#!/usr/bin/env Rscript

print.to.console = function(fmt, ...){
  fmt = paste0(fmt,"\n")
  cat(sprintf(fmt, ...))
}

print.to.console('Running setup_reticulate.R\n')
print.to.console("Working Directory\n'%s'\n", getwd())
print.to.console("Library Directory\n'%s'\n", .libPaths())

pkgs_r = c("argparser", "reticulate", "rmarkdown", "knitr", "tidyverse", "xtable")
install.packages(pkgs = pkgs_r, repos =  "https://cloud.r-project.org", ask = FALSE, checkBuilt = TRUE)