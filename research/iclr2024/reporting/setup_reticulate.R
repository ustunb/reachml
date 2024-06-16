#!/usr/bin/env Rscript

print.to.console = function(fmt, ...){
  fmt = paste0(fmt,"\n")
  cat(sprintf(fmt, ...))
}

print.to.console('Running setup_reticulate.R\n')
print.to.console("Working Directory\n'%s'\n", getwd())
print.to.console("Library Directory\n'%s'\n", .libPaths())


# parse input arguments
library(argparser)

p = arg_parser("setup reticulate environment")

p = add_argument(p,
                 "--envname",
                 type = "character",
                 help = "name of virtual environment",
                 default = "reticulate-python-env")

p = add_argument(p,
                 "--python_path",
                 type = "character",
                 help = "path of the python interpreter to use for the virtual environment")

p = add_argument(p,
                 "--packages",
                 type = "character",
                 help = "python packages to install in virtual environment",
                 nargs = Inf)

# parse args
argv <- parse_args(p);
print.to.console('envname: %s', argv$envname)
print.to.console('python_path: %s', argv$python_path)
print.to.console('packages: %s\n', argv$packages)

library(reticulate)

# setup options
venv_exists = argv$envname %in% virtualenv_list()
pkgs_listed = !is.na(argv$packages)

print.to.console('envname exists: %s', venv_exists)
print.to.console('packages listed: %s', pkgs_listed)


if (venv_exists){
  virtualenv_remove(envname = argv$envname)
}

stopifnot(!(argv$envname %in% virtualenv_list()))

virtualenv_install(
    envname = argv$envname,
    python =  argv$python_path,
    packages = argv$packages,
    system_site_packages = getOption("reticulate.virtualenv.system_site_packages", default = FALSE),
  )




