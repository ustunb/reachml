#!/usr/bin/env Rscript
# This script includes code to render an R Markdown template from the command line

# helper functions used in script
print.to.console = function(fmt, ...){
    fmt = paste0(fmt,"\n")
    cat(sprintf(fmt, ...))
}

Sys.setenv(RSTUDIO_PANDOC="/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools")
options(tinytex.tlmgr.path = tinytex:::find_tlmgr("/Users/avnikothari/.virtualenvs/reticulate-python-env/TinyTex"))

safe.dir = function(dir_name){
    if (is.na(dir_name) || is.null(dir_name)){
        return(NULL)
    }
    last_char = substr(dir_name, nchar(dir_name), nchar(dir_name))
    if (last_char != "/") {
        dir_name = paste0(dir_name, "/")
    }
    return(dir_name)
}

print.to.console("
Running: %s
Working Directory: %s
Library Directory: %s
", 'make_report.R', getwd(), .libPaths())

# parse input arguments
library(argparser)

p = arg_parser("render a report using rmarkdown::render")

p = add_argument(p,
                 "input",
                 type = "character",
                 help = "name of template file")

p = add_argument(p,
                 "report_data",
                 type = "character",
                 help = "name of report data file")

p = add_argument(p,
                 "--output_format",
                 type = "character",
                 help = "format",
                 default ="all")

p = add_argument(p,
                 "--output_file",
                 type = "character",
                 help = "name of output file")

p = add_argument(p,
                 "--output_dir",
                 type = "character",
                 help = "directory for rendered output file")

p = add_argument(p,
                 "--report_python_dir",
                 type = "character",
                 help = "directory for python")

p = add_argument(p,
                 "--build_dir",
                 type = "character",
                 help = "directory where to knit the document")

p = add_argument(p,
                 "--venv_dir",
                 type = "character",
                 help = "directory for the virtual environment for reticulate")


p = add_argument(p,
                 "--output_options",
                 help = "list of output options (overrides options in metadata)")

p = add_argument(p,
                 "--output_yaml",
                 type = "character",
                 help = "path to YAML file(s) specifying output format")

p = add_argument(p,
                 "--runtime",
                 default = "static",
                 type = "character",
                 help = "Runtime target for rendering. Options include: 'auto', which chooses the target based on YAML metadata; 'shiny' to render a Shiny file; 'static' to render a static file. The default is 'auto' or 'static' if there is no YAML metadata.")

p = add_argument(p,
                 "--params",
                 help = "list of named params  (override params in the YAML front-matter)")

# flags
p = add_argument(p,
                 "--clean",
                 flag = TRUE,
                 help = "flag to clean files after rendering")

p = add_argument(p,
                 "--quiet",
                 flag = TRUE,
                 help = "flag to suppress printing the pandoc command line")

p = add_argument(p,
                 "--run_pandoc",
                 flag = TRUE,
                 help = "flag to run pandoc to convert Markdown output")


argv <- parse_args(p);
# other render arguments are ignored because they are fixed/unused

# load required packages
required_packages = c('tools', 'rmarkdown', 'knitr')
for (pkg in required_packages) {
    suppressPackageStartupMessages(library(pkg, character.only = TRUE, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE));
}

# set directories
current_dir = getwd();
build_dir = safe.dir(argv$build_dir);
output_dir = safe.dir(argv$output_dir);


# report parameters
report_file = paste0(output_dir, argv$output_file);
report_data = argv$report_data
report_python_dir = safe.dir(argv$report_python_dir);
venv_dir = safe.dir(argv$venv_dir);
report_params = list(report_data = report_data,
                     report_python_dir = report_python_dir,
                     build_dir = build_dir,
                     venv_dir = venv_dir)

# copy template file to build_dir
template_dir = dirname(argv$input);
template_file = paste0(build_dir,  basename(argv$input));
template_ext = file_ext(template_file);
template_header = gsub(paste0('\\.', template_ext), '', basename(template_file));
invisible(file.copy(from = argv$input, to = template_file));

# store a list of files to delete after compilation
to_delete = c(template_file);

# copy dependent file to build_dir
dependent_files = list.files(path = template_dir, pattern = "\\.tex");
dependent_files = dependent_files[grepl(template_header, dependent_files)];
for (f in dependent_files){
    f_copy = paste0(build_dir,  basename(f));
    file.copy(from = f, to = f_copy);
    append(to_delete, f_copy);
}


tryCatch({

    setwd(report_params$build_dir);

    print.to.console("\nRendering Template")
    print.to.console("build directory: '%s", getwd())
    print.to.console("template_file: '%s", template_file)
    print.to.console("report_data: %s", report_params$report_data)
    print.to.console("report_python_dir: %s\n", report_params$report_python_dir)

    # see: https://rmarkdown.rstudio.com/docs/reference/render.html
    render(
        input = template_file,
        #
        # report parameters
        params = report_params,
        # output file
        output_file = report_file,
        # runtime = argv$runtime,
        #output_format = argv$output_format,
        #output_options = argv$output_options,
        #output_yaml = argv$output_yaml,
        quiet = argv$quiet,
        run_pandoc = argv$run_pandoc,
        #
        # build directory
        #        knit_root_dir = safe.dir(argv$build_dir),
        intermediates_dir = build_dir,
        clean = argv$clean
    )

}, error = function(e) {

    print.to.console("Compilation Error")
    print.to.console("%s", print(e))

    print.to.console("\nrlang::last_error():")
    print.to.console("%s", print(rlang::last_error()))

    print.to.console("\nrlang::last_trace()")
    rlang::last_trace()


}, finally = {

    # change working directory
    setwd(current_dir);

    # delete copies of template files in build directory
    for (f in to_delete){
        if (file.exists(f)){
            unlink(f, force = TRUE);
        }
    }

    # move .tex file to the build directory
    report_ext = paste0("\\.", file_ext(report_file));
    report_tex_file = gsub(report_ext, "\\.tex", report_file);
    if (file.exists(report_tex_file)){
        file.copy(from = report_tex_file, to = paste0(build_dir, basename(report_tex_file)));
        unlink(report_tex_file, force = TRUE);
    }
})
