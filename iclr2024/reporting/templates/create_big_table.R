#### load packages ####
packages = c('reticulate', 'tidyverse', 'xtable', 'stringr', 'kableExtra',  'gridExtra', 'ggthemes', 'viridis', 'scales')
for (pkg in packages) {
    library(pkg, character.only = TRUE, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE)
}
options(dplyr.width = Inf)
options(stringsAsFactors = FALSE)

#### set directories ####
comp_name = "avnimac"
if (comp_name == "avnimac"){
    repo_dir = "/Users/avnikothari/Desktop/UCSD/Research/infeasible-recourse/"
    output_dir = "/Users/avnikothari/Desktop/UCSD/Research/infeasible-recourse/tables/"
} else {
    repo_dir = "repos/infeasible-recourse/"
    output_dir = "infeasible-recourse/tables/"
}


#### load data

all_files = dir(output_dir, pattern = "*.csv") %>% sort(decreasing = TRUE)
raw_file = paste0(output_dir, all_files[1])


all_data_names = c("fico", "give_me_credit", "german")
all_actionset_names = c("simple_1D", "complex_1D", "complex_nD")
all_method_names = c("confine", "ar", "dice")
all_model_types = c("logreg", "xgb", "rf")

# filter raw results to the datasets you care about
raw_df = read.csv(raw_file) %>%
    filter(data_name %in% all_data_names)

# create a data frame of recourse statistics
recourse_df = raw_df %>%
    filter(actionset_name %in% all_actionset_names,
           model_type %in% all_model_types)

# TO DO: Need to include
# # create a data frame with dataset information
# data_df = raw_df %>%
#     filter(stat_type %in% c("d", "n_train", "n_test"))
# 
# # create a data frame with model information
# models_df = raw_df %>%
#     filter(data_name %in% all_data_names,
#            str_detect(stat_type, "error|auc|loss"))

#### create a table of all statistics for the big table ####

table_stats_df = recourse_df %>%
    select(data_name, actionset_name, model_type, method_name, everything())

# format the value of different kinds of metric (svalue = 'string value')
table_stats_df = recourse_df %>%
    mutate(svalue = sprintf("%1.0f", stat_value),
           svalue_pct = sprintf("%1.1f\\%%", 100 * stat_value),
           svalue_dec = sprintf("%1.3f", stat_value)) %>%
    mutate(svalue = ifelse(str_detect("_cnt", stat_name), svalue, svalue_pct)) %>%
    select(-svalue_pct, -svalue_dec)

ACTIONSET_TITLES = c(
    '1d_simple' = 'Simple',
    '1d_complex' = 'Separable',
    'nd_complex' = 'Non-Separable'
)

METHOD_TITLES = c(
    "confine" = "\\confine{}",
    "ar" = "\\ar{}",
    "dice" = "\\dice{}"
)

MODEL_TYPE_TITLES = c(
    'logreg' = '\\LR{}',
    'rf' = '\\RF{}',
    'xgb' = '\\XGB{}'
)

METRIC_TITLES = c(
    'recourse_yes_pct' = 'Recourse',
    'recourse_yes_pos_pct' = 'Recourse +',
    'recourse_yes_neg_pct' = 'Recourse -',
    #
    'recourse_no_pct' = 'No Recourse',
    'recourse_no_pos_pct' = 'No Recourse +',
    'recourse_no_neg_pct' = 'No Recourse -',
    #
    'recourse_idk_pct' = 'Abstain',
    'recourse_idk_pos_pct' = 'Abstain +',
    'recourse_idk_neg_pct' = 'Abstain -',
    #
    'recourse_missing_pct' = 'No Response',
    'recourse_false_pct' = 'False Positive',
    'recourse_fail_pct' = 'False Negative',
)

# select metrics for the cells here
# change order of entries to change order in cell
cell_metric_names = c(
                      'recourse_yes_pct',
                      'recourse_no_pct',
                      'recourse_idk_pct',
                      'recourse_no_pos_pct',
                      'recourse_idk_pos_pct',
                      'recourse_false_pct',
                      'recourse_fail_pct',
                      )

# create the cells for every possible table #
cells_df = table_stats_df %>%
    filter(stat_name %in% cell_metric_names) %>%
    select(-value) %>%
    pivot_wider(
        names_from = stat_name,
        values_from = svalue
    ) %>%
    relocate(all_of(cell_metric_names), .after = last_col())

cells_df = cells_df %>%
    group_by(data_name, actionset_name, method, model_type, stat_name) %>%
    unite(cell_str, sep = "\\\\", all_of(cell_metric_names)) %>%
    mutate(cell_str = sprintf("\\cell{r}{%s}\n", cell_str)) %>%
    ungroup()


#### metrics big table ####
kable_tex_file = sprintf('%s_bigtable.tex', output_dir)

table_df = cells_df %>%
    arrange(data_name,
            match(actionset_name, all_actionset_names),
            match(model_type, all_model_types))

# create headers manually to avoid unique names issues
headers_df = table_df %>%
    mutate(actionset_name = str_to_lower(actionset_name), method_name = str_to_lower(method_name)) %>%
    group_by(actionset_name) %>%
    distinct(method_name)

# top level columns (actionability)
top_headers = headers_df %>%
    group_by(actionset_name) %>%
    count() %>%
    arrange(match(actionset_name, all_actionset_names)) %>%
    mutate(actionset_name = recode(actionset_name, !!!ACTIONSET_TITLES)) %>%
    pull(name = actionset_name) %>%
    prepend(c(" " = 3))

# bottom level columns (methods)
sub_headers = headers_df %>%
    mutate(method_name = str_to_lower(method_name)) %>%
    pull(method_name) %>%
    prepend(c("Dataset", "Model Type", "Metrics")) %>%
    recode(!!!METHOD_TITLES)

kable_df = table_df %>%
    mutate(data_name = paste0("\\textds{",data_name, "}"), metrics = "\\metricsguide{}", model_type = model_type) %>%
    pivot_wider(
        names_from = c(actionset_name, method_name),
        values_from = cell_str,
        names_sort = FALSE,
        names_glue = "{actionset_name}+{method_name}",
    )

overview_table = kable_df %>%
    kable(
        booktabs = TRUE,
        escape = FALSE,
        col.names = sub_headers,
        format = "latex"
    ) %>%
    kable_styling(
        latex_options = c("repeat_header", "scale_down", full_width = FALSE, font_size = 8, table.envir = "table*")) %>%
    add_header_above(top_headers, bold = FALSE, escape = FALSE) %>%
    column_spec(column = 1, latex_column_spec = "r") %>%
    column_spec(column = 2, latex_column_spec = "r") %>%
    column_spec(column = 3, latex_column_spec = "r") %>%
    row_spec(2:nrow(kable_df)-1, hline_after = TRUE, extra_latex_after = "\n")
