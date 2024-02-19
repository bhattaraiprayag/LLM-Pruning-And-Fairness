library(dplyr)
library(data.table)
library(purrr)
library(ggplot2)
library(scales)
library(lubridate)
library(stringr)
library(openxlsx)
library(tidyr)
library(readr)

# Basic settings
base_folder <- SET BASE FOLDER
colours = c('#332288', '#117733', '#CC6677', '#882255', '#88CCEE', '#DDCC77', '#AA4499', '#44AA99')

# Load in the data
results_data <- read_csv(paste0(base_folder,'LLM-Pruning-And-Fairness/results/results.csv')) %>%
  filter(!ID %in% c(10,11)) %>%
  mutate(pruning_method = coalesce(pruning_method, paste0('initial ', task, ' model')))

results_group <- results_data %>%
  group_by(task, pruning_method, sparsity_level) %>%
  summarise(SEAT_gender = mean(SEAT_gender),
            SEAT_gender_max = max(SEAT_gender),
            SEAT_gender_min = min(SEAT_gender),
            WEAT_gender = mean(WEAT_gender),
            WEAT_gender_max = max(WEAT_gender),
            WEAT_gender_min = min(WEAT_gender),
            StereoSet_LM_gender = mean(StereoSet_LM_gender),
            StereoSet_LM_gender_max = max(StereoSet_LM_gender),
            StereoSet_LM_gender_min = min(StereoSet_LM_gender),
            StereoSet_SS_gender = mean(StereoSet_SS_gender),
            StereoSet_SS_gender_max = max(StereoSet_SS_gender),
            StereoSet_SS_gender_min = min(StereoSet_SS_gender),
            BiasNLI_NN = mean(BiasNLI_NN, na.rm = T),
            BiasNLI_NN_max = max(BiasNLI_NN, na.rm = T),
            BiasNLI_NN_min = min(BiasNLI_NN, na.rm = T),
            BiasNLI_FN = mean(BiasNLI_FN, na.rm = T),
            BiasNLI_FN_max = max(BiasNLI_FN, na.rm = T),
            BiasNLI_FN_min = min(BiasNLI_FN, na.rm = T),
            BiasSTS = mean(BiasSTS, na.rm = T),
            BiasSTS_max = max(BiasSTS, na.rm = T),
            BiasSTS_min = min(BiasSTS, na.rm = T),
            Matched_Acc = mean(`Matched Acc`, na.rm = T),
            Matched_Acc_max = max(`Matched Acc`, na.rm = T),
            Matched_Acc_min = min(`Matched Acc`, na.rm = T),
            Mismatched_Acc = mean(`Mismatched Acc`, na.rm = T),
            Mismatched_Acc_max = max(`Mismatched Acc`, na.rm = T),
            Mismatched_Acc_min = min(`Mismatched Acc`, na.rm = T),
            Spearmanr = mean(Spearmanr, na.rm = T),
            Spearmanr_max = max(Spearmanr, na.rm = T),
            Spearmanr_min = min(Spearmanr, na.rm = T),
            Pearson = mean(Pearson, na.rm = T),
            Pearson_max = max(Pearson, na.rm = T),
            Pearson_min = min(Pearson, na.rm = T)) %>%
  ungroup()

### Bias against performance ####

# Function for plotting bias measures against model performance measures
acc_vs_bias_plot <- function(data, acc_measure, bias_measure, base_folder, optimum, task){
  output <-
    ggplot(data, aes(x=.data[[acc_measure]], y=.data[[bias_measure]], group=pruning_method, colour=pruning_method, alpha=1-sparsity_level)) +
    geom_point(size=4) +
    geom_hline(yintercept=optimum, linewidth=2, colour=colours[8], linetype='dashed') +
    scale_colour_manual(values=colours) +
    scale_x_continuous(expand = c(0,0),
                       limits = c(0,1)) +
    scale_y_continuous(expand = c(0,0),
                       limits = c(0,1)) +
    theme_bw() + 
    guides(colour=guide_legend(title='Pruning:'),
           alpha=guide_legend(title='Inverse sparsity')) +
    coord_cartesian(clip='off')
    
  
  ggsave(filename = paste0(base_folder, 'LLM-Pruning-And-Fairness/report/figures/ab_', task, '_', acc_measure, '_', bias_measure, '.png'),
         plot = output,
         width = 2100,
         height = 1400,
         units = "px")
}

# Function to plot all variants of bias measures against performance measures
acc_vs_bias_all <- function(data, base_folder){
  mnli <- data %>% filter(task=='mnli')
  acc_vs_bias_plot(mnli, 'Matched_Acc', 'SEAT_gender', base_folder, 0, task='mnli')
  acc_vs_bias_plot(mnli, 'Matched_Acc', 'WEAT_gender', base_folder, 0, task='mnli')
  acc_vs_bias_plot(mnli, 'Matched_Acc', 'StereoSet_LM_gender', base_folder, 1, task='mnli')
  acc_vs_bias_plot(mnli, 'Matched_Acc', 'StereoSet_SS_gender', base_folder, 0.5, task='mnli')
  acc_vs_bias_plot(mnli, 'Matched_Acc', 'BiasNLI_NN', base_folder, 1, task='mnli')
  acc_vs_bias_plot(mnli, 'Matched_Acc', 'BiasNLI_FN', base_folder, 1, task='mnli')
  stsb <- data %>% filter(task=='stsb')
  acc_vs_bias_plot(stsb, 'Spearmanr', 'SEAT_gender', base_folder, 0, task='stsb')
  acc_vs_bias_plot(stsb, 'Spearmanr', 'WEAT_gender', base_folder, 0, task='stsb')
  acc_vs_bias_plot(stsb, 'Spearmanr', 'StereoSet_LM_gender', base_folder, 1, task='stsb')
  acc_vs_bias_plot(stsb, 'Spearmanr', 'StereoSet_SS_gender', base_folder, 0.5, task='stsb')
  acc_vs_bias_plot(stsb, 'Spearmanr', 'BiasSTS', base_folder, 0, task='stsb')
}

### Bias over sparsity range ####

spars_vs_bias_plot <- function(data, bias_measure, base_folder, optimum, task){
  output <-
    ggplot(data, aes(x=sparsity_level, y=.data[[bias_measure]], group=pruning_method, colour=pruning_method)) +
    geom_line(linewidth=2) +
    geom_point(size=2) +
    geom_hline(yintercept=optimum, linewidth=2, colour=colours[8], linetype='dashed') +
    scale_colour_manual(values=colours) +
    scale_x_continuous(expand = c(0,0)) +
    scale_y_continuous(expand = c(0,0),
                       limits = c(0,1)) +
    theme_bw() + 
    guides(colour=guide_legend(title='Pruning:')) +
    coord_cartesian(clip='off')
  
  ggsave(filename = paste0(base_folder, 'LLM-Pruning-And-Fairness/report/figures/sb_', task, '_', bias_measure, '.png'),
         plot = output,
         width = 2100,
         height = 1400,
         units = "px")
}

spars_vs_bias_all <- function(data, base_folder){
  mnli <- data %>% filter(task=='mnli')
  spars_vs_bias_plot(mnli, 'SEAT_gender', base_folder, 0, task='mnli')
  spars_vs_bias_plot(mnli, 'WEAT_gender', base_folder, 0, task='mnli')
  spars_vs_bias_plot(mnli, 'StereoSet_LM_gender', base_folder, 1, task='mnli')
  spars_vs_bias_plot(mnli, 'StereoSet_SS_gender', base_folder, 0.5, task='mnli')
  spars_vs_bias_plot(mnli, 'BiasNLI_NN', base_folder, 1, task='mnli')
  spars_vs_bias_plot(mnli, 'BiasNLI_FN', base_folder, 1, task='mnli')
}

### Performance over increasing sparsity

# Loading performance data
performance_path <- paste0(base_folder, 'LLM-Pruning-And-Fairness/results/performance')

read_plus <- function(flnm, path) {
  read_csv(paste0(path, '/', flnm), show_col_types = FALSE) %>%
    mutate(task = str_split_i(flnm, '_', 1),
           model_no = str_split_i(flnm, '_', 2),
           pruning_method = str_split_i(flnm, '_', 3))
}

perf_data  <-
  list.files(path = performance_path,
             pattern = "\\.csv$") %>%
  map_df(~read_plus(., performance_path)) %>%
  rename(sparsity=1) %>%
  mutate(pruning_method = str_remove(pruning_method, '\\.csv'))

# STSB

perf_stsb <- function(data, pruning_method, base_folder){
  working <- data %>%
    filter(task=='stsb',
           pruning_method==pruning_method) %>%
    group_by(sparsity) %>%
    summarise(spearmanr = mean(Spearmanr),
              pearson = mean(Pearson),
              spearmanr_max = max(Spearmanr),
              pearson_max = max(Pearson),
              spearmanr_min = min(Spearmanr),
              pearson_min = min(Pearson)) %>%
    pivot_longer(cols = c(spearmanr, pearson),
                 names_to = 'metric',
                 values_to = 'performance') %>%
    mutate(maxi = if_else(metric=='spearmanr', spearmanr_max, pearson_max),
           mini = if_else(metric=='spearmanr', spearmanr_min, pearson_min))
  
  output <-
    ggplot(working, aes(x=sparsity, y=performance, group=metric, colour=metric)) +
    geom_line(linewidth=2) +
    geom_ribbon(aes(x = sparsity, ymax = maxi, ymin = mini, group=metric, fill=metric), alpha = 0.3, colour = NA) +
    scale_fill_manual(values=colours) +
    scale_colour_manual(values=colours) +
    scale_x_continuous(expand = c(0,0),
                       limits = c(0,1)) +
    scale_y_continuous(expand = c(0,0),
                       limits = c(NA,1)) +
    theme_bw() + 
    guides(colour=guide_legend(title='Metric:'),
           fill=guide_legend(title='Metric:')) +
    coord_cartesian(clip='off')
  
  ggsave(filename = paste0(base_folder, 'LLM-Pruning-And-Fairness/report/figures/pc_stsb_', pruning_method, '.png'),
         plot = output,
         width = 2100,
         height = 1400,
         units = "px")
}

# MNLI

perf_mnli <- function(data, pruning_method, base_folder){
  working <- data %>%
    filter(task=='mnli',
           pruning_method==pruning_method) %>%
    group_by(sparsity) %>%
    summarise(matched = mean(`Matched Acc`),
              mismatched = mean(`Mismatched Acc`),
              matched_max = max(`Matched Acc`),
              mismatched_max = max(`Mismatched Acc`),
              matched_min = min(`Matched Acc`),
              mismatched_min = min(`Mismatched Acc`)) %>%
    pivot_longer(cols = c(matched, mismatched),
                 names_to = 'accuracy',
                 values_to = 'performance') %>%
    mutate(maxi = if_else(accuracy=='matched', matched_max, mismatched_max),
           mini = if_else(accuracy=='matched', matched_min, mismatched_min))
  
  output <-
    ggplot(working, aes(x=sparsity, y=performance, group=accuracy, colour=accuracy)) +
    geom_line(linewidth=2) +
    geom_ribbon(aes(x = sparsity, ymax = maxi, ymin = mini, group=accuracy, fill=accuracy), alpha = 0.3, colour = NA) +
    scale_fill_manual(values=colours) +
    scale_colour_manual(values=colours) +
    scale_x_continuous(expand = c(0,0),
                       limits = c(0,1)) +
    scale_y_continuous(expand = c(0,0),
                       limits = c(0,1)) +
    theme_bw() + 
    guides(colour=guide_legend(title='Accuracy:'),
           fill=guide_legend(title='Accuracy:')) +
    coord_cartesian(clip='off')
  
  ggsave(filename = paste0(base_folder, 'LLM-Pruning-And-Fairness/report/figures/pc_mnli_', pruning_method, '.png'),
         plot = output,
         width = 2100,
         height = 1400,
         units = "px")
}

perf_all <- function(data, base_folder){
  perf_stsb(data, 'l1-unstructured', base_folder)
  perf_mnli(data, 'l1-unstructured', base_folder)
}


### Actually running all the functions ####
acc_vs_bias_all(results_group, base_folder)
spars_vs_bias_all(results_group, base_folder)
perf_all(perf_data, base_folder)

