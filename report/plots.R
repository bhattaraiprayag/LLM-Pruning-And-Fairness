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
colours = c('initial-model'='#332288', 
            'structured'='#117733', 
            'l1-unstructured'='#CC6677', 
            'global-unstructured'='#882255', 
            'global-unstructured-attention'='#88CCEE', 
            'imp'='#DDCC77', 
            'random-unstructured'='#AA4499', 
            '#44AA99',
            '#332288', 
            '#117733')
shapes = c('l1-unstructured'=21,
           'global-unstructured'=22,
           'global-unstructured-attention'=23,
           'structured'=24,
           'imp'=25,
           'random-unstructured'=4,
           'initial-model'=8)
names=c('initial-model'='Initial model', 
        'structured'='Structured', 
        'l1-unstructured'='Layerwise L1', 
        'global-unstructured'='Global L1', 
        'global-unstructured-attention'='Global L1 AH', 
        'imp'='Iterative', 
        'random-unstructured'='Random')

# Load in the data
results_data <- read_csv(paste0(base_folder,'LLM-Pruning-And-Fairness/results/results.csv')) %>%
  filter(!ID %in% c(10,11)) %>%
  mutate(pruning_method = coalesce(pruning_method, paste0('initial-model')))

# Group up based on the actual categories - still has both tasks

# First structured pruning
sp_group <- results_data %>%
  filter(pruning_method=='structured') %>%
  group_by(task, pruning_method, masking_threshold) %>%
  summarise(sparsity_level = mean(sparsity_level),
            SEAT_gender_max = max(SEAT_gender),
            SEAT_gender_min = min(SEAT_gender),
            SEAT_gender = mean(SEAT_gender),
            WEAT_gender_max = max(WEAT_gender),
            WEAT_gender_min = min(WEAT_gender),
            WEAT_gender = mean(WEAT_gender),
            StereoSet_LM_gender_max = max(StereoSet_LM_gender),
            StereoSet_LM_gender_min = min(StereoSet_LM_gender),
            StereoSet_LM_gender = mean(StereoSet_LM_gender),
            StereoSet_SS_gender_max = max(StereoSet_SS_gender),
            StereoSet_SS_gender_min = min(StereoSet_SS_gender),
            StereoSet_SS_gender = mean(StereoSet_SS_gender),
            BiasNLI_NN_max = max(BiasNLI_NN, na.rm = T),
            BiasNLI_NN_min = min(BiasNLI_NN, na.rm = T),
            BiasNLI_NN = mean(BiasNLI_NN, na.rm = T),
            BiasNLI_FN_max = max(BiasNLI_FN, na.rm = T),
            BiasNLI_FN_min = min(BiasNLI_FN, na.rm = T),
            BiasNLI_FN = mean(BiasNLI_FN, na.rm = T),
            BiasSTS_max = max(BiasSTS, na.rm = T),
            BiasSTS_min = min(BiasSTS, na.rm = T),
            BiasSTS = mean(BiasSTS, na.rm = T),
            Matched_Acc_max = max(`Matched Acc`, na.rm = T),
            Matched_Acc_min = min(`Matched Acc`, na.rm = T),
            Matched_Acc = mean(`Matched Acc`, na.rm = T),
            Mismatched_Acc_max = max(`Mismatched Acc`, na.rm = T),
            Mismatched_Acc_min = min(`Mismatched Acc`, na.rm = T),
            Mismatched_Acc = mean(`Mismatched Acc`, na.rm = T),
            Spearmanr_max = max(Spearmanr, na.rm = T),
            Spearmanr_min = min(Spearmanr, na.rm = T),
            Spearmanr = mean(Spearmanr, na.rm = T),
            Pearson_max = max(Pearson, na.rm = T),
            Pearson_min = min(Pearson, na.rm = T),
            Pearson = mean(Pearson, na.rm = T)) %>%
  ungroup() %>%
  select(-masking_threshold)

# Then everything else
results_group <- results_data %>%
  filter(pruning_method!='structured',
         pruning_method!='imp-ft',
         pruning_method!='random-unstructured') %>%
  group_by(task, pruning_method, sparsity_level) %>%
  summarise(SEAT_gender_max = max(SEAT_gender),
            SEAT_gender_min = min(SEAT_gender),
            SEAT_gender = mean(SEAT_gender),
            WEAT_gender_max = max(WEAT_gender),
            WEAT_gender_min = min(WEAT_gender),
            WEAT_gender = mean(WEAT_gender),
            StereoSet_LM_gender_max = max(StereoSet_LM_gender),
            StereoSet_LM_gender_min = min(StereoSet_LM_gender),
            StereoSet_LM_gender = mean(StereoSet_LM_gender),
            StereoSet_SS_gender_max = max(StereoSet_SS_gender),
            StereoSet_SS_gender_min = min(StereoSet_SS_gender),
            StereoSet_SS_gender = mean(StereoSet_SS_gender),
            BiasNLI_NN_max = max(BiasNLI_NN, na.rm = T),
            BiasNLI_NN_min = min(BiasNLI_NN, na.rm = T),
            BiasNLI_NN = mean(BiasNLI_NN, na.rm = T),
            BiasNLI_FN_max = max(BiasNLI_FN, na.rm = T),
            BiasNLI_FN_min = min(BiasNLI_FN, na.rm = T),
            BiasNLI_FN = mean(BiasNLI_FN, na.rm = T),
            BiasSTS_max = max(BiasSTS, na.rm = T),
            BiasSTS_min = min(BiasSTS, na.rm = T),
            BiasSTS = mean(BiasSTS, na.rm = T),
            Matched_Acc_max = max(`Matched Acc`, na.rm = T),
            Matched_Acc_min = min(`Matched Acc`, na.rm = T),
            Matched_Acc = mean(`Matched Acc`, na.rm = T),
            Mismatched_Acc_max = max(`Mismatched Acc`, na.rm = T),
            Mismatched_Acc_min = min(`Mismatched Acc`, na.rm = T),
            Mismatched_Acc = mean(`Mismatched Acc`, na.rm = T),
            Spearmanr_max = max(Spearmanr, na.rm = T),
            Spearmanr_min = min(Spearmanr, na.rm = T),
            Spearmanr = mean(Spearmanr, na.rm = T),
            Pearson_max = max(Pearson, na.rm = T),
            Pearson_min = min(Pearson, na.rm = T),
            Pearson = mean(Pearson, na.rm = T)) %>%
  ungroup() %>%
  bind_rows(sp_group)

### Bias against performance ####

# Function for plotting bias measures against model performance measures
acc_vs_bias_plot <- function(data, acc_measure, bias_measure, base_folder, optimum, task){
  output <-
    ggplot(data, aes(x=.data[[acc_measure]], y=.data[[bias_measure]], group=pruning_method, colour=pruning_method, shape=pruning_method, fill=pruning_method, alpha=1-sparsity_level)) +
    geom_point(size=4) +
    geom_hline(yintercept=optimum, linewidth=2, colour=colours[8], linetype='dashed') +
    scale_shape_manual(values=shapes,
                       labels=names)+
    scale_colour_manual(values=colours,
                        labels=names) +
    scale_fill_manual(values=colours,
                      labels=names) +
    scale_x_continuous(expand = c(0,0),
                       limits = c(0,1)) +
    scale_y_continuous(expand = c(0,0),
                       limits = c(0,1)) +
    theme_bw() + 
    guides(colour=guide_legend(title='Pruning:'),
           shape=guide_legend(title='Pruning:'),
           fill=guide_legend(title='Pruning:'),
           alpha=guide_legend(title='Density')) +
    coord_cartesian(clip='off') +
    labs(x = str_replace_all(acc_measure, '_', ' ') %>% str_to_sentence() %>% str_replace('acc', 'accuracy'),
         y = str_replace_all(bias_measure, '_', ' '))
  
  
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

# Function for including the base model at the start of all lines
zero_spars <- function(data){
  working <- data %>% filter(str_starts(pruning_method, 'initial'))
  
  output <- data %>% filter(!str_starts(pruning_method, 'initial'))
  
  pruning_methods <- output$pruning_method %>% unique()
  
  for(i in pruning_methods){
    output <- output %>%
      add_row(working %>% mutate(pruning_method=i))
  }
  
  return(output)
}

spars_vs_bias_plot <- function(data, bias_measure, base_folder, optimum, task, bad_models=F){
  
  if(task=='mnli' & bad_models==F){
    data = data %>%
      filter(Matched_Acc > 0.66)
  } else if(bad_models==F){
    data = data %>%
      filter(Spearmanr > 0.5)}
  
  output <-
    data %>%
    filter(pruning_method!='random-unstructured') %>%
    zero_spars() %>%
    ggplot(aes(x=sparsity_level, y=.data[[bias_measure]], group=pruning_method, colour=pruning_method)) +
    geom_line(linewidth=2) +
    geom_ribbon(aes(x = sparsity_level, ymax = .data[[paste0(bias_measure, '_max')]], ymin = .data[[paste0(bias_measure, '_min')]], group=pruning_method, fill=pruning_method), alpha = 0.2, colour = NA) +
    geom_point(size=2) +
    geom_hline(yintercept=optimum, linewidth=2, colour=colours[8], linetype='dashed') +
    scale_colour_manual(values=colours,
                        labels=names) +
    scale_fill_manual(values=colours,
                      labels=names) +
    scale_x_continuous(expand = c(0,0)) +
    scale_y_continuous(expand = c(0,0),
                       limits = c(0,1)) +
    theme_bw() + 
    guides(colour=guide_legend(title='Pruning:'),
           fill=guide_legend(title='Pruning:')) +
    coord_cartesian(clip='off') +
    labs(x = 'Sparsity level',
         y = str_replace_all(bias_measure, '_', ' '))
  
  if(bad_models==T){
    output_file = paste0(base_folder, 'LLM-Pruning-And-Fairness/report/figures/sb_', task, '_', bias_measure, '_all.png')
  } else {
    output_file = paste0(base_folder, 'LLM-Pruning-And-Fairness/report/figures/sb_', task, '_', bias_measure, '.png')
  }
  
  ggsave(filename = output_file,
         plot = output,
         width = 2100,
         height = 1400,
         units = "px")
}

spars_vs_bias_all <- function(data, base_folder, bad_models=F){
  mnli <- data %>% filter(task=='mnli')
  spars_vs_bias_plot(mnli, 'SEAT_gender', base_folder, 0, task='mnli', bad_models)
  spars_vs_bias_plot(mnli, 'WEAT_gender', base_folder, 0, task='mnli', bad_models)
  spars_vs_bias_plot(mnli, 'StereoSet_LM_gender', base_folder, 1, task='mnli', bad_models)
  spars_vs_bias_plot(mnli, 'StereoSet_SS_gender', base_folder, 0.5, task='mnli', bad_models)
  spars_vs_bias_plot(mnli, 'BiasNLI_NN', base_folder, 1, task='mnli', bad_models)
  spars_vs_bias_plot(mnli, 'BiasNLI_FN', base_folder, 1, task='mnli', bad_models)
  stsb <- data %>% filter(task=='stsb')
  spars_vs_bias_plot(stsb, 'SEAT_gender', base_folder, 0, task='stsb', bad_models)
  spars_vs_bias_plot(stsb, 'WEAT_gender', base_folder, 0, task='stsb', bad_models)
  spars_vs_bias_plot(stsb, 'StereoSet_LM_gender', base_folder, 1, task='stsb', bad_models)
  spars_vs_bias_plot(stsb, 'StereoSet_SS_gender', base_folder, 0.5, task='stsb', bad_models)
  spars_vs_bias_plot(stsb, 'BiasSTS', base_folder, 0, task='stsb', bad_models)
}

### Performance over increasing sparsity ####

# Loading performance data
performance_path <- paste0(base_folder, 'LLM-Pruning-And-Fairness/results/performance')

read_plus <- function(flnm, path) {
  read_csv(paste0(path, '/', flnm), show_col_types = FALSE) %>%
    mutate(task = str_split_i(flnm, '_', 1),
           model_no = str_split_i(flnm, '_', 2),
           pruning_method = str_split_i(flnm, '_', 3),
           seed = str_split_i(flnm, '_', 4))
}

perf_data  <-
  list.files(path = performance_path,
             pattern = "\\.csv$") %>%
  map_df(~read_plus(., performance_path)) %>%
  rename(sparsity=1) %>%
  mutate(pruning_method = str_remove(pruning_method, '\\.csv'),
         seed = as.integer(str_remove(seed, '\\.csv')))

# STSB

perf_stsb <- function(data, pruning_method_set, base_folder){
  working <- data %>%
    filter(task=='stsb',
           pruning_method==pruning_method_set) %>%
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
    geom_ribbon(aes(x = sparsity, ymax = maxi, ymin = mini, group=metric, fill=metric), alpha = 0.2, colour = NA) +
    geom_hline(yintercept=0.5, linewidth=2, colour=colours[8], linetype='dashed') +
    scale_fill_manual(values=colours) +
    scale_colour_manual(values=colours) +
    scale_x_continuous(expand = c(0,0),
                       limits = c(0,1)) +
    scale_y_continuous(expand = c(0,0),
                       limits = c(NA,1)) +
    theme_bw() + 
    guides(colour=guide_legend(title='Metric:'),
           fill=guide_legend(title='Metric:')) +
    coord_cartesian(clip='off') +
    labs(x = 'Sparsity level',
         y = 'Performance')
  
  ggsave(filename = paste0(base_folder, 'LLM-Pruning-And-Fairness/report/figures/pc_stsb_', pruning_method_set, '.png'),
         plot = output,
         width = 2100,
         height = 1400,
         units = "px")
}

perf_stsb_compare <- function(data, base_folder){
  working <- data %>%
    filter(task=='stsb') %>%
    group_by(pruning_method, sparsity) %>%
    summarise(spearmanr = mean(Spearmanr))
  
  output <-
    ggplot(working, aes(x=sparsity, y=spearmanr, group=pruning_method, colour=pruning_method)) +
    geom_line(linewidth=2) +
    scale_fill_manual(values=colours,
                      labels=names) +
    scale_colour_manual(values=colours,
                        labels=names) +
    geom_hline(yintercept=0.5, linewidth=2, colour=colours[8], linetype='dashed') +
    scale_x_continuous(expand = c(0,0),
                       limits = c(0,1)) +
    scale_y_continuous(expand = c(0,0),
                       limits = c(NA,1)) +
    theme_bw() + 
    guides(colour=guide_legend(title='Metric:')) +
    coord_cartesian(clip='off') +
    labs(x = 'Sparsity level',
         y = 'Spearmanr')
  
  ggsave(filename = paste0(base_folder, 'LLM-Pruning-And-Fairness/report/figures/pc_stsb_compare.png'),
         plot = output,
         width = 2100,
         height = 1400,
         units = "px")
}

# MNLI

perf_mnli <- function(data, pruning_method_set, base_folder){
  working <- data %>%
    filter(task=='mnli',
           pruning_method==pruning_method_set) %>%
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
    geom_ribbon(aes(x = sparsity, ymax = maxi, ymin = mini, group=accuracy, fill=accuracy), alpha = 0.2, colour = NA) +
    scale_fill_manual(values=colours[9:10]) +
    geom_hline(yintercept=0.66, linewidth=2, colour=colours[8], linetype='dashed') +
    scale_colour_manual(values=colours[9:10]) +
    scale_x_continuous(expand = c(0,0),
                       limits = c(0,1)) +
    scale_y_continuous(expand = c(0,0),
                       limits = c(0,1)) +
    theme_bw() + 
    guides(colour=guide_legend(title='Accuracy:'),
           fill=guide_legend(title='Accuracy:')) +
    coord_cartesian(clip='off') +
    labs(x = 'Sparsity level',
         y = 'Performance')
  
  ggsave(filename = paste0(base_folder, 'LLM-Pruning-And-Fairness/report/figures/pc_mnli_', pruning_method_set, '.png'),
         plot = output,
         width = 2100,
         height = 1400,
         units = "px")
}

perf_mnli_compare <- function(data, base_folder){
  working <- data %>%
    filter(task=='mnli') %>%
    group_by(pruning_method, sparsity) %>%
    summarise(matched = mean(`Matched Acc`))
  
  output <-
    ggplot(working, aes(x=sparsity, y=matched, group=pruning_method, colour=pruning_method)) +
    geom_line(linewidth=2) +
    scale_fill_manual(values=colours,
                      labels=names) +
    geom_hline(yintercept=0.66, linewidth=2, colour=colours[8], linetype='dashed') +
    scale_colour_manual(values=colours,
                        labels=names) +
    scale_x_continuous(expand = c(0,0),
                       limits = c(0,1)) +
    scale_y_continuous(expand = c(0,0),
                       limits = c(0,1)) +
    theme_bw() + 
    guides(colour=guide_legend(title='Accuracy:')) +
    coord_cartesian(clip='off') +
    labs(x = 'Sparsity level',
         y = 'Matched accuracy')
  
  ggsave(filename = paste0(base_folder, 'LLM-Pruning-And-Fairness/report/figures/pc_mnli_compare.png'),
         plot = output,
         width = 2100,
         height = 1400,
         units = "px")
}

perf_all <- function(data, base_folder){
  perf_stsb(data, 'l1-unstructured', base_folder)
  perf_mnli(data, 'l1-unstructured', base_folder)
  perf_stsb(data, 'random-unstructured', base_folder)
  perf_mnli(data, 'random-unstructured', base_folder)
  perf_stsb(data, 'global-unstructured', base_folder)
  perf_mnli(data, 'global-unstructured', base_folder)
  perf_stsb_compare(data, base_folder)
  perf_mnli_compare(data, base_folder)
}


### Actually running all the functions ####
acc_vs_bias_all(results_group, base_folder)
spars_vs_bias_all(results_group, base_folder)
spars_vs_bias_all(results_group, base_folder, bad_models=T)
perf_all(perf_data, base_folder)