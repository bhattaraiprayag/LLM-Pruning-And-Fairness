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
base_folder <- SET FOLDER LOCATION
colours = c('#332288', '#117733', '#CC6677', '#882255', '#88CCEE', '#DDCC77', '#AA4499', '#44AA99')

# Load in the data
results_data <- read_csv(paste0(base_folder,'LLM-Pruning-And-Fairness/results/results.csv')) %>%
  filter(!ID %in% c(10,11)) %>%
  mutate(pruning_method = coalesce(pruning_method, paste0('initial ', task, ' model')))

# Group up based on the actual categories - still has both tasks
results_group <- results_data %>%
  group_by(task, pruning_method, sparsity_level, temperature) %>%
  summarise(SEAT_gender = mean(SEAT_gender),
         WEAT_gender = mean(WEAT_gender),
         StereoSet_LM_gender = mean(StereoSet_LM_gender),
         StereoSet_SS_gender = mean(StereoSet_SS_gender),
         BiasNLI_NN = mean(BiasNLI_NN, na.rm = T),
         BiasNLI_FN = mean(BiasNLI_FN, na.rm = T),
         BiasSTS = mean(BiasSTS, na.rm = T),
         Matched_Acc = mean(`Matched Acc`),
         Mismatched_Acc = mean(`Mismatched Acc`)) %>%
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
}

### Bias over sparsity range ####

spars_vs_bias_plot <- function(data, bias_measure, base_folder, optimum, task){
  output <-
    ggplot(data, aes(x=sparsity_level, y=.data[[bias_measure]], group=pruning_method, colour=pruning_method)) +
    geom_line(size=2) +
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


### Actually running all the functions ####
acc_vs_bias_all(results_group, base_folder)
spars_vs_bias_all(results_group, base_folder)

