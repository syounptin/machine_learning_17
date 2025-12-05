# Load packages
library(tidyverse)
library(tidymodels)

# Load customer data
load('customers.RData')

# Configure parallel execution of code
future::plan(future::multisession(
  workers = parallel::detectCores() - 1))

# Colorbind-safe palette:
pal <- c("#E69F00", "#56B4E9", "#009E73", 
         "#0072B2", "#D55E00", "#CC79A7", "#F0E442")

##########################################################################
# Split the data:
# * The analysis set will contain all observations from the start
#   of record keeping through the end of calendar year 2024
# * The assessment set will contain all observations from calendar
#   year 2025.


# about 80% of the data is in the analysis set:
prop_analysis <- summarise(customers, p = sum(year < 2025)/n())$p
prop_analysis

# set the seed for reproducibility
set.seed(101)

# `initial_time_split` splits the data based on row ordering. The `customers`
# data frame is already sorted by year. By splitting the first
# `prop_analysis` observations into the analysis set, the analysis set will
# contain all pre-2025 observations, and the assessment set will contain all
# 2025 observations.
split <- initial_time_split(customers, prop = prop_analysis)
analysis_set <- training(split)
# [Addition to the code] Create the assessment set (2025 data)
assessment_set <- testing(split)

# [Addition to the code] Sanity Check: Verify the years in each set
cat("Analysis Set Years: ", paste(range(analysis_set$year), collapse = " - "), "\n")
cat("Assessment Set Years: ", paste(range(assessment_set$year), collapse = " - "), "\n")

# [Addition to the code] Check dimensions
dim(analysis_set)
dim(assessment_set)

# [Addition to the code] Check Churn Rate in Analysis Set
analysis_set |> 
  count(churn) |> 
  mutate(prop = n / sum(n))

# [Addition to the code] Prepare the summary data first for easier labeling
churn_summary <- analysis_set |>
  count(churn) |>
  mutate(
    prop = n / sum(n),
    label = paste0(scales::percent(prop, accuracy = 1), "\n(", n, ")")
  )

# [Addition to the code] Visualizing the Class Imbalance

theme_set(theme_minimal(base_size = 14))

p1_pie <- churn_summary |>
  ggplot(aes(x = "", y = prop, fill = churn)) +
  geom_bar(stat = "identity", width = 1, color = "white") +
  coord_polar("y", start = 0) +
  scale_fill_manual(values = c("churned" = "#D55E00", "renewed" = "#009E73")) +

  geom_text(aes(label = label),
            position = position_stack(vjust = 0.5),
            color = "white",
            fontface = "bold",
            size = 5) +
  labs(
    title = "Current Churn Landscape (2024 & prior)",
    fill = "Status"
  ) +
  theme_void() +
  theme(legend.position = "right",
        plot.title = element_text(hjust = 0.5, face = "bold", size = 16))

print(p1_pie)

# [Addition to the code] Boxplot of Tenure by Churn status

p2 <- analysis_set |>
  ggplot(aes(x = churn, y = tenure, fill = churn)) +
  geom_boxplot(alpha = 0.6) +
  scale_fill_manual(values = c("churned" = "#D55E00", "renewed" = "#009E73")) +
  labs(
    title = "Customer Tenure vs. Churn Decisions",
    subtitle = "Newer customers (lower tenure) appear more likely to churn",
    x = NULL,
    y = "Years as Customer"
  ) +
  theme(legend.position = "none")

print(p2)

# [Addition to the code] Churn rates by Sector and Company Size

p3 <- analysis_set |>
  group_by(sector, size) |>
  summarise(churn_rate = mean(churn == "churned"), .groups = "drop") |>
  ggplot(aes(x = sector, y = churn_rate, fill = size)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = pal[1:3]) + # Uses Parker's palette
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Risk Profile by Sector and Size",
    subtitle = "Large companies in Sector C have the highest churn risk",
    x = "Industry Sector",
    y = "Churn Rate",
    fill = "Company Size"
  )

print(p3)

# Create CV folds. By choosing v = 5, the train/test split within each CV fold
# will be about 80% training and 20% test. That roughly matches the proportion
# that will eventually be used for the final assessment.
folds <- vfold_cv(training(split), v = 5, repeats = 4)


##########################################################################
# Fit a simple logistic regression as a baseline model. Nothing fancy.

# Define a workflow for logistic regression
wf_logistic <-
  workflow() |> 
  add_recipe(
    recipe(formula = churn ~ ., data = analysis_set) |> 
      update_role(id, new_role = 'metadata') |> 
      step_dummy(all_nominal_predictors()) |> 
      step_interact(~all_predictors():all_predictors()) |> 
      step_zv(all_predictors())) |> 
  add_model(logistic_reg(engine = 'glm'))

# Evaluate the baseline model using cross validation. I use the `tune_grid`
# function for this even though there is no tuning to be done. The
# `suppressWarnings` call at the end stops `tune_grid` from printing a warning
# about the lack of tunable parameters.
cv_results_logistic <- 
  wf_logistic |> 
  tune_grid(folds,
            metrics = metric_set(roc_auc, recall, precision,
                                 kap, bal_accuracy, f_meas),
            control = control_grid(save_pred = TRUE, 
                                   parallel_over = 'resamples')) |> 
  suppressWarnings()

# Baseline statistics calculated from CV predictions:
cv_results_logistic |> 
  collect_metrics()



##########################################################################
# TODO: Identify and train a better predicting model and 
#       evaluate its predictions using 2025 customer data








########################################################################## 
# TODO: Demonstrate how to use the model to select which
#       customers we will contact









