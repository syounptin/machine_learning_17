# Load packages
library(tidyverse)
library(tidymodels)
library(themis)
library(skimr)
library(ggridges)
library(vip)

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

# We define the metrics here so they can be used in tuning below.
# mn_log_loss is added for probability calibration.
my_metrics <- metric_set(roc_auc, pr_auc, mn_log_loss, accuracy)


##########################################################################
# Modeling
##########################################################################
# MODEL 1: LOGISTIC REGRESSION (BASELINE) ------------------------------

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
print("Logistic Regression Results:")
cv_results_logistic |> 
  collect_metrics()

# --- MODEL 2: RANDOM FOREST ---

# Specification: Tuning mtry (vars per split) and min_n (node size)
rf_spec <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) |> 
  set_engine("ranger", importance = "impurity") |> 
  set_mode("classification")

# Recipe: Trees generally don't need interaction terms or one-hot encoding,
# but tidymodels handles factors automatically. We just remove ID.
tree_recipe <- 
  recipe(churn ~ ., data = analysis_set) |> 
  update_role(id, new_role = 'metadata') |> 
  step_dummy(all_nominal_predictors()) |> # Dummy encoding is safe for Ranger
  step_zv(all_predictors())

wf_rf <- 
  workflow() |> 
  add_recipe(tree_recipe) |> 
  add_model(rf_spec)

# Tuning
# We let tidymodels pick 10 random combinations to try
print("Tuning Random Forest...")
tune_rf <- 
  wf_rf |> 
  tune_grid(folds,
            grid = 10, 
            metrics = my_metrics,
            control = control_grid(save_pred = TRUE, parallel_over = 'resamples'))

# Show best RF results
tune_rf |> show_best(metric = "roc_auc")



# ==============================================================================
# MODEL 2: XGBOOST  ----------------------------
# ==============================================================================

# Specification: Tuning tree depth, learn rate, and loss reduction
xgb_spec <- 
  boost_tree(
    trees = 1000, 
    tree_depth = tune(), 
    min_n = tune(), 
    loss_reduction = tune(),                     
    sample_size = tune(), 
    mtry = tune(),         
    learn_rate = tune()                          
  ) |> 
  set_engine("xgboost") |> 
  set_mode("classification")

# Recipe: XGBoost requires all numeric input (one-hot encoding)
xgb_recipe <- 
  recipe(churn ~ ., data = analysis_set) |> 
  update_role(id, new_role = 'metadata') |> 
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |> 
  step_zv(all_predictors())

wf_xgb <- 
  workflow() |> 
  add_recipe(xgb_recipe) |> 
  add_model(xgb_spec)

# Tuning: Using a Latin Hypercube grid for better coverage of many parameters
xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), analysis_set),
  learn_rate(),
  size = 15 # Try 15 combinations
)

print("Tuning XGBoost...")
tune_xgb <- 
  wf_xgb |> 
  tune_grid(folds,
            grid = xgb_grid,
            metrics = my_metrics,
            control = control_grid(save_pred = TRUE, parallel_over = 'resamples'))

# Show best XGB results
tune_xgb |> show_best(metric = "roc_auc")

# ==============================================================================
# 7. MODEL SELECTION & FINAL FIT ------------------------------------------
# ==============================================================================

# Compare all three models
results_log <- collect_metrics(cv_results_logistic) |> mutate(model = "Logistic")
results_rf  <- collect_metrics(tune_rf) |> mutate(model = "Random Forest")
results_xgb <- collect_metrics(tune_xgb) |> mutate(model = "XGBoost")

# Combine and look at ROC AUC
bind_rows(results_log, results_rf, results_xgb) |> 
  filter(.metric == "roc_auc") |> 
  group_by(model) |> 
  slice_max(mean, n = 1) |> 
  arrange(desc(mean))

# Let's assume XGBoost won (common in this data). We select the best parameters.
best_xgb <- select_best(tune_xgb, metric = "roc_auc")

# Finalize workflow with best parameters
final_wf <- wf_xgb |> finalize_workflow(best_xgb)

# Fit on the FULL analysis set (years < 2025)
final_fit <- fit(final_wf, data = analysis_set)


# ==============================================================================
# 8. DEMONSTRATION (THE BUSINESS CASE) ------------------------------------
# ==============================================================================
# "Using the model trained on the entire analysis set, generate soft predictions 
#  for all observations in the assessment set (i.e., customer data from 2025)."

# 1. Predict Probabilities for 2025
pred_2025 <- augment(final_fit, new_data = assessment_set)

# 2. Calculate Expected Loss and Net Value
# Formula: E[Lost Revenue] = Spend_2025 * Prob(Churn)
# Decision: Target if E[Lost Revenue] > Retention Cost (€20,000)

retention_cost <- 20000

business_impact <- pred_2025 |> 
  select(id, spend, .pred_churned) |> 
  mutate(
    expected_loss = spend * .pred_churned,
    should_target = expected_loss > retention_cost,
    net_value_of_retention = expected_loss - retention_cost
  ) |> 
  arrange(desc(net_value_of_retention))

# 3. Summary for Presentation
# How many people to target?
target_count <- sum(business_impact$should_target)
total_exp_value <- sum(business_impact$net_value_of_retention[business_impact$should_target])

cat("\n--- Business Recommendation ---\n")
cat("Optimal number of customers to target:", target_count, "\n")
cat("Total Expected Value of Retention Plan: €", format(round(total_exp_value), big.mark=","), "\n")

# 4. Visualization: Cumulative Value Curve (Great for the presentation!)
# This shows where the profit peaks
business_impact |> 
  mutate(
    rank = row_number(),
    cumulative_value = cumsum(net_value_of_retention)
  ) |> 
  ggplot(aes(x = rank, y = cumulative_value)) +
  geom_line(color = "blue", size = 1.2) +
  geom_vline(xintercept = target_count, linetype = "dashed", color = "red") +
  annotate("text", x = target_count + 50, y = 0, label = "Optimal Cutoff", color = "red", angle = 90) +
  labs(title = "Optimization of Retention Campaign",
       subtitle = "Cumulative Net Value by Customer Rank",
       x = "Number of Customers Targeted",
       y = "Cumulative Expected Value (€)") +
  theme_minimal()

# =====================================================================
# 1. Logistic regression baseline recipe
# --------------------------------------
logistic_recipe <-
  recipe(churn ~ ., data = analysis_set) |>
  update_role(id, new_role = "metadata") |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_predictors())


# 2. Random forest recipe (with downsampling)
# -------------------------------------------------------------
rf_recipe <-
  recipe(churn ~ ., data = analysis_set) |>
  update_role(id, new_role = "metadata") |>
  themis::step_downsample(churn)

rf_recipe

# 3. Gradient boosted tree (xgboost) recipe 
# ------------------------------------------------------------
xgb_recipe <-
  recipe(churn ~ ., data = analysis_set) |>
  # remove id from the predictors (xgboost doesnt need an ID column)
  step_rm(id) |>
  # one-hot encode all categorical predictors (as in the xgb)
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  # handle class imbalance
  themis::step_downsample(churn)
xgb_recipe

#Check for later
logistic_recipe |> summary()
rf_recipe |> summary()
xgb_recipe |> summary()

glimpse(logistic_recipe)
glimpse(rf_recipe)
glimpse(xgb_recipe)



##########################################################################
# TODO: Identify and train a better predicting model and 
#       evaluate its predictions using 2025 customer data








########################################################################## 
# TODO: Demonstrate how to use the model to select which
#       customers we will contact










