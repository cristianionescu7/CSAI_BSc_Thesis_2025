# Define list of required packages
required_packages <- c(
  "tidyverse", "ggplot2", "emmeans", "multcomp",
  "wesanderson", "ggsci", "gridExtra", "grid"
)

# Install missing packages
installed <- required_packages %in% rownames(installed.packages())
if (any(!installed)) {
  install.packages(required_packages[!installed])
}

# Load all packages
lapply(required_packages, library, character.only = TRUE)

# Read the data
df <- read_csv("models/subject_metrics.csv")

df_summary <- df %>%
  group_by(model_id, model_type, procedure, interval, electrodes) %>%
  summarise(
    acc_mean = mean(accuracy),
    acc_sd = sd(accuracy),
    f1_mean = mean(f1),
    f1_sd = sd(f1),
    .groups = "drop"
  ) %>%
  arrange(desc(acc_mean))
print(df_summary, width = Inf)

subject_summary <- df %>%
  filter(model_id == 2) %>%  # Focus only on best model
  group_by(subject) %>%
  summarise(
    `Mean Accuracy` = mean(accuracy),
    `Mean Precision` = mean(precision),
    `Mean Recall` = mean(recall),
    `Mean F1` = mean(f1),
    .groups = "drop"
  ) %>%
  arrange(`subject`)

subject_summary

plot_specs <- list(
  list(y = "acc_mean", sd = "acc_sd", label = "Mean Accuracy",  title = "Model Accuracy with SD"),
  list(y = "f1_mean",  sd = "f1_sd",  label = "Mean F1 Score",  title = "Model F1 Score with SD")
)

for (spec in plot_specs) {
  p <- ggplot(df_summary, aes_string(x = paste0("reorder(model_id, -", spec$y, ")"),
                                     y = spec$y, fill = "model_type")) +
    geom_col() +
    geom_errorbar(aes_string(ymin = paste0(spec$y, " - ", spec$sd),
                             ymax = paste0(spec$y, " + ", spec$sd)),
                  width = 0.2) +
    labs(
      x = "Model ID",
      y = spec$label,
      title = spec$title,
      fill = "Model Type"
    ) +
    scale_fill_manual(values = wes_palette("FrenchDispatch")) +
    theme(
      plot.title = element_text(size = 24, face = "bold"),
      axis.title = element_text(size = 20),
      axis.text = element_text(size = 16),
      legend.text = element_text(size = 18),
      legend.title = element_text(size = 20)
    )

  filename <- paste0(gsub(" ", "_", spec$title), ".png")
  ggsave(filename, plot = p, width = 10, height = 6, dpi = 300)
}

plot_specs <- list(
  list(x = "model_type",  y = "acc_mean", fill = "model_type",  palette = "FrenchDispatch",         title = "Accuracy by Model Type"),
  list(x = "model_type",  y = "f1_mean",  fill = "model_type",  palette = "FrenchDispatch",         title = "F1 Score by Model Type"),
  list(x = "procedure",  y = "acc_mean", fill = "procedure",  palette = "Royal1",         title = "Accuracy by Procedure"),
  list(x = "procedure",  y = "f1_mean",  fill = "procedure",  palette = "Royal1",         title = "F1 Score by Procedure"),
  list(x = "interval",   y = "acc_mean", fill = "interval",   palette = "GrandBudapest2", title = "Accuracy by Interval Type"),
  list(x = "interval",   y = "f1_mean",  fill = "interval",   palette = "GrandBudapest2", title = "F1 Score by Interval Type"),
  list(x = "electrodes", y = "acc_mean", fill = "electrodes", palette = "AsteroidCity2",  title = "Accuracy by Electrode Set"),
  list(x = "electrodes", y = "f1_mean",  fill = "electrodes", palette = "AsteroidCity2",  title = "F1 Score by Electrode Set")
)

for (spec in plot_specs) {
  fill_label <- tools::toTitleCase(gsub("_", " ", spec$fill))
  
  p <- ggplot(df_summary, aes_string(x = spec$x, y = spec$y, fill = spec$fill)) +
    geom_boxplot() +
    labs(
      x = tools::toTitleCase(spec$x),
      y = ifelse(spec$y == "acc_mean", "Accuracy", "F1 Score"),
      title = spec$title,
      fill = fill_label
    ) +
    scale_fill_manual(values = wes_palette(spec$palette)) +
    theme(
      plot.title = element_text(size = 24, face = "bold"),
      axis.title = element_text(size = 20),
      axis.text = element_text(size = 16),
      legend.position = "none"
    )
  
  filename <- paste0(gsub(" ", "_", spec$title), ".png")
  ggsave(filename, plot = p, width = 10, height = 6, dpi = 300)
}

plot_specs <- list(
  list(y = "accuracy", title = "Average Accuracy per Subject Across All Models"),
  list(y = "f1",       title = "Average F1 Score per Subject Across All Models")
)

for (spec in plot_specs) {
  p <- ggplot(df, aes_string(x = "factor(subject)", y = spec$y)) +
    geom_boxplot(fill = "lightblue") +
    labs(
      title = spec$title,
      x = "Subject",
      y = ifelse(spec$y == "accuracy", "Accuracy", "F1 Score")
    ) +
    theme(
      plot.title = element_text(size = 24, face = "bold"),
      axis.title = element_text(size = 20),
      axis.text = element_text(size = 16),
      legend.position = "none"
    )
  
  filename <- paste0(gsub(" ", "_", spec$title), ".png")
  ggsave(filename, plot = p, width = 10, height = 6, dpi = 300)
}

plot_specs <- list(
  list(y = "accuracy", fill = "model_type", palette = "FrenchDispatch",  title = "Accuracy per Subject by Model Type"),
  list(y = "f1",       fill = "model_type", palette = "FrenchDispatch",  title = "F1 score per Subject by Model Type"),
  list(y = "accuracy", fill = "procedure",  palette = "Royal1",          title = "Accuracy per Subject by Procedure"),
  list(y = "f1",       fill = "procedure",  palette = "Royal1",          title = "F1 score per Subject by Procedure"),
  list(y = "accuracy", fill = "interval",   palette = "GrandBudapest2",  title = "Accuracy per Subject by Interval Type"),
  list(y = "f1",       fill = "interval",   palette = "GrandBudapest2",  title = "F1 score per Subject by Interval Type"),
  list(y = "accuracy", fill = "electrodes", palette = "AsteroidCity2",   title = "Accuracy per Subject by Electrode Set"),
  list(y = "f1",       fill = "electrodes", palette = "AsteroidCity2",   title = "F1 score per Subject by Electrode Set")
)

for (spec in plot_specs) {
  p <- ggplot(df, aes_string(x = "factor(subject)", y = spec$y, fill = spec$fill)) +
    geom_boxplot() +
    labs(
      title = spec$title,
      x = "Subject",
      y = ifelse(spec$y == "accuracy", "Accuracy", "F1 Score"),
      fill_label <- tools::toTitleCase(gsub("_", " ", spec$fill)),
      fill = fill_label
    ) +
    scale_fill_manual(values = wes_palette(spec$palette)) +
    theme(
      plot.title = element_text(size = 24, face = "bold"),
      axis.title = element_text(size = 20),
      axis.text = element_text(size = 16),
      legend.text = element_text(size = 18),
      legend.title = element_text(size = 20)
    )
  
  filename <- paste0(gsub(" ", "_", spec$title), ".png")
  ggsave(filename, plot = p, width = 10, height = 10, dpi = 300)
}

# Make sure `procedure` and `interval` are factors
df$procedure <- factor(df$procedure, levels = c("within", "LOSO"))
df$interval <- factor(df$interval, levels = c("action", "full"))

metrics <- list(
  list(var = "accuracy", mean_col = "mean_accuracy", se_col = "se", ylab = "Mean Accuracy", title = "Interaction Between Procedure and Interval (Accuracy)"),
  list(var = "f1",       mean_col = "mean_f1",      se_col = "se", ylab = "Mean F1 Score", title = "Interaction Between Procedure and Interval (F1 Score)")
)

for (spec in metrics) {
  # Aggregate mean and SE
  summary_df <- aggregate(
    formula(paste(spec$var, "~ procedure + interval")),
    data = df,
    FUN = function(x) c(mean = mean(x), se = sd(x) / sqrt(length(x)))
  )
  
  summary_df <- do.call(data.frame, summary_df)
  colnames(summary_df) <- c("procedure", "interval", spec$mean_col, spec$se_col)
  
  # Build and save the plot
  p <- ggplot(summary_df, aes_string(x = "interval", y = spec$mean_col, color = "procedure", group = "procedure")) +
    geom_point(size = 4) +
    geom_line(linewidth = 1.5) +
    geom_errorbar(aes_string(ymin = paste0(spec$mean_col, " - ", spec$se_col),
                             ymax = paste0(spec$mean_col, " + ", spec$se_col)),
                  width = 0.1) +
    labs(
      title = spec$title,
      x = "Interval",
      y = spec$ylab,
      color = "Procedure"
    ) +
    scale_color_manual(values = wes_palette("Royal1")) +
    theme(
      plot.title = element_text(size = 24, face = "bold"),
      axis.title = element_text(size = 20),
      axis.text = element_text(size = 16),
      legend.text = element_text(size = 18),
      legend.title = element_text(size = 20)
    )
  
  filename <- paste0(gsub(" ", "_", spec$title), ".png")
  ggsave(filename, plot = p, width = 10, height = 6, dpi = 300)
}

plot_specs <- list(
  list(y = "accuracy", fill = "procedure",  palette = "Royal1",         title = "Accuracy by Model Type and Procedure"),
  list(y = "f1",       fill = "procedure",  palette = "Royal1",         title = "F1 Score by Model Type and Procedure"),
  list(y = "accuracy", fill = "interval",   palette = "GrandBudapest2", title = "Accuracy by Model Type and Interval"),
  list(y = "f1",       fill = "interval",   palette = "GrandBudapest2", title = "F1 Score by Model Type and Interval"),
  list(y = "accuracy", fill = "electrodes", palette = "AsteroidCity2",  title = "Accuracy by Model Type and Electrodes"),
  list(y = "f1",       fill = "electrodes", palette = "AsteroidCity2",  title = "F1 Score by Model Type and Electrodes")
)

for (spec in plot_specs) {
  p <- ggplot(df, aes_string(x = "model_type", y = spec$y, fill = spec$fill)) +
    geom_boxplot() +
    labs(
      title = spec$title,
      x = "Model Type",
      y = ifelse(spec$y == "accuracy", "Accuracy", "F1 Score"),
      fill = tools::toTitleCase(spec$fill)
    ) +
    scale_fill_manual(values = wes_palette(spec$palette)) +
    theme(
      plot.title = element_text(size = 24, face = "bold"),
      axis.title = element_text(size = 20),
      axis.text = element_text(size = 16),
      legend.text = element_text(size = 18),
      legend.title = element_text(size = 20)
    )
  
  filename <- paste0(gsub(" ", "_", spec$title), ".png")
  ggsave(filename, plot = p, width = 10, height = 10, dpi = 300)
}
