# Define list of required packages
required_packages <- c(
  "tidyverse", "ggplot2", "emmeans", "multcomp",
  "wesanderson", "ggsci", "gridExtra", "grid", "effsize"
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

#ANOVA
aov1 <- aov(accuracy ~ model_type * procedure * electrodes * interval, data = df) # nolint # nolint
summary(aov1)

#ANOVA
aov2 <- aov(f1 ~ model_type * procedure * electrodes * interval, data = df) # nolint # nolint
summary(aov2)

#effect size of interval
cohens_d <- cohen.d(accuracy ~ interval, data = df)
print(cohens_d)

cohens_d <- cohen.d(f1 ~ interval, data = df)
print(cohens_d)

#subject as factor
df$subject <- as.factor(df$subject)

aov_within1 <- aov(accuracy ~ subject, data = df[df$procedure == "within", ])
summary(aov_within1)

aov_within2 <- aov(f1 ~ subject, data = df[df$procedure == "within", ])
summary(aov_within2)

#Post-hoc pairwise subject comparisons
model_subject_acc <- aov(accuracy ~ subject, data = df[df$procedure == "within", ]) # nolint
emmeans(model_subject_acc, pairwise ~ subject)

# Fit the model with factor as usual
model1 <- lm(accuracy ~ subject, data = df[df$procedure == "within", ])

# Define contrast: Subject 8 vs all others
k <- rep(-1 / (length(unique(df$subject)) - 1), length(unique(df$subject)))
names(k) <- levels(df$subject)
k["8"] <- 1

# Run contrast test
test <- glht(model1, linfct = mcp(subject = k))
summary(test)

#Post-hoc pairwise f1 subject comparisons
model_subject_f1 <- aov(f1 ~ subject, data = df[df$procedure == "within", ])
emmeans(model_subject_f1, pairwise ~ subject)

# Fit the model with factor as usual
model2 <- lm(f1 ~ subject, data = df[df$procedure == "within", ])

# Define contrast: Subject 8 vs all others
k <- rep(-1 / (length(unique(df$subject)) - 1), length(unique(df$subject)))
names(k) <- levels(df$subject)
k["8"] <- 1

# Run contrast test
test <- glht(model2, linfct = mcp(subject = k))
summary(test)
