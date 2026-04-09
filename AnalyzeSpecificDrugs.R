library(data.table)
library(dplyr)
library(ggplot2)
library(viridis)

# ---- CONFIG ----
input_file <- "/Users/nguyent46/Library/CloudStorage/OneDrive-Cedars-SinaiHealthSystem/Offsides/offsides/results/2024-2024/hdpsm_nrep5_mratio5_maxsamp25000_drug_reaction_associations.csv"
or_thresh <- 2.0
top_n <- 50

# ---- LIST OF HYPOGLYCEMICS ----
# Replace with your actual drug names exactly as they appear in drug_name_x
drug_interest <- c(
  "liraglutide"
)

# ---- LOAD FILE ----
dt <- fread(input_file)

dt <- dt %>%
  mutate(
    drug = tolower(drug_name_x),
    reaction = reaction_name
  )

# ---- FILTER TO HYPOGLYCEMICS ----
dt_hypo <- dt %>%
  filter(drug %in% drug_interest)

# ---- FILTER ENRICHED EVENTS ----
enriched <- dt_hypo %>%
  filter(
    OR > or_thresh,
    is.finite(OR)
  )

# ---- COUNT ACROSS DRUGS ----
common_enriched <- enriched %>%
  group_by(reaction) %>%
  summarise(
    n_drugs = n_distinct(drug),
    median_OR = median(OR, na.rm = TRUE)
  ) %>%
  arrange(desc(n_drugs), desc(median_OR))

# ---- VISUALIZATION ----
top_events <- common_enriched %>%
  slice_max(order_by = n_drugs, n = top_n, with_ties = FALSE)

p <- ggplot(top_events,
            aes(x = reorder(reaction, n_drugs),
                y = n_drugs,
                fill = median_OR)) +
  geom_bar(stat = "identity", color = "black", linewidth = 0.3) +
  coord_flip() +
  scale_fill_viridis_c(option = "plasma", name = "Median OR") +
  labs(
    title = "Most Common Enriched Adverse Events Across Hypoglycemics",
    subtitle = paste0("Events enriched (OR > ", or_thresh, ")"),
    x = "Adverse Event",
    y = "Number of Hypoglycemic Drugs Enriched"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text.y = element_text(size = 11)
  )

print(p)

library(data.table)
library(dplyr)
library(ggplot2)
library(viridis)

# ---- CONFIG ----
input_file <- "offsides/results/2024-2024/hdpsm_nrep5_mratio5_maxsamp25000_drug_reaction_associations.csv"
or_thresh <- 2
top_n <- 50

hypoglycemics <- c(
  "acarbose","canagliflozin","dapagliflozin","dulaglutide",
  "empagliflozin","exenatide","glimepiride","glipizide",
  "glyburide","linagliptin","liraglutide","metformin",
  "miglitol","nateglinide","pioglitazone","repaglinide",
  "rosiglitazone","saxagliptin","sitagliptin","vildagliptin"
)

# ---- LOAD FILE ----
dt <- fread(input_file)

dt <- dt %>%
  mutate(
    drug = tolower(drug_name_x),
    reaction = reaction_name
  )

# ---- FILTER TO HYPOGLYCEMICS ----
dt_hypo <- dt %>%
  filter(drug %in% hypoglycemics)

# ---- FILTER ENRICHED EVENTS (UNCORRECTED) ----
enriched_uncorrected <- dt_hypo %>%
  filter(
    uncorrected_OR > or_thresh,
    is.finite(uncorrected_OR)
  )

# ---- COUNT ACROSS DRUGS ----
common_uncorrected <- enriched_uncorrected %>%
  group_by(reaction) %>%
  summarise(
    n_drugs = n_distinct(drug),
    median_OR = median(uncorrected_OR, na.rm = TRUE)
  ) %>%
  arrange(desc(n_drugs), desc(median_OR))

# ---- VISUALIZATION ----
top_events_uncorrected <- common_uncorrected %>%
  slice_max(order_by = n_drugs, n = top_n, with_ties = FALSE)

p_uncorrected <- ggplot(top_events_uncorrected,
                        aes(x = reorder(reaction, n_drugs),
                            y = n_drugs,
                            fill = median_OR)) +
  geom_bar(stat = "identity", color = "black", linewidth = 0.3) +
  coord_flip() +
  scale_fill_viridis_c(option = "plasma", name = "Median OR (Uncorrected)") +
  labs(
    title = "Most Common Enriched Adverse Events Across Hypoglycemics (Uncorrected)",
    subtitle = paste0("Events enriched (Uncorrected OR > ", or_thresh, ")"),
    x = "Adverse Event",
    y = "Number of Hypoglycemic Drugs Enriched"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text.y = element_text(size = 11)
  )

print(p_uncorrected)


