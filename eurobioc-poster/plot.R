library(ragg)
library(ggplot2)

results <- read.csv(here::here("eurobioc-poster", "benchmark.csv"))

N <
ks <- unique(results$k)


p <- ggplot(results, aes(k, time_s, color = impl, group = impl)) +
          geom_line(linewidth = 1.4, na.rm = TRUE) +
            geom_point(size = 5, na.rm = TRUE) +
              scale_color_manual(values = c(
                                                "base R"   = "#888888",
                                                    "anvl CPU" = "#1a3a5c",
                                                    "anvl GPU" = "#c5832b"
                                                      )) +
  scale_y_log10(labels = function(x) ifelse(x < 1, paste0(round(x*1000), " ms"),
                                                                                        paste0(round(x, 1), " s"))) +
  scale_x_continuous(breaks = ks) +
    labs(
             x = "target rank k",
                 y = "wall time (log scale)",
                 color = NULL,
                     title = sprintf("Randomized SVD on a %d x %d matrix", N, P),
                     subtitle = "JIT compile + host<->device transfers excluded (steady-state cached calls)"
                       ) +
  theme_minimal(base_size = 16, base_family = "Arial") +
    theme(
              plot.title = element_text(face = "bold", size = 17),
                  plot.subtitle = element_text(size = 13, color = "#5a6b7d"),
                  legend.position = "right",
                      legend.text = element_text(size = 14),
                      legend.key.width = grid::unit(2, "lines")
                        )

  if (!has_cuda) {
            p <- p + annotate(
                                  "text",
                                      x = max(ks), y = min(results$time_s, na.rm = TRUE) * 0.55,
                                      label = "anvl GPU - cuda plugin not installed", hjust = 1, vjust = 0,
                                          color = "#888888", size = 6, fontface = "italic"
                                        )
  }

  ggsave("figures/benchmark.png", p, width = 9, height = 5, dpi = 200,
                device = ragg::agg_png)
  cat("Wrote figures/benchmark.png\n")

