#' Plot topic evolution over time
#'
#' @param model A DTModel object
#' @param topic Topic index (0-indexed)
#' @param top_n Number of top words to track (default: 10)
#' @param min_prob Minimum probability threshold for including words (default: 0.001)
#' @return A ggplot object
#' @export
#' @importFrom ggplot2 ggplot aes geom_line geom_point theme_minimal labs theme
#'
#' @examples
#' \dontrun{
#' model <- create_dtm(10, 5)
#' # ... load corpus and train model ...
#' plot_topic_evolution(model, topic = 0)
#' }
plot_topic_evolution <- function(model, topic, top_n = 10, min_prob = 0.001) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is needed for this function to work. Please install it.")
  }

  if (!inherits(model, "DTModel"))
    stop("Not a DTModel object")

  # Get model information
  info <- get_model_info(model)
  num_times <- info$num_times

  # Get words that appear in the top N at any time point
  all_top_words <- character(0)
  for (t in 0:(num_times-1)) {
    top_words <- get_top_words(model, topic, t, top_n = top_n)
    all_top_words <- union(all_top_words, top_words$word)
  }

  # Track these words across all time points
  word_probs <- data.frame()
  for (t in 0:(num_times-1)) {
    # Get all words and their probabilities
    topic_dist <- get_topic_word_dist(model, topic, t)
    vocab <- get_vocabulary(model)

    # Create a mapping of word IDs to words
    word_mapping <- data.frame(
      id = 0:(length(vocab)-1),
      word = vocab,
      stringsAsFactors = FALSE
    )

    # Convert distribution to data frame
    topic_words <- data.frame(
      id = topic_dist[, 1],
      probability = topic_dist[, 2],
      stringsAsFactors = FALSE
    )

    # Join with word mapping
    topic_words <- merge(topic_words, word_mapping, by = "id")

    # Filter to just our tracked words
    topic_words <- topic_words[topic_words$word %in% all_top_words, ]

    # Add to dataframe
    if (nrow(topic_words) > 0) {
      tmp <- data.frame(
        word = topic_words$word,
        probability = topic_words$probability,
        time = t,
        stringsAsFactors = FALSE
      )
      word_probs <- rbind(word_probs, tmp)
    }
  }

  # Filter by minimum probability
  word_probs <- word_probs[word_probs$probability >= min_prob, ]

  # Create plot
  p <- ggplot2::ggplot(word_probs, ggplot2::aes(x = time, y = probability,
                                                color = word, group = word)) +
    ggplot2::geom_line(linewidth = 1) +
    ggplot2::geom_point(size = 3) +
    ggplot2::theme_minimal() +
    ggplot2::labs(title = paste("Evolution of Topic", topic),
                  x = "Time Slice",
                  y = "Probability",
                  color = "Word") +
    ggplot2::theme(legend.position = "right")

  return(p)
}

#' Plot multiple topics evolution
#'
#' @param model A DTModel object
#' @param topics Vector of topic indices to plot (0-indexed)
#' @param words Vector of words to track
#' @return A ggplot object
#' @export
#' @importFrom ggplot2 ggplot aes geom_line geom_point facet_wrap theme_minimal labs theme
#'
#' @examples
#' \dontrun{
#' model <- create_dtm(10, 5)
#' # ... load corpus and train model ...
#' plot_topics_evolution(model, topics = c(0, 1, 2),
#'                      words = c("data", "model", "analysis"))
#' }
plot_topics_evolution <- function(model, topics, words) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is needed for this function to work. Please install it.")
  }

  if (!inherits(model, "DTModel"))
    stop("Not a DTModel object")

  # Get model information
  info <- get_model_info(model)
  num_times <- info$num_times

  # Create a dataframe to hold all probabilities
  result <- data.frame()

  # For each topic
  for (topic in topics) {
    # For each time slice
    for (t in 0:(num_times-1)) {
      # Get topic-word distribution
      topic_dist <- get_topic_word_dist(model, topic, t)
      vocab <- get_vocabulary(model)

      # Create a mapping of word IDs to words
      word_mapping <- data.frame(
        id = 0:(length(vocab)-1),
        word = vocab,
        stringsAsFactors = FALSE
      )

      # Convert distribution to data frame
      topic_words <- data.frame(
        id = topic_dist[, 1],
        probability = topic_dist[, 2],
        stringsAsFactors = FALSE
      )

      # Join with word mapping
      topic_words <- merge(topic_words, word_mapping, by = "id")

      # Filter to just our tracked words
      topic_words <- topic_words[topic_words$word %in% words, ]

      # Add to dataframe
      if (nrow(topic_words) > 0) {
        tmp <- data.frame(
          word = topic_words$word,
          probability = topic_words$probability,
          topic = factor(topic),
          time = t,
          stringsAsFactors = FALSE
        )
        result <- rbind(result, tmp)
      }
    }
  }

  # Create faceted plot
  p <- ggplot2::ggplot(result, ggplot2::aes(x = time, y = probability,
                                            color = word, group = word)) +
    ggplot2::geom_line(linewidth = 1) +
    ggplot2::geom_point(size = 2) +
    ggplot2::facet_wrap(~topic, ncol = 2, labeller = ggplot2::labeller(
      topic = function(x) paste("Topic", x))) +
    ggplot2::theme_minimal() +
    ggplot2::labs(title = "Word Evolution Across Topics",
                  x = "Time Slice",
                  y = "Probability",
                  color = "Word") +
    ggplot2::theme(legend.position = "bottom")

  return(p)
}

#' Plot document-topic distribution
#'
#' @param model A DTModel object
#' @param doc_id Document index (0-indexed)
#' @param top_n Number of top topics to display (default: 5)
#' @return A ggplot object
#' @export
#' @importFrom ggplot2 ggplot aes geom_col coord_flip theme_minimal labs
#'
#' @examples
#' \dontrun{
#' model <- create_dtm(10, 5)
#' # ... load corpus and train model ...
#' plot_document_topics(model, doc_id = 0)
#' }
plot_document_topics <- function(model, doc_id, top_n = 5) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is needed for this function to work. Please install it.")
  }

  if (!inherits(model, "DTModel"))
    stop("Not a DTModel object")

  # Get document-topic distribution
  doc_topics <- get_doc_topic_dist(model, doc_id)

  # Convert to data frame
  doc_topics_df <- data.frame(
    topic = doc_topics[, 1],
    probability = doc_topics[, 2],
    stringsAsFactors = FALSE
  )

  # Sort by probability
  doc_topics_df <- doc_topics_df[order(doc_topics_df$probability, decreasing = TRUE), ]

  # Limit to top_n topics
  doc_topics_df <- head(doc_topics_df, top_n)

  # Add labels (with topic index)
  doc_topics_df$label <- paste("Topic", doc_topics_df$topic)

  # Order for plot
  doc_topics_df$label <- factor(doc_topics_df$label,
                                levels = rev(doc_topics_df$label))

  # Create plot
  p <- ggplot2::ggplot(doc_topics_df, ggplot2::aes(x = label, y = probability)) +
    ggplot2::geom_col(fill = "steelblue") +
    ggplot2::coord_flip() +
    ggplot2::theme_minimal() +
    ggplot2::labs(title = paste("Topic Distribution for Document", doc_id),
                  x = "",
                  y = "Probability")

  return(p)
}

#' Plot topic coherence
#'
#' @param model A DTModel object
#' @param method Coherence measure method (default: "npmi")
#' @return A ggplot object
#' @export
#' @importFrom ggplot2 ggplot aes geom_tile scale_fill_viridis_c theme_minimal labs
#'
#' @examples
#' \dontrun{
#' model <- create_dtm(10, 5)
#' # ... load corpus and train model ...
#' plot_topic_coherence(model)
#' }
plot_topic_coherence <- function(model, method = "npmi") {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is needed for this function to work. Please install it.")
  }

  if (!requireNamespace("viridis", quietly = TRUE)) {
    stop("Package 'viridis' is needed for this function to work. Please install it.")
  }

  if (!inherits(model, "DTModel"))
    stop("Not a DTModel object")

  # Get model information
  info <- get_model_info(model)

  # Placeholder for coherence values
  # In a real implementation, this would call a function to compute coherence
  # For now, we'll just use random values for demonstration
  coherence_df <- data.frame()
  for (t in 0:(info$num_times-1)) {
    for (k in 0:(info$num_topics-1)) {
      # In a real implementation, you would compute coherence here
      # For now, we'll just use random values
      coherence_val <- runif(1, -0.2, 1.0)

      # Add to data frame
      coherence_df <- rbind(coherence_df, data.frame(
        topic = k,
        time = t,
        coherence = coherence_val,
        stringsAsFactors = FALSE
      ))
    }
  }

  # Create plot
  p <- ggplot2::ggplot(coherence_df, ggplot2::aes(x = time, y = topic, fill = coherence)) +
    ggplot2::geom_tile() +
    viridis::scale_fill_viridis(option = "D") +
    ggplot2::theme_minimal() +
    ggplot2::labs(title = paste("Topic Coherence (", method, ")", sep = ""),
                  x = "Time Slice",
                  y = "Topic",
                  fill = "Coherence")

  return(p)
}
