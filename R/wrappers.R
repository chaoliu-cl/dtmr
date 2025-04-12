#' Create a Dynamic Topic Model
#'
#' Creates a new Dynamic Topic Model (DTM) with the specified parameters.
#'
#' @param k Number of topics
#' @param t Number of time slices
#' @param alpha Dirichlet prior for document-topic distribution (default: 0.1)
#' @param beta Dirichlet prior for topic-word distribution (default: 0.01)
#' @param gamma Parameter controlling topic evolution between time slices (default: 0.1)
#' @param random_seed Random seed for reproducibility (default: -1, uses random initialization)
#' @return A DTModel object
#' @export
#'
#' @examples
#' # Create a model with 10 topics and 5 time slices
#' model <- create_dtm(10, 5)
create_dtm <- function(k, t, alpha = 0.1, beta = 0.01, gamma = 0.1, random_seed = -1) {
  # Parameter validation
  if (!is.numeric(k) || k <= 0 || k != as.integer(k))
    stop("Number of topics (k) must be a positive integer")
  if (!is.numeric(t) || t <= 0 || t != as.integer(t))
    stop("Number of time slices (t) must be a positive integer")
  if (!is.numeric(alpha) || alpha <= 0)
    stop("Alpha must be a positive number")
  if (!is.numeric(beta) || beta <= 0)
    stop("Beta must be a positive number")
  if (!is.numeric(gamma) || gamma <= 0 || gamma > 1)
    stop("Gamma must be a positive number less than or equal to 1")

  # Create model using Rcpp function
  model <- create_dtm_model_rcpp(as.integer(k), as.integer(t), alpha, beta, gamma, as.integer(random_seed))

  # Set class and return
  class(model) <- "DTModel"
  return(model)
}

#' Add a document to the DTM model
#'
#' @param model A DTModel object
#' @param text Character vector of tokens
#' @param time_slice Time slice for the document (0-indexed)
#' @return The model, invisibly
#' @export
#'
#' @examples
#' model <- create_dtm(10, 5)
#' add_document(model, c("word1", "word2", "word3"), time_slice = 0)
add_document <- function(model, text, time_slice) {
  # Parameter validation
  if (!inherits(model, "DTModel"))
    stop("Not a DTModel object")
  if (!is.character(text))
    stop("Text must be a character vector")
  if (!is.numeric(time_slice) || time_slice < 0 || time_slice != as.integer(time_slice))
    stop("Time slice must be a non-negative integer")

  # Add document using Rcpp function
  add_document_rcpp(model, text, as.integer(time_slice))

  # Return model invisibly
  invisible(model)
}

#' Add multiple documents to the DTM model
#'
#' @param model A DTModel object
#' @param docs A list of character vectors, each representing a tokenized document
#' @param time_slices A vector of time slices corresponding to each document
#' @return The model, invisibly
#' @export
#'
#' @examples
#' model <- create_dtm(10, 5)
#' docs <- list(c("word1", "word2"), c("word2", "word3"))
#' time_slices <- c(0, 1)
#' add_documents(model, docs, time_slices)
add_documents <- function(model, docs, time_slices) {
  # Parameter validation
  if (!inherits(model, "DTModel"))
    stop("Not a DTModel object")
  if (!is.list(docs))
    stop("Documents must be a list of character vectors")
  if (length(docs) != length(time_slices))
    stop("Length of docs and time_slices must match")

  # Add each document
  for (i in seq_along(docs)) {
    add_document(model, docs[[i]], time_slices[i])
  }

  # Return model invisibly
  invisible(model)
}

#' Train the dynamic topic model
#'
#' @param model A DTModel object
#' @param iterations Number of training iterations
#' @param calc_perplexity Whether to calculate perplexity during training
#' @param show_progress Whether to show training progress
#' @return The model, invisibly
#' @export
#'
#' @examples
#' model <- create_dtm(10, 5)
#' # ... add documents ...
#' train(model, iterations = 100)
train <- function(model, iterations = 100, calc_perplexity = FALSE, show_progress = TRUE) {
  # Parameter validation
  if (!inherits(model, "DTModel"))
    stop("Not a DTModel object")
  if (!is.numeric(iterations) || iterations <= 0 || iterations != as.integer(iterations))
    stop("Iterations must be a positive integer")

  # Train model using Rcpp function
  train_model_rcpp(model, as.integer(iterations), calc_perplexity, show_progress)

  # Return model invisibly
  invisible(model)
}

#' Get topic-word distribution
#'
#' Gets the distribution of words for a specific topic and time slice.
#'
#' @param model A DTModel object
#' @param topic Topic index (0-indexed)
#' @param time Time slice (0-indexed)
#' @return A matrix with word indices and probabilities
#' @export
#'
#' @examples
#' model <- create_dtm(10, 5)
#' # ... add documents and train model ...
#' topic_word_dist <- get_topic_word_dist(model, topic = 0, time = 0)
get_topic_word_dist <- function(model, topic, time) {
  # Parameter validation
  if (!inherits(model, "DTModel"))
    stop("Not a DTModel object")

  # Get topic-word distribution using Rcpp function
  dist <- get_topic_word_dist_rcpp(model, as.integer(topic), as.integer(time))

  # Return distribution matrix
  return(dist)
}

#' Get document-topic distribution
#'
#' Gets the distribution of topics for a specific document.
#'
#' @param model A DTModel object
#' @param doc_id Document index (0-indexed)
#' @return A matrix with topic indices and probabilities
#' @export
#'
#' @examples
#' model <- create_dtm(10, 5)
#' # ... add documents and train model ...
#' doc_topic_dist <- get_doc_topic_dist(model, doc_id = 0)
get_doc_topic_dist <- function(model, doc_id) {
  # Parameter validation
  if (!inherits(model, "DTModel"))
    stop("Not a DTModel object")

  # Get document-topic distribution using Rcpp function
  dist <- get_doc_topic_dist_rcpp(model, as.integer(doc_id))

  # Return distribution matrix
  return(dist)
}

#' Get model vocabulary
#'
#' @param model A DTModel object
#' @return A character vector containing the vocabulary
#' @export
#'
#' @examples
#' model <- create_dtm(10, 5)
#' # ... add documents ...
#' vocabulary <- get_vocabulary(model)
get_vocabulary <- function(model) {
  # Parameter validation
  if (!inherits(model, "DTModel"))
    stop("Not a DTModel object")

  # Get vocabulary using Rcpp function
  vocab <- get_vocabulary_rcpp(model)

  # Return vocabulary
  return(vocab)
}

#' Get model perplexity
#'
#' Gets the perplexity (a measure of model fit) for each time slice.
#'
#' @param model A DTModel object
#' @return A numeric vector with perplexity for each time slice
#' @export
#'
#' @examples
#' model <- create_dtm(10, 5)
#' # ... add documents and train model with calc_perplexity = TRUE ...
#' perplexity <- get_perplexity(model)
get_perplexity <- function(model) {
  # Parameter validation
  if (!inherits(model, "DTModel"))
    stop("Not a DTModel object")

  # Get perplexity using Rcpp function
  perp <- get_perplexity_rcpp(model)

  # Return perplexity
  return(perp)
}

#' Get model information
#'
#' Gets basic information about the model.
#'
#' @param model A DTModel object
#' @return A list with model information
#' @export
#'
#' @examples
#' model <- create_dtm(10, 5)
#' # ... add documents ...
#' model_info <- get_model_info(model)
get_model_info <- function(model) {
  # Parameter validation
  if (!inherits(model, "DTModel"))
    stop("Not a DTModel object")

  # Get model information using Rcpp function
  info <- get_model_info_rcpp(model)

  # Return model information
  return(info)
}

#' Set word prior
#'
#' Sets prior probabilities for words in a specific topic. This can be used to guide
#' the topic model to associate certain words with specific topics.
#'
#' @param model A DTModel object
#' @param topic Topic index (0-indexed)
#' @param words Character vector of words
#' @param priors Numeric vector of prior probabilities
#' @return The model, invisibly
#' @export
#'
#' @examples
#' model <- create_dtm(10, 5)
#' # ... add documents ...
#' set_word_prior(model, topic = 0,
#'               words = c("science", "research"),
#'               priors = c(2.0, 1.5))
set_word_prior <- function(model, topic, words, priors) {
  # Parameter validation
  if (!inherits(model, "DTModel"))
    stop("Not a DTModel object")
  if (!is.character(words))
    stop("Words must be a character vector")
  if (!is.numeric(priors))
    stop("Priors must be a numeric vector")
  if (length(words) != length(priors))
    stop("Length of words and priors must match")

  # Set word prior using Rcpp function
  set_word_prior_rcpp(model, as.integer(topic), words, priors)

  # Return model invisibly
  invisible(model)
}

#' Get top words for a topic
#'
#' Gets the top words (by probability) for a specific topic and time slice.
#'
#' @param model A DTModel object
#' @param topic Topic index (0-indexed)
#' @param time Time slice (0-indexed)
#' @param top_n Number of top words to return
#' @return A list with words and probabilities
#' @export
#'
#' @examples
#' model <- create_dtm(10, 5)
#' # ... add documents and train model ...
#' top_words <- get_top_words(model, topic = 0, time = 0, top_n = 10)
get_top_words <- function(model, topic, time, top_n = 10) {
  # Parameter validation
  if (!inherits(model, "DTModel"))
    stop("Not a DTModel object")
  if (!is.numeric(top_n) || top_n <= 0 || top_n != as.integer(top_n))
    stop("top_n must be a positive integer")

  # Get top words using Rcpp function
  top_words <- get_top_words_rcpp(model, as.integer(topic), as.integer(time), as.integer(top_n))

  # Convert to data frame
  result <- data.frame(
    word = top_words$words,
    probability = top_words$probabilities,
    stringsAsFactors = FALSE
  )

  # Return result
  return(result)
}

#' Save model to file
#'
#' @param model A DTModel object
#' @param filename Path to save the model
#' @return The model, invisibly
#' @export
#'
#' @examples
#' model <- create_dtm(10, 5)
#' # ... add documents and train model ...
#' save_model(model, "my_model.bin")
save_model <- function(model, filename) {
  # Parameter validation
  if (!inherits(model, "DTModel"))
    stop("Not a DTModel object")
  if (!is.character(filename) || length(filename) != 1)
    stop("Filename must be a single character string")

  # Save model using Rcpp function
  save_model_rcpp(model, filename)

  # Return model invisibly
  invisible(model)
}

#' Load model from file
#'
#' @param filename Path to the model file
#' @return A DTModel object
#' @export
#'
#' @examples
#' # ... previously saved model ...
#' model <- load_model("my_model.bin")
load_model <- function(filename) {
  # Parameter validation
  if (!is.character(filename) || length(filename) != 1)
    stop("Filename must be a single character string")
  if (!file.exists(filename))
    stop("File does not exist")

  # Load model using Rcpp function
  model <- load_model_rcpp(filename)

  # Set class and return
  class(model) <- "DTModel"
  return(model)
}

#' Print method for DTModel objects
#'
#' @param x A DTModel object
#' @param ... Additional arguments (not used)
#' @export
print.DTModel <- function(x, ...) {
  # Get model information
  info <- get_model_info(x)

  # Print basic information
  cat("Dynamic Topic Model\n")
  cat("Number of topics:", info$num_topics, "\n")
  cat("Number of time slices:", info$num_times, "\n")
  cat("Vocabulary size:", info$vocab_size, "\n")
  cat("Number of documents:", info$num_docs, "\n")
}

#' Summary method for DTModel objects
#'
#' @param object A DTModel object
#' @param ... Additional arguments (not used)
#' @export
summary.DTModel <- function(object, ...) {
  # Get model information
  info <- get_model_info(object)

  # Create summary object
  result <- list(
    num_topics = info$num_topics,
    num_times = info$num_times,
    vocab_size = info$vocab_size,
    num_docs = info$num_docs
  )

  # Add perplexity if available
  if (info$num_docs > 0) {
    tryCatch({
      result$perplexity <- get_perplexity(object)
    }, error = function(e) {
      # Perplexity might not be available if not calculated during training
    })
  }

  # Add top words for each topic and time slice
  if (info$num_docs > 0 && info$vocab_size > 0) {
    top_words <- list()
    for (t in 0:(info$num_times - 1)) {
      top_words[[t + 1]] <- list()
      for (k in 0:(info$num_topics - 1)) {
        tryCatch({
          top_words[[t + 1]][[k + 1]] <- get_top_words(object, k, t, top_n = 10)
        }, error = function(e) {
          # Skip if there's an error getting top words
        })
      }
    }
    result$top_words <- top_words
  }

  class(result) <- "summary.DTModel"
  return(result)
}

#' Print method for summary.DTModel objects
#'
#' @param x A summary.DTModel object
#' @param ... Additional arguments (not used)
#' @export
print.summary.DTModel <- function(x, ...) {
  cat("Dynamic Topic Model Summary\n")
  cat("Number of topics:", x$num_topics, "\n")
  cat("Number of time slices:", x$num_times, "\n")
  cat("Vocabulary size:", x$vocab_size, "\n")
  cat("Number of documents:", x$num_docs, "\n")

  if (x$num_docs > 0 && !is.null(x$perplexity)) {
    cat("\nPerplexity by time slice:\n")
    for (t in 1:length(x$perplexity)) {
      cat("  Time", t-1, ":", round(x$perplexity[t], 2), "\n")
    }
  }

  if (x$num_docs > 0 && !is.null(x$top_words)) {
    cat("\nTop words for selected topics and time slices:\n")
    # Print top words for a subset of topics and time slices
    max_times_to_show <- min(3, x$num_times)
    max_topics_to_show <- min(3, x$num_topics)

    for (t in 1:max_times_to_show) {
      cat("\nTime slice", t-1, ":\n")
      for (k in 1:max_topics_to_show) {
        if (!is.null(x$top_words[[t]][[k]])) {
          words <- x$top_words[[t]][[k]]$word
          cat("  Topic", k-1, ":", paste(words[1:min(5, length(words))], collapse = ", "), "...\n")
        }
      }
      if (x$num_topics > max_topics_to_show) cat("  ...\n")
    }
    if (x$num_times > max_times_to_show) cat("...\n")
  }
}
