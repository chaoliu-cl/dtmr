#' Preprocess text for DTM modeling
#'
#' @param text A character vector of raw text documents
#' @param lowercase Convert to lowercase (default: TRUE)
#' @param remove_numbers Remove numbers (default: TRUE)
#' @param remove_punctuation Remove punctuation (default: TRUE)
#' @param remove_stopwords Remove stopwords (default: TRUE)
#' @param min_chars Minimum number of characters for words (default: 3)
#' @param stopwords Character vector of stopwords to remove (default: common English stopwords)
#' @return A list of character vectors with tokenized words
#' @export
#'
#' @examples
#' texts <- c("This is a sample document.", "Another example text with numbers 123.")
#' preprocessed <- preprocess_text(texts)
preprocess_text <- function(text, lowercase = TRUE, remove_numbers = TRUE,
                            remove_punctuation = TRUE, remove_stopwords = TRUE,
                            min_chars = 3, stopwords = NULL) {

  # Default English stopwords if none provided
  if (is.null(stopwords) && remove_stopwords) {
    stopwords <- c("a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
                   "from", "had", "has", "have", "he", "her", "his", "i", "in",
                   "is", "it", "of", "on", "that", "the", "they", "this", "to",
                   "was", "were", "will", "with")
  }

  # Process each document
  processed <- vector("list", length(text))
  for (i in seq_along(text)) {
    # Get current document text
    doc <- text[i]

    # Convert to lowercase if requested
    if (lowercase) {
      doc <- tolower(doc)
    }

    # Remove numbers if requested
    if (remove_numbers) {
      doc <- gsub("[0-9]+", "", doc)
    }

    # Remove punctuation if requested
    if (remove_punctuation) {
      doc <- gsub("[[:punct:]]+", "", doc)
    }

    # Tokenize (split by whitespace)
    tokens <- strsplit(doc, "\\s+")[[1]]

    # Remove empty tokens
    tokens <- tokens[tokens != ""]

    # Filter by minimum length
    if (min_chars > 0) {
      tokens <- tokens[nchar(tokens) >= min_chars]
    }

    # Remove stopwords if requested
    if (remove_stopwords && !is.null(stopwords)) {
      tokens <- tokens[!tokens %in% stopwords]
    }

    # Save processed tokens
    processed[[i]] <- tokens
  }

  return(processed)
}

#' Create a DTM-ready corpus from a data frame
#'
#' @param df A data frame with text and time columns
#' @param text_col Name of the column containing document text (default: "text")
#' @param time_col Name of the column containing time information (default: "time")
#' @param preprocess Whether to preprocess text (default: TRUE)
#' @param ... Additional arguments passed to preprocess_text()
#' @return A list with documents and time slices
#' @export
#'
#' @examples
#' df <- data.frame(
#'   text = c("This is document 1", "This is document 2"),
#'   time = c(0, 1)
#' )
#' corpus <- create_corpus(df)
create_corpus <- function(df, text_col = "text", time_col = "time",
                          preprocess = TRUE, ...) {
  if (!is.data.frame(df)) {
    stop("Input must be a data frame")
  }
  if (!text_col %in% colnames(df)) {
    stop(paste("Text column", text_col, "not found in data frame"))
  }
  if (!time_col %in% colnames(df)) {
    stop(paste("Time column", time_col, "not found in data frame"))
  }

  # Extract text and time
  texts <- df[[text_col]]
  times <- df[[time_col]]

  # Preprocess if requested
  if (preprocess) {
    processed <- preprocess_text(texts, ...)
  } else {
    # Just tokenize without other preprocessing
    processed <- lapply(texts, function(doc) {
      strsplit(doc, "\\s+")[[1]]
    })
  }

  # Create corpus
  corpus <- list(
    documents = processed,
    times = times
  )

  class(corpus) <- "DTMCorpus"
  return(corpus)
}

#' Load corpus from model
#'
#' @param model A DTModel object
#' @param corpus A DTMCorpus object
#' @return The model, with corpus loaded
#' @export
#'
#' @examples
#' model <- create_dtm(10, 5)
#' df <- data.frame(
#'   text = c("This is document 1", "This is document 2"),
#'   time = c(0, 1)
#' )
#' corpus <- create_corpus(df)
#' load_corpus(model, corpus)
load_corpus <- function(model, corpus) {
  # Parameter validation
  if (!inherits(model, "DTModel"))
    stop("Not a DTModel object")
  if (!inherits(corpus, "DTMCorpus"))
    stop("Not a DTMCorpus object")

  # Add each document in the corpus to the model
  for (i in seq_along(corpus$documents)) {
    add_document(model, corpus$documents[[i]], corpus$times[i])
  }

  # Return model invisibly
  invisible(model)
}

#' Load a corpus from text files
#'
#' @param files Character vector of file paths
#' @param times Vector of time slices corresponding to each file
#' @param preprocess Whether to preprocess text (default: TRUE)
#' @param encoding File encoding (default: "UTF-8")
#' @param ... Additional arguments passed to preprocess_text()
#' @return A DTMCorpus object
#' @export
#'
#' @examples
#' \dontrun{
#' files <- c("doc1.txt", "doc2.txt")
#' times <- c(0, 1)
#' corpus <- load_text_files(files, times)
#' }
load_text_files <- function(files, times, preprocess = TRUE,
                            encoding = "UTF-8", ...) {
  if (length(files) != length(times)) {
    stop("Length of files and times must match")
  }

  # Read files
  texts <- character(length(files))
  for (i in seq_along(files)) {
    tryCatch({
      texts[i] <- paste(readLines(files[i], encoding = encoding), collapse = " ")
    }, error = function(e) {
      warning(paste("Could not read file:", files[i], "-", e$message))
      texts[i] <- ""
    })
  }

  # Create data frame and use create_corpus
  df <- data.frame(text = texts, time = times, stringsAsFactors = FALSE)
  return(create_corpus(df, preprocess = preprocess, ...))
}

#' Print method for DTMCorpus objects
#'
#' @param x A DTMCorpus object
#' @param ... Additional arguments (not used)
#' @export
print.DTMCorpus <- function(x, ...) {
  cat("DTModel Corpus\n")
  cat("Number of documents:", length(x$documents), "\n")

  # Count documents per time slice
  time_counts <- table(x$times)
  cat("Documents per time slice:\n")
  for (t in names(time_counts)) {
    cat("  Time", t, ":", time_counts[t], "documents\n")
  }

  # Sample some documents
  cat("Sample documents:\n")
  n_samples <- min(3, length(x$documents))
  for (i in 1:n_samples) {
    cat("  Document", i, "time", x$times[i], ":",
        paste(head(x$documents[[i]], 5), collapse = " "), "...\n")
  }
}

#' Convert corpus to DTM model
#'
#' @param corpus A DTMCorpus object
#' @param k Number of topics
#' @param alpha Dirichlet prior for document-topic distribution (default: 0.1)
#' @param beta Dirichlet prior for topic-word distribution (default: 0.01)
#' @param gamma Parameter controlling topic evolution between time slices (default: 0.1)
#' @param random_seed Random seed for reproducibility (default: -1)
#' @return A DTModel object
#' @export
#'
#' @examples
#' \dontrun{
#' df <- data.frame(
#'   text = c("This is document 1", "This is document 2"),
#'   time = c(0, 1)
#' )
#' corpus <- create_corpus(df)
#' model <- corpus_to_model(corpus, k = 10)
#' }
corpus_to_model <- function(corpus, k, alpha = 0.1, beta = 0.01, gamma = 0.1, random_seed = -1) {
  if (!inherits(corpus, "DTMCorpus")) {
    stop("Input must be a DTMCorpus object")
  }

  # Determine number of time slices
  t <- length(unique(corpus$times))

  # Create model
  model <- create_dtm(k, t, alpha, beta, gamma, random_seed)

  # Add documents
  load_corpus(model, corpus)

  return(model)
}
