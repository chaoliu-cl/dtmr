// src/rcpp_dtm.cpp
#include <Rcpp.h>
#include "TopicModel/DTModel.hpp"

using namespace Rcpp;

// [[Rcpp::export]]
SEXP create_dtm_model_rcpp(int k, int t, double alpha = 0.1,
                           double beta = 0.01, double gamma = 0.1,
                           int random_seed = -1) {
  // Create the model with the specified parameters
  tomoto::DTModel* model = new tomoto::DTModel(k, t, alpha, beta, gamma);

  // Set random seed if provided
  if (random_seed > 0) {
    model->set_random_seed(random_seed);
  }

  // Create an external pointer to the model
  XPtr<tomoto::DTModel> model_ptr(model, true);
  return model_ptr;
}

// [[Rcpp::export]]
void add_document_rcpp(SEXP model_ptr, CharacterVector words, int time_slice) {
  // Convert external pointer to DTModel pointer
  XPtr<tomoto::DTModel> model(model_ptr);

  // Convert R character vector to C++ string vector
  std::vector<std::string> word_vec(words.size());
  for (int i = 0; i < words.size(); i++) {
    word_vec[i] = as<std::string>(words[i]);
  }

  // Add document to the model
  model->add_doc(word_vec, time_slice);
}

// [[Rcpp::export]]
void train_model_rcpp(SEXP model_ptr, int iterations,
                      bool calc_perplexity = false,
                      bool show_progress = true) {
  // Convert external pointer to DTModel pointer
  XPtr<tomoto::DTModel> model(model_ptr);

  // Train the model
  model->train(iterations, calc_perplexity, show_progress);
}

// [[Rcpp::export]]
NumericMatrix get_topic_word_dist_rcpp(SEXP model_ptr, int topic, int time) {
  // Convert external pointer to DTModel pointer
  XPtr<tomoto::DTModel> model(model_ptr);

  // Get vocabulary size
  int vocab_size = model->get_vocab_size();

  // Create matrix to hold results
  NumericMatrix result(vocab_size, 2);

  // Get topic-word distribution
  auto dist = model->get_topic_word_dist(topic, time);

  // Copy results to R matrix
  for (int i = 0; i < vocab_size; i++) {
    result(i, 0) = i;  // Word ID
    result(i, 1) = dist[i];  // Probability
  }

  return result;
}

// [[Rcpp::export]]
NumericMatrix get_doc_topic_dist_rcpp(SEXP model_ptr, int doc_id) {
  // Convert external pointer to DTModel pointer
  XPtr<tomoto::DTModel> model(model_ptr);

  // Get number of topics
  int num_topics = model->get_num_topics();

  // Create matrix to hold results
  NumericMatrix result(num_topics, 2);

  // Get document-topic distribution
  auto dist = model->get_doc_topic_dist(doc_id);

  // Copy results to R matrix
  for (int i = 0; i < num_topics; i++) {
    result(i, 0) = i;  // Topic ID
    result(i, 1) = dist[i];  // Probability
  }

  return result;
}

// [[Rcpp::export]]
CharacterVector get_vocabulary_rcpp(SEXP model_ptr) {
  // Convert external pointer to DTModel pointer
  XPtr<tomoto::DTModel> model(model_ptr);

  // Get vocabulary
  auto vocab = model->get_vocabulary();

  // Convert to R character vector
  CharacterVector result(vocab.size());
  for (size_t i = 0; i < vocab.size(); i++) {
    result[i] = vocab[i];
  }

  return result;
}

// [[Rcpp::export]]
NumericVector get_perplexity_rcpp(SEXP model_ptr) {
  // Convert external pointer to DTModel pointer
  XPtr<tomoto::DTModel> model(model_ptr);

  // Get perplexity for each time slice
  auto perplexity = model->get_perplexity();

  // Convert to R numeric vector
  NumericVector result(perplexity.size());
  for (size_t i = 0; i < perplexity.size(); i++) {
    result[i] = perplexity[i];
  }

  return result;
}

// [[Rcpp::export]]
List get_model_info_rcpp(SEXP model_ptr) {
  // Convert external pointer to DTModel pointer
  XPtr<tomoto::DTModel> model(model_ptr);

  // Create list to hold model information
  List info;

  // Add model parameters
  info["num_topics"] = model->get_num_topics();
  info["num_times"] = model->get_num_times();
  info["vocab_size"] = model->get_vocab_size();
  info["num_docs"] = model->get_num_docs();

  return info;
}

// [[Rcpp::export]]
void set_word_prior_rcpp(SEXP model_ptr, int topic, CharacterVector words,
                         NumericVector priors) {
  // Convert external pointer to DTModel pointer
  XPtr<tomoto::DTModel> model(model_ptr);

  // Check input dimensions
  if (words.size() != priors.size()) {
    stop("Length of words and priors must match");
  }

  // Convert R inputs to C++ map
  std::unordered_map<std::string, double> word_prior;
  for (int i = 0; i < words.size(); i++) {
    word_prior[as<std::string>(words[i])] = priors[i];
  }

  // Set word priors
  model->set_word_prior(topic, word_prior);
}

// [[Rcpp::export]]
List get_top_words_rcpp(SEXP model_ptr, int topic, int time, int top_n = 10) {
  // Convert external pointer to DTModel pointer
  XPtr<tomoto::DTModel> model(model_ptr);

  // Get top words
  auto top_words = model->get_top_words(topic, time, top_n);

  // Convert to R lists
  CharacterVector words(top_words.size());
  NumericVector probs(top_words.size());

  for (size_t i = 0; i < top_words.size(); i++) {
    words[i] = top_words[i].first;
    probs[i] = top_words[i].second;
  }

  // Return as a list
  List result;
  result["words"] = words;
  result["probabilities"] = probs;

  return result;
}

// [[Rcpp::export]]
void save_model_rcpp(SEXP model_ptr, std::string filename) {
  // Convert external pointer to DTModel pointer
  XPtr<tomoto::DTModel> model(model_ptr);

  // Save model to file
  model->save(filename);
}

// [[Rcpp::export]]
SEXP load_model_rcpp(std::string filename) {
  // Create a new DTModel by loading from file
  tomoto::DTModel* model = new tomoto::DTModel(filename);

  // Create an external pointer to the model
  XPtr<tomoto::DTModel> model_ptr(model, true);
  return model_ptr;
}

// [[Rcpp::export]]
int get_num_docs_rcpp(SEXP model_ptr) {
  // Convert external pointer to DTModel pointer
  XPtr<tomoto::DTModel> model(model_ptr);

  // Return number of documents
  return model->get_num_docs();
}

// [[Rcpp::export]]
int get_num_topics_rcpp(SEXP model_ptr) {
  // Convert external pointer to DTModel pointer
  XPtr<tomoto::DTModel> model(model_ptr);

  // Return number of topics
  return model->get_num_topics();
}

// [[Rcpp::export]]
int get_num_times_rcpp(SEXP model_ptr) {
  // Convert external pointer to DTModel pointer
  XPtr<tomoto::DTModel> model(model_ptr);

  // Return number of time slices
  return model->get_num_times();
}

// [[Rcpp::export]]
int get_vocab_size_rcpp(SEXP model_ptr) {
  // Convert external pointer to DTModel pointer
  XPtr<tomoto::DTModel> model(model_ptr);

  // Return vocabulary size
  return model->get_vocab_size();
}

// [[Rcpp::export]]
NumericVector get_alpha_rcpp(SEXP model_ptr) {
  // Convert external pointer to DTModel pointer
  XPtr<tomoto::DTModel> model(model_ptr);

  // Get alpha parameter
  double alpha = model->get_alpha();

  // Return as NumericVector
  NumericVector result(1);
  result[0] = alpha;
  return result;
}

// [[Rcpp::export]]
NumericVector get_beta_rcpp(SEXP model_ptr) {
  // Convert external pointer to DTModel pointer
  XPtr<tomoto::DTModel> model(model_ptr);

  // Get beta parameter
  double beta = model->get_beta();

  // Return as NumericVector
  NumericVector result(1);
  result[0] = beta;
  return result;
}

// [[Rcpp::export]]
NumericVector get_gamma_rcpp(SEXP model_ptr) {
  // Convert external pointer to DTModel pointer
  XPtr<tomoto::DTModel> model(model_ptr);

  // Get gamma parameter
  double gamma = model->get_gamma();

  // Return as NumericVector
  NumericVector result(1);
  result[0] = gamma;
  return result;
}
