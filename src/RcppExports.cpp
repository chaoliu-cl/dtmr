// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// create_dtm_model_rcpp
SEXP create_dtm_model_rcpp(int k, int t, double alpha, double beta, double gamma, int random_seed);
RcppExport SEXP _DTModelR_create_dtm_model_rcpp(SEXP kSEXP, SEXP tSEXP, SEXP alphaSEXP, SEXP betaSEXP, SEXP gammaSEXP, SEXP random_seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< int >::type t(tSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< int >::type random_seed(random_seedSEXP);
    rcpp_result_gen = Rcpp::wrap(create_dtm_model_rcpp(k, t, alpha, beta, gamma, random_seed));
    return rcpp_result_gen;
END_RCPP
}
// add_document_rcpp
void add_document_rcpp(SEXP model_ptr, CharacterVector words, int time_slice);
RcppExport SEXP _DTModelR_add_document_rcpp(SEXP model_ptrSEXP, SEXP wordsSEXP, SEXP time_sliceSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type model_ptr(model_ptrSEXP);
    Rcpp::traits::input_parameter< CharacterVector >::type words(wordsSEXP);
    Rcpp::traits::input_parameter< int >::type time_slice(time_sliceSEXP);
    add_document_rcpp(model_ptr, words, time_slice);
    return R_NilValue;
END_RCPP
}
// train_model_rcpp
void train_model_rcpp(SEXP model_ptr, int iterations, bool calc_perplexity, bool show_progress);
RcppExport SEXP _DTModelR_train_model_rcpp(SEXP model_ptrSEXP, SEXP iterationsSEXP, SEXP calc_perplexitySEXP, SEXP show_progressSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type model_ptr(model_ptrSEXP);
    Rcpp::traits::input_parameter< int >::type iterations(iterationsSEXP);
    Rcpp::traits::input_parameter< bool >::type calc_perplexity(calc_perplexitySEXP);
    Rcpp::traits::input_parameter< bool >::type show_progress(show_progressSEXP);
    train_model_rcpp(model_ptr, iterations, calc_perplexity, show_progress);
    return R_NilValue;
END_RCPP
}
// get_topic_word_dist_rcpp
NumericMatrix get_topic_word_dist_rcpp(SEXP model_ptr, int topic, int time);
RcppExport SEXP _DTModelR_get_topic_word_dist_rcpp(SEXP model_ptrSEXP, SEXP topicSEXP, SEXP timeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type model_ptr(model_ptrSEXP);
    Rcpp::traits::input_parameter< int >::type topic(topicSEXP);
    Rcpp::traits::input_parameter< int >::type time(timeSEXP);
    rcpp_result_gen = Rcpp::wrap(get_topic_word_dist_rcpp(model_ptr, topic, time));
    return rcpp_result_gen;
END_RCPP
}
// get_doc_topic_dist_rcpp
NumericMatrix get_doc_topic_dist_rcpp(SEXP model_ptr, int doc_id);
RcppExport SEXP _DTModelR_get_doc_topic_dist_rcpp(SEXP model_ptrSEXP, SEXP doc_idSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type model_ptr(model_ptrSEXP);
    Rcpp::traits::input_parameter< int >::type doc_id(doc_idSEXP);
    rcpp_result_gen = Rcpp::wrap(get_doc_topic_dist_rcpp(model_ptr, doc_id));
    return rcpp_result_gen;
END_RCPP
}
// get_vocabulary_rcpp
CharacterVector get_vocabulary_rcpp(SEXP model_ptr);
RcppExport SEXP _DTModelR_get_vocabulary_rcpp(SEXP model_ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type model_ptr(model_ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(get_vocabulary_rcpp(model_ptr));
    return rcpp_result_gen;
END_RCPP
}
// get_perplexity_rcpp
NumericVector get_perplexity_rcpp(SEXP model_ptr);
RcppExport SEXP _DTModelR_get_perplexity_rcpp(SEXP model_ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type model_ptr(model_ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(get_perplexity_rcpp(model_ptr));
    return rcpp_result_gen;
END_RCPP
}
// get_model_info_rcpp
List get_model_info_rcpp(SEXP model_ptr);
RcppExport SEXP _DTModelR_get_model_info_rcpp(SEXP model_ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type model_ptr(model_ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(get_model_info_rcpp(model_ptr));
    return rcpp_result_gen;
END_RCPP
}
// set_word_prior_rcpp
void set_word_prior_rcpp(SEXP model_ptr, int topic, CharacterVector words, NumericVector priors);
RcppExport SEXP _DTModelR_set_word_prior_rcpp(SEXP model_ptrSEXP, SEXP topicSEXP, SEXP wordsSEXP, SEXP priorsSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type model_ptr(model_ptrSEXP);
    Rcpp::traits::input_parameter< int >::type topic(topicSEXP);
    Rcpp::traits::input_parameter< CharacterVector >::type words(wordsSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type priors(priorsSEXP);
    set_word_prior_rcpp(model_ptr, topic, words, priors);
    return R_NilValue;
END_RCPP
}
// get_top_words_rcpp
List get_top_words_rcpp(SEXP model_ptr, int topic, int time, int top_n);
RcppExport SEXP _DTModelR_get_top_words_rcpp(SEXP model_ptrSEXP, SEXP topicSEXP, SEXP timeSEXP, SEXP top_nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type model_ptr(model_ptrSEXP);
    Rcpp::traits::input_parameter< int >::type topic(topicSEXP);
    Rcpp::traits::input_parameter< int >::type time(timeSEXP);
    Rcpp::traits::input_parameter< int >::type top_n(top_nSEXP);
    rcpp_result_gen = Rcpp::wrap(get_top_words_rcpp(model_ptr, topic, time, top_n));
    return rcpp_result_gen;
END_RCPP
}
// save_model_rcpp
void save_model_rcpp(SEXP model_ptr, std::string filename);
RcppExport SEXP _DTModelR_save_model_rcpp(SEXP model_ptrSEXP, SEXP filenameSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type model_ptr(model_ptrSEXP);
    Rcpp::traits::input_parameter< std::string >::type filename(filenameSEXP);
    save_model_rcpp(model_ptr, filename);
    return R_NilValue;
END_RCPP
}
// load_model_rcpp
SEXP load_model_rcpp(std::string filename);
RcppExport SEXP _DTModelR_load_model_rcpp(SEXP filenameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::string >::type filename(filenameSEXP);
    rcpp_result_gen = Rcpp::wrap(load_model_rcpp(filename));
    return rcpp_result_gen;
END_RCPP
}
// get_num_docs_rcpp
int get_num_docs_rcpp(SEXP model_ptr);
RcppExport SEXP _DTModelR_get_num_docs_rcpp(SEXP model_ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type model_ptr(model_ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(get_num_docs_rcpp(model_ptr));
    return rcpp_result_gen;
END_RCPP
}
// get_num_topics_rcpp
int get_num_topics_rcpp(SEXP model_ptr);
RcppExport SEXP _DTModelR_get_num_topics_rcpp(SEXP model_ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type model_ptr(model_ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(get_num_topics_rcpp(model_ptr));
    return rcpp_result_gen;
END_RCPP
}
// get_num_times_rcpp
int get_num_times_rcpp(SEXP model_ptr);
RcppExport SEXP _DTModelR_get_num_times_rcpp(SEXP model_ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type model_ptr(model_ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(get_num_times_rcpp(model_ptr));
    return rcpp_result_gen;
END_RCPP
}
// get_vocab_size_rcpp
int get_vocab_size_rcpp(SEXP model_ptr);
RcppExport SEXP _DTModelR_get_vocab_size_rcpp(SEXP model_ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type model_ptr(model_ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(get_vocab_size_rcpp(model_ptr));
    return rcpp_result_gen;
END_RCPP
}
// get_alpha_rcpp
NumericVector get_alpha_rcpp(SEXP model_ptr);
RcppExport SEXP _DTModelR_get_alpha_rcpp(SEXP model_ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type model_ptr(model_ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(get_alpha_rcpp(model_ptr));
    return rcpp_result_gen;
END_RCPP
}
// get_beta_rcpp
NumericVector get_beta_rcpp(SEXP model_ptr);
RcppExport SEXP _DTModelR_get_beta_rcpp(SEXP model_ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type model_ptr(model_ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(get_beta_rcpp(model_ptr));
    return rcpp_result_gen;
END_RCPP
}
// get_gamma_rcpp
NumericVector get_gamma_rcpp(SEXP model_ptr);
RcppExport SEXP _DTModelR_get_gamma_rcpp(SEXP model_ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type model_ptr(model_ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(get_gamma_rcpp(model_ptr));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_DTModelR_create_dtm_model_rcpp", (DL_FUNC) &_DTModelR_create_dtm_model_rcpp, 6},
    {"_DTModelR_add_document_rcpp", (DL_FUNC) &_DTModelR_add_document_rcpp, 3},
    {"_DTModelR_train_model_rcpp", (DL_FUNC) &_DTModelR_train_model_rcpp, 4},
    {"_DTModelR_get_topic_word_dist_rcpp", (DL_FUNC) &_DTModelR_get_topic_word_dist_rcpp, 3},
    {"_DTModelR_get_doc_topic_dist_rcpp", (DL_FUNC) &_DTModelR_get_doc_topic_dist_rcpp, 2},
    {"_DTModelR_get_vocabulary_rcpp", (DL_FUNC) &_DTModelR_get_vocabulary_rcpp, 1},
    {"_DTModelR_get_perplexity_rcpp", (DL_FUNC) &_DTModelR_get_perplexity_rcpp, 1},
    {"_DTModelR_get_model_info_rcpp", (DL_FUNC) &_DTModelR_get_model_info_rcpp, 1},
    {"_DTModelR_set_word_prior_rcpp", (DL_FUNC) &_DTModelR_set_word_prior_rcpp, 4},
    {"_DTModelR_get_top_words_rcpp", (DL_FUNC) &_DTModelR_get_top_words_rcpp, 4},
    {"_DTModelR_save_model_rcpp", (DL_FUNC) &_DTModelR_save_model_rcpp, 2},
    {"_DTModelR_load_model_rcpp", (DL_FUNC) &_DTModelR_load_model_rcpp, 1},
    {"_DTModelR_get_num_docs_rcpp", (DL_FUNC) &_DTModelR_get_num_docs_rcpp, 1},
    {"_DTModelR_get_num_topics_rcpp", (DL_FUNC) &_DTModelR_get_num_topics_rcpp, 1},
    {"_DTModelR_get_num_times_rcpp", (DL_FUNC) &_DTModelR_get_num_times_rcpp, 1},
    {"_DTModelR_get_vocab_size_rcpp", (DL_FUNC) &_DTModelR_get_vocab_size_rcpp, 1},
    {"_DTModelR_get_alpha_rcpp", (DL_FUNC) &_DTModelR_get_alpha_rcpp, 1},
    {"_DTModelR_get_beta_rcpp", (DL_FUNC) &_DTModelR_get_beta_rcpp, 1},
    {"_DTModelR_get_gamma_rcpp", (DL_FUNC) &_DTModelR_get_gamma_rcpp, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_DTModelR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
