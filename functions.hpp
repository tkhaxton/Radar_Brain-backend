#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

void parse_time(std::string filename_string, int year, int counter, int time_depth, double *minutes);

void read_from_file(std::string filestring, int asc_header_size, int xsize, int ysize, int counter, int time_bins, float ***data, int first, double *pcellsize, double *plat, double *plon);

void read_from_file_limited(std::string filestring, int asc_header_size, int xsize, int ysize, int x_begin, int x_finish, int y_begin, int y_finish, int counter, int time_bins, float ***data, int first, double *pcellsize, double *plat, double *plon);

void read_from_file_binary_limited(std::string filestring, int xsize, int ysize, int x_begin, int x_finish, int y_begin, int y_finish, int counter, int time_bins, float ***data);

void calculate_averages_and_correlations(float ***data, double *minutes, int t, int time_bins, int xsize, int ysize, double cellsize_lat_km, double cellsize_lon_km, double distance_binwidth, double time_binwidth, double interval_binwidth, int interval_bins, double **time_average, long long int *interval_histogram, double *time_correlation, long long int *time_count, double *distance_correlation, long long int *distance_count, int pixelwidth_x, int pixelwidth_y, int **distance_bin_array, long long int *dBZ_histogram, double dBZ_binwidth, int dBZ_nbins, double *psquareaverage);

void output_averages_and_correlations(std::string pathstring, double **time_average, int nframes, int xsize, int ysize, long long int *interval_histogram, int interval_bins, double interval_binwidth, double *time_correlation, long long int *time_count, double *distance_correlation, long long int *distance_count, int time_bins, int distance_bins, double time_binwidth, double distance_binwidth, long long int *dBZ_histogram, double dBZ_binwidth, int dBZ_nbins, double squareaverage);

void initialize_network_Eulerian(double ****bias, double *****weight, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, int x_width, int y_width, double agree_value, double (*activation_function_inverse)(double), double (*activation_function_inverse_derivative)(double));

void feed_forward_and_backpropagate(float ***data, double min_dBZ, double max_dBZ, int y_start, int y_end, int x_start, int x_end, double *minutes, int counter, int time_depth, int time_depth_past, int time_depth_future, double interval, int y_width, int x_width, int number_total_layers, int *number_neurons, double **activation, double **weighted_input, double ****bias, double *****weight, double **error, double ***cost, double ***cost_persistent, double ****dcost_dbias, double *****dcost_dweight, long long int *pbatch_count, double (*activation_function)(double), double (*activation_function_derivative)(double), double (*cost_function)(double, double), double (*cost_derivative)(double, double));

void feed_forward_validate(float ***data, double min_dBZ, double max_dBZ, int y_start, int y_end, int x_start, int x_end, double *minutes, int counter, int time_depth, int time_depth_past, int time_depth_future, double interval, int y_width, int x_width, int number_total_layers, int *number_neurons, double **activation, double **weighted_input, double ****bias, double *****weight, double ***cost, double ***cost_persistent, long long int *pbatch_count, double (*activation_function)(double), double (*activation_function_derivative)(double), double (*cost_function)(double, double), int epoch, double ***forecast_observation_correlation, double ***forecast_observation_correlation_persistent, double ***forecast_squareaverage, double ***forecast_persistent_squareaverage, double ***observation_squareaverage);

void feed_forward_generalize(float ***data, double min_dBZ, double max_dBZ, int y_start, int y_end, int x_start, int x_end, double *minutes, int counter, int time_depth, int time_depth_past, int time_depth_future, double interval, int y_width, int x_width, int number_total_layers, int *number_neurons, double **activation, double **weighted_input, double **bias_average, double ***weight_average, double ***cost, double ***cost_persistent, long long int *pbatch_count, double (*activation_function)(double), double (*activation_function_derivative)(double), double (*cost_function)(double, double), int epoch, double ***forecast_observation_correlation, double ***forecast_observation_correlation_persistent, double ***forecast_squareaverage, double ***forecast_persistent_squareaverage, double ***observation_squareaverage);

void increment_neural_network(int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ****bias, double *****weight, double ****dcost_dbias, double *****dcost_dweight, double learning_rate, long long int *pbatch_count, long long int *poutput_batch_count);

void output_neural_network(std::string output_directory, int epoch, time_t start_cputime, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ****bias, double *****weight, double ***cost, long long int *pbatch_count, long long int *poverall_batch_count, int *pdo_delete);

void input_neural_network(std::string input_directory, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double **bias, double ***weight);

void output_persistent_cost(std::string output_directory, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ***cost_persistent, long long int batch_count);

void increment_and_output_neural_network(std::string output_directory, int epoch, time_t start_cputime, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ****bias, double *****weight, double ***cost, double ***cost_persistent, double ****dcost_dbias, double *****dcost_dweight, double learning_rate, long long int *pbatch_count, int *pdo_delete);

void output_validation_cost(std::string output_directory, int epoch, time_t start_cputime, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ***cost, double ***cost_persistent, double ***forecast_observation_correlation, double ***forecast_observation_correlation_persistent, double ***forecast_squareaverage, double ***forecast_persistent_squareaverage, double ***observation_squareaverage, long long int *pbatch_count, int *pdo_delete);

void output_validation_cost_training_count(std::string output_directory, int epoch, time_t start_cputime, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ***cost, double ***cost_persistent, double ***forecast_observation_correlation, double ***forecast_observation_correlation_persistent, double ***forecast_squareaverage, double ***forecast_persistent_squareaverage, double ***observation_squareaverage, long long int *pbatch_count, int *pdo_delete, long long int training_batch_count);

void output_generalization_cost(std::string output_directory, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ***cost, double ***cost_persistent, double ***forecast_observation_correlation, double ***forecast_observation_correlation_persistent, double ***forecast_squareaverage, double ***forecast_persistent_squareaverage, double ***observation_squareaverage, long long int *pbatch_count);

double sigmoid_function(double z);

double sigmoid_function_derivative(double z);

double inverse_sigmoid_function(double z);

double inverse_sigmoid_function_derivative(double z);

double quadratic_function(double observation, double forecast);

double quadratic_function_derivative(double observation, double forecast);

double cross_entropy_function(double observation, double forecast);

double cross_entropy_function_derivative(double observation, double forecast);


