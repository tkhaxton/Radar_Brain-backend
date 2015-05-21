#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

// Parses the date and time in the radar data filename,
// converts it to the number of minutes since 12am, Jan 1, 1970,
// and enters the minutes into an array indexed by the file index.
void parse_time(std::string filename_string, int year, int counter, int time_depth, double *minutes);

// Reads a rectangular block of decibel reflectivity (dBZ) data with pixel range
// (x_start <= x < x_finish, y_start <= y < y_finish)
// from a data file with pixel range (0 <= x < xsize, 0 <= y < ysize).
// Enters the dBZ data into a 3d array indexed by the file index, x, and y.
void read_from_file_binary_limited(std::string filestring, int xsize, int ysize, int x_begin, int x_finish, int y_begin, int y_finish, int counter, int time_bins, float ***data);

// Creates a chain of neurons leading from input to output that enforces
// out_i(in) = in and d(out_in)/d(x)|_{x = in} = 1
// for in = most recent radar data at the central location
// and {out_i} = the set of all output radar predictions.
//
// Because the activation function is nonlinear, these requirements
// cannot be enforced in general. Instead, this function enforces them for a
// particular value of the decibel reflectivity (dBZ = agree_value).
void initialize_network_Eulerian(double ****bias, double *****weight, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, int x_width, int y_width, double agree_value, double (*activation_function_inverse)(double), double (*activation_function_inverse_derivative)(double));

// Feeds inputs forward through the neural network and backpropagates the error
// to calculate the derivative of the cross entropy cost function with respect
// to the weights and biases.  Calculates the average quadratic cost for the
// outputs.
void feed_forward_and_backpropagate(float ***data, double min_dBZ, double max_dBZ, int y_start, int y_end, int x_start, int x_end, double *minutes, int counter, int time_depth, int time_depth_past, int time_depth_future, double interval, int y_width, int x_width, int number_total_layers, int *number_neurons, double **activation, double **weighted_input, double ****bias, double *****weight, double **error, double ***cost, double ***cost_persistent, double ****dcost_dbias, double *****dcost_dweight, long long int *pbatch_count, double (*activation_function)(double), double (*activation_function_derivative)(double), double (*cost_function)(double, double), double (*cost_derivative)(double, double));

// Feeds inputs forward through the neural network, calculating the quadratic
// costs and forecast-observation correlations using both the fed-forward
// outputs and the outpust assuming Eulerian persistence
void feed_forward_validate(float ***data, double min_dBZ, double max_dBZ, int y_start, int y_end, int x_start, int x_end, double *minutes, int counter, int time_depth, int time_depth_past, int time_depth_future, double interval, int y_width, int x_width, int number_total_layers, int *number_neurons, double **activation, double **weighted_input, double ****bias, double *****weight, double ***cost, double ***cost_persistent, long long int *pbatch_count, double (*activation_function)(double), double (*activation_function_derivative)(double), double (*cost_function)(double, double), int epoch, double ***forecast_observation_correlation, double ***forecast_observation_correlation_persistent, double ***forecast_squareaverage, double ***forecast_persistent_squareaverage, double ***observation_squareaverage);

// Feeds inputs (for possibly many forecast locations) forward through a single
// (average) neural network, calculating the quadratic costs and forecast-
// observation correlations using both the fed-forward outputs and the outpust
// assuming Eulerian persistence.
void feed_forward_generalize(float ***data, double min_dBZ, double max_dBZ, int y_start, int y_end, int x_start, int x_end, double *minutes, int counter, int time_depth, int time_depth_past, int time_depth_future, double interval, int y_width, int x_width, int number_total_layers, int *number_neurons, double **activation, double **weighted_input, double **bias_average, double ***weight_average, double ***cost, double ***cost_persistent, long long int *pbatch_count, double (*activation_function)(double), double (*activation_function_derivative)(double), double (*cost_function)(double, double), int epoch, double ***forecast_observation_correlation, double ***forecast_observation_correlation_persistent, double ***forecast_squareaverage, double ***forecast_persistent_squareaverage, double ***observation_squareaverage);

// Increments the biases and weights of the neural network in proportion to the
// learning rate and the calculated average derivatives of the cost function
// with respect to the biases and weights.
void increment_neural_network(int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ****bias, double *****weight, double ****dcost_dbias, double *****dcost_dweight, double learning_rate, long long int *pbatch_count, long long int *poutput_batch_count);

// Outputs the neural network biases, weights, and average costs.
void output_neural_network(std::string output_directory, int epoch, time_t start_cputime, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ****bias, double *****weight, double ***cost, long long int *pbatch_count, long long int *poverall_batch_count, int *pdo_delete);

// Inputs the neural network biases and weights.
void input_neural_network(std::string input_directory, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double **bias, double ***weight);

// Outputs the average costs assuming Eulerian persistence.
void output_persistent_cost(std::string output_directory, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ***cost_persistent, long long int batch_count);

// Outputs the average costs and forecast-observation correlations for the
// validation set, both from the neural network and from assuming Eulerian
// persistence
void output_validation_cost_training_count(std::string output_directory, int epoch, time_t start_cputime, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ***cost, double ***cost_persistent, double ***forecast_observation_correlation, double ***forecast_observation_correlation_persistent, double ***forecast_squareaverage, double ***forecast_persistent_squareaverage, double ***observation_squareaverage, long long int *pbatch_count, int *pdo_delete, long long int training_batch_count);

// Outputs the average costs and forecast-observation correlations for
// the generalization set, both from the neural network and  from assuming
// Eulerian persistence
void output_generalization_cost(std::string output_directory, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ***cost, double ***cost_persistent, double ***forecast_observation_correlation, double ***forecast_observation_correlation_persistent, double ***forecast_squareaverage, double ***forecast_persistent_squareaverage, double ***observation_squareaverage, long long int *pbatch_count);

// Returns 1/(1+exp(-z)
double sigmoid_function(double z);

// Returns exp(-z)/pow(1+exp(-z), 2)
double sigmoid_function_derivative(double z);

// Returns -ln(1/z-1)
double inverse_sigmoid_function(double z);

// Returns 1/(z+z*z)
double inverse_sigmoid_function_derivative(double z);

// Returns pow(forecast - observation, 2)/2
double quadratic_function(double observation, double forecast);

// Returns forecast - observation
double quadratic_function_derivative(double observation, double forecast);

// Returns -observation * ln(forecast) + (observation - 1) * ln(1 - forecast)
double cross_entropy_function(double observation, double forecast);

// Returns -observation / forecast + (1 - observation) / (1 - forecast)
double cross_entropy_function_derivative(double observation, double forecast);


