#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>
#include <getopt.h>
#include <math.h>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include "functions.hpp"

// Definitions for getopt_long

#define no_argument 0
#define required_argument 1
#define optional_argument 2

int main(int argc, char * argv[])
{
   // Static parameters
   
   std::string indir, topdir = "../data/single_radar_station/full_year/", station, input_directory, output_directory;
   std::string outpath = "../output/generalize/";
   int xsize = 300;
   int ysize = 300;
      
   // min_dBZ and max_dBZ are used to map decibel reflectivity dBZ to the
   // interval (0, 1). min_dBZ and max_dBZ are set beyond dBZ range (0, 75)
   // so the derivative of cross entropy cost will be defined even when the
   // forecast dBZ is 0 or 75.
   
   double min_dBZ = -5;
   double max_dBZ = 80;
   
   // Default parameters

   int x_start_train = 0;
   int x_end_train = 300;
   int y_start_train = 0;
   int y_end_train = 300;
   int x_start_test = 0;
   int x_end_test = 300;
   int y_start_test = 0;
   int y_end_test = 300;
   int time_depth_past = 1;
   int time_depth_future = 1;
   int x_width = 1;
   int y_width = 1;
   int number_hidden_layers = 1;
   int neurons_per_layer = 1;
   double agree_dBZ = 0;
   double interval = 6;
   
   // Command-line options
   
   const struct option longopts[] =
   {
      // Required
      
      {"station", required_argument, 0, 's'},
      {"firstyear", required_argument, 0, 'f'},
      {"lastyear", required_argument, 0, 'l'},
      {"input_directory", required_argument, 0, 'i'},
      {"output_directory", required_argument, 0, 'o'},
      
      // Optional
      
      {"time_depth_past", required_argument, 0, 'p'},
      {"time_depth_future", required_argument, 0, 't'},
      {"x_width", required_argument, 0, 'x'},
      {"y_width", required_argument, 0, 'y'},
      {"neurons_per_layer", required_argument, 0, 'n'},
      {"x_start_train", required_argument, 0, 'a'},
      {"x_end_train", required_argument, 0, 'b'},
      {"y_start_train", required_argument, 0, 'c'},
      {"y_end_train", required_argument, 0, 'd'},
      {"x_start_test", required_argument, 0, 'e'},
      {"x_end_test", required_argument, 0, 'g'},
      {"y_start_test", required_argument, 0, 'j'},
      {"y_end_test", required_argument, 0, 'k'},
      {"number_hidden_layers", required_argument, 0, 'h'},
      {NULL,0,0,0},
   };
   char shortopts[] = "s:f:l:i:o:p:t:x:y:n:a:b:c:d:e:g:j:k:h";
   int required_options_binary = 1 + 2 + 4 + 8 + 16, given_options_binary = 0;
   int index;
   int iarg=0;
   
   // Turn on getopt error message
   
   opterr = 1;
   
   // Parse arguments from command line
   
   int firstyear, lastyear, year;
   std::string firstyear_string, lastyear_string;
   while(iarg != -1)
   {
      iarg = getopt_long(argc, argv, shortopts, longopts, &index);
      switch (iarg)
      {
         case 's':
            station = optarg;
            given_options_binary ++;
            break;
         case 'f':
            firstyear_string = optarg;
            firstyear = atoi(optarg);
            given_options_binary += 2;
            break;
         case 'l':
            lastyear_string = optarg;
            lastyear = atoi(optarg);
            given_options_binary += 4;
            break;
         case 'i':
            input_directory = optarg;
            given_options_binary += 8;
            break;
         case 'o':
            output_directory = optarg;
            given_options_binary += 16;
            break;
         case 'p':
            time_depth_past = atoi(optarg);
            break;
         case 't':
            time_depth_future = atoi(optarg);
            break;
         case 'x':
            x_width = atoi(optarg);
            break;
         case 'y':
            y_width = atoi(optarg);
            break;
         case 'n':
            neurons_per_layer = atoi(optarg);
            break;
         case 'a':
            x_start_train = atoi(optarg);
            break;
         case 'b':
            x_end_train = atoi(optarg);
            break;
         case 'c':
            y_start_train = atoi(optarg);
            break;
         case 'd':
            y_end_train = atoi(optarg);
            break;
         case 'e':
            x_start_test = atoi(optarg);
            break;
         case 'g':
            x_end_test = atoi(optarg);
            break;
         case 'j':
            y_start_test = atoi(optarg);
            break;
         case 'k':
            y_end_test = atoi(optarg);
            break;
         case 'h':
            number_hidden_layers = atoi(optarg);
            break;
      }
   }
   if(given_options_binary != required_options_binary){
      std::cerr << "Usage: ./analyze --station <station> --firstyear <firstyear> --lastyear <lastyear> --input_directory <input_directory> --output_directory <output_directory> [optional arguments]\n";
      exit(1);
   }
   if(((x_width - 1)%2 != 0)||((y_width - 1)%2 != 0)){
      std::cerr << "x_width and y_width must be odd\n";
      exit(1);
   }
   
   // Derived parameters
   
   double agree_value = (agree_dBZ - min_dBZ) / (max_dBZ - min_dBZ);
   if(x_start_train < (x_width - 1) / 2){
      x_start_train = (x_width - 1) / 2;
   }
   if(x_end_train > xsize - (x_width - 1) / 2){
      x_end_train = xsize - (x_width - 1) / 2;
   }
   if(y_start_train < (y_width - 1) / 2){
      y_start_train = (y_width - 1) / 2;
   }
   if(y_end_train > ysize - (y_width - 1) / 2){
      y_end_train = ysize - (y_width - 1) / 2;
   }
   if(x_start_test < (x_width - 1) / 2){
      x_start_test = (x_width - 1) / 2;
   }
   if(x_end_test > xsize - (x_width - 1) / 2){
      x_end_test = xsize - (x_width - 1) / 2;
   }
   if(y_start_test < (y_width - 1) / 2){
      y_start_test = (y_width - 1) / 2;
   }
   if(y_end_test > ysize - (y_width - 1) / 2){
      y_end_test = ysize - (y_width - 1) / 2;
   }
   int time_depth = time_depth_past + time_depth_future;
   
   // Create data arrays
   
   float ***data = new float**[time_depth];
   for(int t = 0; t < time_depth; t ++){
      data[t] = new float*[ysize];
      for(int j = 0; j < ysize; j++){
         data[t][j] = new float[xsize];
      }
   }
   double *minutes = new double[time_depth];
   
   // Create neural network
   
   int number_total_layers = number_hidden_layers + 2;
   int *number_neurons = new int[number_total_layers];
   number_neurons[0] = time_depth_past * x_width * y_width;
   for(int i = 1; i < number_total_layers - 1; i ++){
      number_neurons[i] = neurons_per_layer;
   }
   number_neurons[number_total_layers - 1] = time_depth_future;   
   double **bias_average = new double*[number_total_layers];
   double ***weight_average = new double**[number_total_layers];
   double ***cost = new double**[ysize];
   double ***cost_persistent = new double**[ysize];
   double ***forecast_squareaverage = new double**[ysize];
   double ***forecast_persistent_squareaverage = new double**[ysize];
   double ***observation_squareaverage = new double**[ysize];
   double ***forecast_observation_correlation = new double**[ysize];
   double ***forecast_observation_correlation_persistent = new double**[ysize];
   
   for(int k = 1; k < number_total_layers; k ++){
      bias_average[k] = new double[number_neurons[k]];
      weight_average[k] = new double*[number_neurons[k]];
      for(int l = 0; l < number_neurons[k]; l ++){
         weight_average[k][l] = new double[number_neurons[k - 1]];
      }
   }
   
   // Nest neural network arrays location on the outside because independent
   // neural networks are created for each location.
   
   for(int i = y_start_test; i < y_end_test; i ++){
      cost[i] = new double*[xsize];
      cost_persistent[i] = new double*[xsize];
      forecast_squareaverage[i] = new double*[xsize];
      forecast_persistent_squareaverage[i] = new double*[xsize];
      observation_squareaverage[i] = new double*[xsize];
      forecast_observation_correlation[i] = new double*[xsize];
      forecast_observation_correlation_persistent[i] = new double*[xsize];
      for(int j = x_start_test; j < x_end_test; j ++){
         cost[i][j] = new double[number_neurons[number_total_layers - 1]];
         cost_persistent[i][j] = new double[number_neurons[number_total_layers - 1]];
         forecast_squareaverage[i][j] = new double[number_neurons[number_total_layers - 1]];
         forecast_persistent_squareaverage[i][j] = new double[number_neurons[number_total_layers - 1]];
         observation_squareaverage[i][j] = new double[number_neurons[number_total_layers - 1]];
         forecast_observation_correlation[i][j] = new double[number_neurons[number_total_layers - 1]];
         forecast_observation_correlation_persistent[i][j] = new double[number_neurons[number_total_layers - 1]];
      }
   }
   
   // Temporary arrays for repeated use inside feed_forward
   
   double **activation = new double*[number_total_layers];
   double **weighted_input = new double*[number_total_layers];
   double **error = new double*[number_total_layers];
   for(int k = 1; k < number_total_layers; k ++){
      weighted_input[k] = new double[number_neurons[k]];
      error[k] = new double[number_neurons[k]];
   }
   for(int k = 0; k < number_total_layers; k ++){
      activation[k] = new double[number_neurons[k]];
   }
   
   // Input neural network
   
   input_neural_network(input_directory, y_start_train, y_end_train, x_start_train, x_end_train, number_total_layers, number_neurons, bias_average, weight_average);
      
   // Additional variables
   
   long long int batch_count = 0;
   std::string slashstring = "/", yearstring, filename_string;
   int counter, first = 1;

   // Loop through directories for range of years in generalization set
         
   counter = 0;
   for(int year = firstyear; year <= lastyear; year++){
      std::stringstream ss;
      ss << year;
      indir = topdir + station + slashstring + ss.str() + "/binary_esri";
      
      // Iterate over files using Boost
      
      boost::filesystem::path targetDir(indir);
      boost::filesystem::directory_iterator it(targetDir), eod;
      BOOST_FOREACH(boost::filesystem::path const &p, std::make_pair(it, eod))
      {
         if(is_regular_file(p))
         {
            if(extension(p) == ".asc")
            {
               
               // Parse date and time from filename
               
               parse_time(p.filename().string(), year, counter, time_depth, minutes);
               
               // Read radar data prepared in uniform binary format
               
               read_from_file_binary_limited(p.string(), xsize, ysize, x_start_test - (x_width - 1) / 2, x_end_test + (x_width - 1) / 2, y_start_test - (y_width - 1) / 2, y_end_test + (y_width - 1) / 2, counter % time_depth, time_depth, data);
               
               // Feed forward through the neural network and calculate the
               // average quadratic cost and forecast-observation correlation
               // for both the neural network forecast and the Eulerian
               // persistence forecast
               
               feed_forward_generalize(data, min_dBZ, max_dBZ, y_start_test, y_end_test, x_start_test, x_end_test, minutes, counter, time_depth, time_depth_past, time_depth_future, interval, y_width, x_width, number_total_layers, number_neurons, activation, weighted_input, bias_average, weight_average, cost, cost_persistent, &batch_count, sigmoid_function, sigmoid_function_derivative, quadratic_function, 0, forecast_observation_correlation, forecast_observation_correlation_persistent, forecast_squareaverage, forecast_persistent_squareaverage, observation_squareaverage);
               counter ++;
            }
         }
      }
   }
   
   // Output average quadratic costs and forecast-observation correlations for
   // the generalization set, obtained both from the neural network and (for
   // comparison) from assuming Eulerian persistence (static radar pattern)
   
   output_generalization_cost(output_directory, y_start_test, y_end_test, x_start_test, x_end_test, number_total_layers, number_neurons, cost, cost_persistent, forecast_observation_correlation, forecast_observation_correlation_persistent, forecast_squareaverage, forecast_persistent_squareaverage, observation_squareaverage, &batch_count);
   
   // Delete arrays

   for(int t = 0; t < time_depth; t ++){
      for(int j = 0; j < ysize; j ++){
         delete[] data[t][j];
      }
      delete[] data[t];
   }
   delete[] data;
   delete[] minutes;
   for(int k = 1; k < number_total_layers; k ++){
      delete[] bias_average[k];
      for(int l = 0; l < number_neurons[k]; l ++){
         delete[] weight_average[k][l];
      }
      delete[] weight_average[k];
   }
   delete[] bias_average;
   delete[] weight_average;
   for(int i = y_start_test; i < y_end_test; i ++){
      for(int j = x_start_test; j < x_end_test; j ++){
         delete[] cost[i][j];
         delete[] cost_persistent[i][j];
         delete[] forecast_squareaverage[i][j];
         delete[] forecast_persistent_squareaverage[i][j];
         delete[] observation_squareaverage[i][j];
         delete[] forecast_observation_correlation[i][j];
         delete[] forecast_observation_correlation_persistent[i][j];
      }
      delete[] cost_persistent[i];
      delete[] forecast_squareaverage[i];
      delete[] forecast_persistent_squareaverage[i];
      delete[] observation_squareaverage[i];
      delete[] forecast_observation_correlation[i];
      delete[] forecast_observation_correlation_persistent[i];
   }
   delete[] number_neurons;
   delete[] cost;
   delete[] cost_persistent;
   delete[] forecast_squareaverage;
   delete[] forecast_persistent_squareaverage;
   delete[] observation_squareaverage;
   delete[] forecast_observation_correlation;
   delete[] forecast_observation_correlation_persistent;
   
   for(int k = 1; k < number_total_layers; k ++){
      delete[] weighted_input[k];
      delete[] error[k];
   }
   for(int k = 0; k < number_total_layers; k ++){
      delete[] activation[k];
   }
   delete[] weighted_input;
   delete[] error;
   delete[] activation;
}