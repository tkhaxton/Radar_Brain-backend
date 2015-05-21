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
   
   std::string indir, topdir = "../data/single_radar_station/full_year/", station, output_directory;
   std::string outpath = "../output/learn/";
   int xsize = 300;
   int ysize = 300;
   int do_delete = 1, do_delete_validate = 1;
   
   // min_dBZ and max_dBZ are used to map decibel reflectivity dBZ to the
   // interval (0, 1). min_dBZ and max_dBZ are set beyond dBZ range (0, 75)
   // so the derivative of cross entropy cost will be defined even when the
   // forecast dBZ is 0 or 75.
   
   double min_dBZ = -5;
   double max_dBZ = 80;
   
   // Default parameters

   int x_start = 0;
   int x_end = 300;
   int y_start = 0;
   int y_end = 300;
   int time_depth_past = 1;
   int time_depth_future = 1;
   int x_width = 1;
   int y_width = 1;
   double learning_rate = 0.001;
   int number_hidden_layers = 1;
   int neurons_per_layer = 1;
   double agree_dBZ = 0;
   double interval = 6;
   int validate_skip = 1;
   
   // Command-line options
   
   const struct option longopts[] =
   {
      // Required
      
      {"station", required_argument, 0, 's'},
      {"firstyear", required_argument, 0, 'f'},
      {"lastyear", required_argument, 0, 'l'},
      {"epochs", required_argument, 0, 'e'},
      {"output_directory", required_argument, 0, 'o'},
      {"batch_size", required_argument, 0, 'z'},
      {"batches_per_output", required_argument, 0, 'u'},
      
      // Optional
      
      {"time_depth_past", required_argument, 0, 'p'},
      {"time_depth_future", required_argument, 0, 't'},
      {"x_width", required_argument, 0, 'x'},
      {"y_width", required_argument, 0, 'y'},
      {"neurons_per_layer", required_argument, 0, 'n'},
      {"x_start", required_argument, 0, 'a'},
      {"x_end", required_argument, 0, 'b'},
      {"y_start", required_argument, 0, 'c'},
      {"y_end", required_argument, 0, 'd'},
      {"learning_rate", required_argument, 0, 'r'},
      {"number_hidden_layers", required_argument, 0, 'i'},
      {"firstyear_validate", required_argument, 0, 'v'},
      {"lastyear_validate", required_argument, 0, 'w'},
      {"validate_skip", required_argument, 0, 'k'},
      {NULL,0,0,0},
   };
   char shortopts[] = "s:f:l:e:o:z:u:p:t:x:y:n:a:b:c:d:r:i:v:w:k:";
   int required_options_binary = 1 + 2 + 4 + 8 + 16 + 32 + 64, given_options_binary = 0;
   int index;
   int iarg=0;
   
   // Turn on getopt error message
   
   opterr = 1;
   
   // Parse arguments from command line
   
   int firstyear, lastyear, year, epochs, firstyear_validate = 0, lastyear_validate = 0, batch_size, batches_per_output;
   std::string firstyear_string, lastyear_string, firstyear_validate_string, lastyear_validate_string;
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
         case 'e':
            epochs = atoi(optarg);
            given_options_binary += 8;
            break;
         case 'o':
            output_directory = optarg;
            given_options_binary += 16;
            break;
         case 'z':
            batch_size = atoi(optarg);
            given_options_binary += 32;
            break;
         case 'u':
            batches_per_output = atoi(optarg);
            given_options_binary += 64;
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
            x_start = atoi(optarg);
            break;
         case 'b':
            x_end = atoi(optarg);
            break;
         case 'c':
            y_start = atoi(optarg);
            break;
         case 'd':
            y_end = atoi(optarg);
            break;
         case 'r':
            learning_rate = atof(optarg);
            break;
         case 'i':
            number_hidden_layers = atoi(optarg);
            break;
         case 'v':
            firstyear_validate_string = optarg;
            firstyear_validate = atoi(optarg);
            break;
         case 'w':
            lastyear_validate_string = optarg;
            lastyear_validate = atoi(optarg);
            break;
         case 'k':
            validate_skip = atoi(optarg);
            break;
      }
   }
   if(given_options_binary != required_options_binary){
      std::cerr << "Usage: ./analyze --station <station> --firstyear <firstyear> --lastyear <lastyear> --epochs <epochs> --output_directory <output_directory> --batch_size <batch_size> --batches_per_output <batches_per_output> [optional arguments]\n";
      exit(1);
   }
   if(((x_width - 1)%2 != 0)||((y_width - 1)%2 != 0)){
      std::cerr << "x_width and y_width must be odd\n";
      exit(1);
   }
   
   // Derived parameters
   
   double agree_value = (agree_dBZ - min_dBZ) / (max_dBZ - min_dBZ);
   if(x_start < (x_width - 1) / 2){
      x_start = (x_width - 1) / 2;
   }
   if(x_end > xsize - (x_width - 1) / 2){
      x_end = xsize - (x_width - 1) / 2;
   }
   if(y_start < (y_width - 1) / 2){
      y_start = (y_width - 1) / 2;
   }
   if(y_end > ysize - (y_width - 1) / 2){
      y_end = ysize - (y_width - 1) / 2;
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
   double ****bias = new double***[ysize];
   double *****weight = new double****[ysize];
   double ***cost = new double**[ysize];
   double ***cost_persistent = new double**[ysize];
   double ***cost_validation = new double**[ysize];
   double ***cost_validation_persistent = new double**[ysize];
   double ****dcost_dbias = new double***[ysize];
   double *****dcost_dweight = new double****[ysize];
   double ***forecast_squareaverage = new double**[ysize];
   double ***forecast_persistent_squareaverage = new double**[ysize];
   double ***observation_squareaverage = new double**[ysize];
   double ***forecast_observation_correlation = new double**[ysize];
   double ***forecast_observation_correlation_persistent = new double**[ysize];
   
   // Nest neural network arrays location on the outside because independent
   // neural networks are created for each location.
   
   for(int i = y_start; i < y_end; i ++){
      bias[i] = new double**[xsize];
      weight[i] = new double***[xsize];
      cost[i] = new double*[xsize];
      cost_persistent[i] = new double*[xsize];
      cost_validation[i] = new double*[xsize];
      cost_validation_persistent[i] = new double*[xsize];
      dcost_dbias[i] = new double**[xsize];
      dcost_dweight[i] = new double***[xsize];
      forecast_squareaverage[i] = new double*[xsize];
      forecast_persistent_squareaverage[i] = new double*[xsize];
      observation_squareaverage[i] = new double*[xsize];
      forecast_observation_correlation[i] = new double*[xsize];
      forecast_observation_correlation_persistent[i] = new double*[xsize];
      for(int j = x_start; j < x_end; j ++){
         bias[i][j] = new double*[number_total_layers];
         weight[i][j] = new double**[number_total_layers];
         cost[i][j] = new double[number_neurons[number_total_layers - 1]];
         cost_persistent[i][j] = new double[number_neurons[number_total_layers - 1]];
         cost_validation[i][j] = new double[number_neurons[number_total_layers - 1]];
         cost_validation_persistent[i][j] = new double[number_neurons[number_total_layers - 1]];
         dcost_dbias[i][j] = new double*[number_total_layers];
         dcost_dweight[i][j] = new double**[number_total_layers];
         for(int k = 1; k < number_total_layers; k ++){
            bias[i][j][k] = new double[number_neurons[k]];
            weight[i][j][k] = new double*[number_neurons[k]];
            dcost_dbias[i][j][k] = new double[number_neurons[k]];
            dcost_dweight[i][j][k] = new double*[number_neurons[k]];
            for(int l = 0; l < number_neurons[k]; l ++){
               weight[i][j][k][l] = new double[number_neurons[k - 1]];
               dcost_dweight[i][j][k][l] = new double[number_neurons[k - 1]];
            }
         }
         forecast_squareaverage[i][j] = new double[number_neurons[number_total_layers - 1]];
         forecast_persistent_squareaverage[i][j] = new double[number_neurons[number_total_layers - 1]];
         observation_squareaverage[i][j] = new double[number_neurons[number_total_layers - 1]];
         forecast_observation_correlation[i][j] = new double[number_neurons[number_total_layers - 1]];
         forecast_observation_correlation_persistent[i][j] = new double[number_neurons[number_total_layers - 1]];
      }
   }
   
   // Temporary arrays for repeated use inside feed_forward_and_backpropagate
   
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
   
   // Initialize neural network
   
   initialize_network_Eulerian(bias, weight, y_start, y_end, x_start, x_end, number_total_layers, number_neurons, x_width, y_width, agree_value, inverse_sigmoid_function, inverse_sigmoid_function_derivative);
   time_t start_cputime;
   time(&start_cputime);
   
   // Additional variables
      
   std::string slashstring = "/", yearstring, filename_string;
   int counter;
   double lat, lon, cellsize;
   long long int batch_count = 0, output_batch_count = 0, overall_batch_count = 0, validation_batch_count = 0;

   for(int epoch = 0; epoch < epochs; epoch ++){
      
      // Loop through directories for range of years in training set

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
                  
                  read_from_file_binary_limited(p.string(), xsize, ysize, x_start - (x_width - 1) / 2, x_end + (x_width - 1) / 2, y_start - (y_width - 1) / 2, y_end + (y_width - 1) / 2, counter % time_depth, time_depth, data);
                  
                  // Feed forward through neural network and backpropagate to
                  // calculate the derivative of the cross entropy cost function
                  // with respect to the weights and biases.
                  
                  feed_forward_and_backpropagate(data, min_dBZ, max_dBZ, y_start, y_end, x_start, x_end, minutes, counter, time_depth, time_depth_past, time_depth_future, interval, y_width, x_width, number_total_layers, number_neurons, activation, weighted_input, bias, weight, error, cost, cost_persistent, dcost_dbias, dcost_dweight, &batch_count, sigmoid_function, sigmoid_function_derivative, quadratic_function, cross_entropy_function_derivative);
                  
                  if(batch_count >= batch_size){
                     
                     // Increment neural network
                     
                     increment_neural_network(y_start, y_end, x_start, x_end, number_total_layers, number_neurons, bias, weight, dcost_dbias, dcost_dweight, learning_rate, &batch_count, &output_batch_count);
                     if(output_batch_count >= batch_size * batches_per_output){
                        
                        // Output neural network biases and weights
                        // and average quadratic costs for training set
                        
                        output_neural_network(output_directory, epoch, start_cputime, y_start, y_end, x_start, x_end, number_total_layers, number_neurons, bias, weight, cost, &output_batch_count, &overall_batch_count, &do_delete);
                        do_delete = 0;
                     }
                  }
                  counter ++;
               }
            }
         }
      }
      if(epoch == 0){
         
         // Output average quadratic costs and forecast-observation correlations
         // for the training set assuming Eulerian persistence (static radar
         // pattern)
         
         output_persistent_cost(output_directory, y_start, y_end, x_start, x_end, number_total_layers, number_neurons, cost_persistent, overall_batch_count + output_batch_count + batch_count);
      }
      if((firstyear_validate > 0)&&(epoch % validate_skip == 0)){
         
         // Loop through directories for range of years in validation set
         
         counter = 0;
         for(int year = firstyear_validate; year <= lastyear_validate; year++){
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
                     
                     read_from_file_binary_limited(p.string(), xsize, ysize, x_start - (x_width - 1) / 2, x_end + (x_width - 1) / 2, y_start - (y_width - 1) / 2, y_end + (y_width - 1) / 2, counter % time_depth, time_depth, data);
                     
                     // Calculate cost
                     
                     feed_forward_validate(data, min_dBZ, max_dBZ, y_start, y_end, x_start, x_end, minutes, counter, time_depth, time_depth_past, time_depth_future, interval, y_width, x_width, number_total_layers, number_neurons, activation, weighted_input, bias, weight, cost_validation, cost_validation_persistent, &validation_batch_count, sigmoid_function, sigmoid_function_derivative, quadratic_function, epoch, forecast_observation_correlation, forecast_observation_correlation_persistent, forecast_squareaverage, forecast_persistent_squareaverage, observation_squareaverage);
                     counter ++;
                  }
               }
            }
         }
      }

      // Output average quadratic costs and forecast-observation correlations
      // for the validation set, obtained both from the neural network and
      // (for comparison) from assuming Eulerian persistence

      output_validation_cost_training_count(output_directory, epoch, start_cputime, y_start, y_end, x_start, x_end, number_total_layers, number_neurons, cost_validation, cost_validation_persistent, forecast_observation_correlation, forecast_observation_correlation_persistent, forecast_squareaverage, forecast_persistent_squareaverage, observation_squareaverage, &validation_batch_count, &do_delete_validate, overall_batch_count + output_batch_count + batch_count);
      do_delete_validate = 0;
   }
   
   // Delete arrays

   for(int t = 0; t < time_depth; t ++){
      for(int j = 0; j < ysize; j ++){
         delete[] data[t][j];
      }
      delete[] data[t];
   }
   delete[] data;
   delete[] minutes;
   for(int i = y_start; i < y_end; i ++){
      for(int j = x_start; j < x_end; j ++){
         for(int k = 1; k < number_total_layers; k ++){
            delete[] bias[i][j][k];
            delete[] dcost_dbias[i][j][k];
            for(int l = 0; l < number_neurons[k]; l ++){
               delete[] weight[i][j][k][l];
               delete[] dcost_dweight[i][j][k][l];
            }
            delete[] weight[i][j][k];
            delete[] dcost_dweight[i][j][k];
         }
         delete[] bias[i][j];
         delete[] weight[i][j];
         delete[] cost[i][j];
         delete[] cost_persistent[i][j];
         delete[] cost_validation[i][j];
         delete[] dcost_dbias[i][j];
         delete[] dcost_dweight[i][j];
         delete[] forecast_squareaverage[i][j];
         delete[] forecast_persistent_squareaverage[i][j];
         delete[] observation_squareaverage[i][j];
         delete[] forecast_observation_correlation[i][j];
         delete[] forecast_observation_correlation_persistent[i][j];
      }
      delete[] bias[i];
      delete[] weight[i];
      delete[] cost_persistent[i];
      delete[] cost_validation[i];
      delete[] dcost_dbias[i];
      delete[] dcost_dweight[i];
      delete[] forecast_squareaverage[i];
      delete[] forecast_persistent_squareaverage[i];
      delete[] observation_squareaverage[i];
      delete[] forecast_observation_correlation[i];
      delete[] forecast_observation_correlation_persistent[i];
   }
   delete[] bias;
   delete[] weight;
   delete[] number_neurons;
   delete[] cost;
   delete[] cost_persistent;
   delete[] cost_validation;
   delete[] dcost_dbias;
   delete[] dcost_dweight;
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