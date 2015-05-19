#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <math.h>
#include "functions.hpp"

void parse_time(std::string filename_string, int year, int counter, int time_depth, double *minutes)
{
   int month;
   struct tm timetm;
   time_t timet;
   timetm.tm_year = year;
   std::stringstream(filename_string.substr(13, 2)) >> month;

   // Convert months to months since Jan (struct tm format)

   month --;
   timetm.tm_mon = month;
   std::stringstream(filename_string.substr(15, 2)) >> timetm.tm_mday;
   std::stringstream(filename_string.substr(18, 2)) >> timetm.tm_hour;
   std::stringstream(filename_string.substr(20, 2)) >> timetm.tm_min;
   std::stringstream(filename_string.substr(22, 2)) >> timetm.tm_sec;
   
   //  Set daylight savings flag to zero; otherwise it can be initialized randomly
   
   timetm.tm_isdst = 0;
   
   // Convert to minutes
   
   timet = std::mktime(&(timetm));
   minutes[counter % time_depth] = (double) (timet / 60.);
}

void read_from_file(std::string filestring, int asc_header_size, int xsize, int ysize, int counter, int time_bins, float ***data, int first, double *pcellsize, double *plat, double *plon)
{
   
   // Read from file

   int linecounter = 0, valcounter, i, j, parse_xsize, parse_ysize;
   std::string line;
   std::string tmp;
   std::ifstream infile;
   double val;
   infile.open(filestring.c_str(),std::ios::in);
   while (std::getline(infile, line))
   {
      std::istringstream iss(line);
      if((first == 1) && (linecounter < asc_header_size)){
         
         // Parse header
         
         if(linecounter == 0){
            std::stringstream(line.substr(6, 3)) >> parse_xsize;
            if(parse_xsize != xsize){
               std::cout << "xsize " << parse_xsize << " in " << infile << ", expecting " << xsize << std::endl;
               exit(1);
            }
         }
         else if(linecounter == 1){
            std::stringstream(line.substr(6, 3)) >> parse_ysize;
            if(parse_ysize != ysize){
               std::cout << "ysize " << parse_ysize << " in " << infile << ", expecting " << ysize << std::endl;
               exit(1);
            }
         }
         else if(linecounter == 2){
            std::stringstream(line.substr(10, 8)) >> *plon;
         }
         else if(linecounter == 3){
            std::stringstream(line.substr(10, 8)) >> *plat;
         }
         else if(linecounter == 4){
            std::stringstream(line.substr(9, 8)) >> *pcellsize;
            
            // Adjust lat, lon from lower left corner to center
            
            (*plon) += ((*pcellsize) * 0.5 * xsize);
            (*plat) += ((*pcellsize) * 0.5 * ysize);
         }
      }      
      if(linecounter >= asc_header_size){
         if((linecounter == asc_header_size)&&(line == "E")){

            // E indicates radar below threshold for whole file

            for(j = 0; j < ysize; j ++){
               for(i = 0; i < xsize; i ++){
                  data[counter % time_bins][j][i] = 0;
               }
            }
         }
         else if(line == "N"){
            
            // N indicates radar below threshold for whole row
            
            for(i = 0; i < xsize; i ++){
               data[counter % time_bins][linecounter - asc_header_size][i] = 0;
            }
         }
         else{
            
            // Parse string
            
            valcounter = 0;
            while(iss >> tmp){
               
               // n indicates below threshold
               
               if(tmp == "n") val = 0;
               else{
                  std::istringstream i(tmp);
                  i >> val;
               }
               data[counter % time_bins][linecounter - asc_header_size][valcounter] = val;
               valcounter++;
            }
         }
      }
      linecounter ++;
   }
   infile.close();
}

void read_from_file_limited(std::string filestring, int asc_header_size, int xsize, int ysize, int x_begin, int x_finish, int y_begin, int y_finish, int counter, int time_bins, float ***data, int first, double *pcellsize, double *plat, double *plon)
{
   
   // Read from file
   
   int linecounter = 0, valcounter, i, j, parse_xsize, parse_ysize;
   std::string line;
   std::string tmp;
   std::ifstream infile;
   double val;
   infile.open(filestring.c_str(),std::ios::in);
   while (std::getline(infile, line))
   {
      std::istringstream iss(line);
      if((first == 1) && (linecounter < asc_header_size)){
         
         // Parse header
         
         if(linecounter == 0){
            std::stringstream(line.substr(6, 3)) >> parse_xsize;
            if(parse_xsize != xsize){
               std::cout << "xsize " << parse_xsize << " in " << infile << ", expecting " << xsize << std::endl;
               exit(1);
            }
         }
         else if(linecounter == 1){
            std::stringstream(line.substr(6, 3)) >> parse_ysize;
            if(parse_ysize != ysize){
               std::cout << "ysize " << parse_ysize << " in " << infile << ", expecting " << ysize << std::endl;
               exit(1);
            }
         }
         else if(linecounter == 2){
            std::stringstream(line.substr(10, 8)) >> *plon;
         }
         else if(linecounter == 3){
            std::stringstream(line.substr(10, 8)) >> *plat;
         }
         else if(linecounter == 4){
            std::stringstream(line.substr(9, 8)) >> *pcellsize;
            
            // Adjust lat, lon from lower left corner to center
            
            (*plon) += ((*pcellsize) * 0.5 * xsize);
            (*plat) += ((*pcellsize) * 0.5 * ysize);
         }
      }
      else if((linecounter == asc_header_size)&&(line == "E")){
         
         // E indicates radar below threshold for whole file
         
         for(j = y_begin; j < y_finish; j ++){
            for(i = x_begin; i < x_finish; i ++){
               data[counter % time_bins][j][i] = 0;
            }
         }
      }
      else if((linecounter >= asc_header_size + y_begin)&&(linecounter < asc_header_size + y_finish)){
         if(line == "N"){
            
            // N indicates radar below threshold for whole row
            
            for(i = x_begin; i < x_finish; i ++){
               data[counter % time_bins][linecounter - asc_header_size][i] = 0;
            }
         }
         else{
            
            // Parse string
            
            valcounter = 0;
            while(iss >> tmp){
               
               // n indicates below threshold
               
               if(tmp == "n") val = 0;
               else{
                  std::istringstream i(tmp);
                  i >> val;
               }
               data[counter % time_bins][linecounter - asc_header_size][valcounter] = val;
               valcounter++;
            }
         }
      }
      linecounter ++;
   }
   infile.close();
   /*for(j = y_begin; j < y_finish; j ++){
      for(i = x_begin; i < x_finish; i ++){
         std::cout << data[counter % time_bins][j][i] << " ";
      }
      std::cout << "\n";
   }*/
   //exit(1);
}

void read_from_file_binary_limited(std::string filestring, int xsize, int ysize, int x_begin, int x_finish, int y_begin, int y_finish, int counter, int time_bins, float ***data)
{
   std::ifstream infile;
   infile.open(filestring.c_str(),std::ios::in);
   int xbytes = 4 * (x_finish - x_begin), i, j;
   //std::cout << infile.tellg() << "\n";
   for(j = y_begin; j < y_finish; j ++){
      infile.seekg(4 * (j * xsize + x_begin));
      //std::cout << infile.tellg() << " " << xbytes << "\n";
      infile.read((char *) &(data[counter % time_bins][j][x_begin]), xbytes);
   }
   //std::cout << y_begin << " " << y_finish << " " << x_begin << " " << x_finish << "\n";
   //std::cout << "\n";
   infile.close();
   /*for(j = y_begin; j < y_finish; j ++){
      for(i = x_begin; i < x_finish; i ++){
         std::cout << data[counter % time_bins][j][i] << " ";
      }
      std::cout << "\n";
   }
   exit(1);*/
}

void calculate_averages_and_correlations(float ***data, double *minutes, int t, int time_bins, int xsize, int ysize, double cellsize_lat_km, double cellsize_lon_km, double distance_binwidth, double time_binwidth, double interval_binwidth, int interval_bins, double **time_average, long long int *interval_histogram, double *time_correlation, long long int *time_count, double *distance_correlation, long long int *distance_count, int pixelwidth_x, int pixelwidth_y, int **distance_bin_array, long long int *dBZ_histogram, double dBZ_binwidth, int dBZ_bins, double *psquareaverage)
{
   int i, j, dt, di, dj, time_bins_having_data, now_index, before_index, time_bin, distance_bin, dBZ_bin;
   if(t + 1 < time_bins) time_bins_having_data = t + 1;
   else time_bins_having_data = time_bins;
   double distance, minutes_difference;
   time_t time_difference;   
   for(j = 0; j < ysize; j ++ ){
      for(i = 0; i < xsize; i ++ ){
         //for(j = 10; j < 11; j ++ ){
            //for(i = 10; i < 11; i ++ ){
         now_index = t % time_bins;
         time_average[j][i] += data[now_index][j][i];
         dBZ_bin = (int) (data[now_index][j][i]/dBZ_binwidth + 0.5);
         if(dBZ_bin < dBZ_bins){
            dBZ_histogram[dBZ_bin] ++;
         }
         *psquareaverage += pow(data[now_index][j][i], 2);
         for(dt = 0; dt < time_bins_having_data; dt ++){
            before_index = (t - dt) % time_bins;
            minutes_difference = minutes[now_index] - minutes[before_index];
            time_bin = (int) (minutes_difference / time_binwidth + 0.5);
            if(time_bin < time_bins){
               time_correlation[time_bin] += (data[now_index][j][i] * data[before_index][j][i]);
               time_count[time_bin] ++;
            }
         }
         if(dBZ_bin > 0){
            for(dj = 0; dj < pixelwidth_y; dj ++){
               for(di = 0; di < pixelwidth_x; di ++){
                  distance_bin = distance_bin_array[dj][di];
                  if(distance_bin > -1){
                     distance_correlation[distance_bin] += (data[now_index][j][i] * data[now_index][(j + dj) % ysize][(i + di) % xsize]);
                     distance_count[distance_bin] ++;
                  }
               }
            }
            for(dj = 1; dj < pixelwidth_y; dj ++){
               for(di = -pixelwidth_x + 1; di < 0; di ++){
                  distance_bin = distance_bin_array[dj][-di];
                  if(distance_bin > -1){
                     distance_correlation[distance_bin] += (data[now_index][j][i] * data[now_index][(j + dj) % ysize][(i + di) % xsize]);
                     distance_count[distance_bin] ++;
                  }
               }
            }
         }
      }
   }
   if(t > 0){
      minutes_difference = (minutes[t % time_bins] - minutes[(t - 1) % time_bins]);
      time_bin = (int) (minutes_difference / interval_binwidth + 0.5);
      if(time_bin < interval_bins){
         interval_histogram[time_bin] ++;
      }
   }
   
}

void output_averages_and_correlations(std::string pathstring, double **time_average, int nframes, int xsize, int ysize, long long int *interval_histogram, int interval_bins, double interval_binwidth, double *time_correlation, long long int *time_count, double *distance_correlation, long long int *distance_count, int time_bins, int distance_bins, double time_binwidth, double distance_binwidth, long long int *dBZ_histogram, double dBZ_binwidth, int dBZ_bins, double squareaverage)
{
   std::ofstream outfile;
   std::string suffix;
   std::string filename;
   int i, j;
   double overall_average = 0;
   
   // Output time-averaged values
   
   suffix = "_timeaverage.txt";
   filename = pathstring + suffix;
   outfile.open (filename.c_str(),std::ios::out);
   for(j = 0; j < ysize; j ++){
      for(i = 0; i < xsize; i ++){
         time_average[j][i] /= (1.*nframes);
         overall_average += time_average[j][i];
         outfile << time_average[j][i] << " ";
      }
      outfile << "\n";
   }
   outfile.close();
   overall_average /= (ysize * xsize);
   
   // Output histogram of time intervals
   
   suffix = "_intervalhistogram.txt";
   filename = pathstring + suffix;
   outfile.open (filename.c_str(),std::ios::out);
   for(i = 0; i < interval_bins; i ++){
      outfile << interval_binwidth * i << " " << interval_histogram[i] << "\n";
   }
   outfile.close();

   // Output dBZ histogram
   
   suffix = "_dBZhistogram.txt";
   filename = pathstring + suffix;
   outfile.open (filename.c_str(),std::ios::out);
   for(i = 0; i < dBZ_bins; i ++){
      outfile << dBZ_binwidth * i << " " << dBZ_histogram[i] << "\n";
   }
   outfile.close();

   // Output time correlation
   
   suffix = "_time.txt";
   filename = pathstring + suffix;
   outfile.open (filename.c_str(),std::ios::out);
   long long int divisor = xsize;
   divisor *= ysize;
   divisor *= nframes;
   squareaverage /= divisor;
   for(i = 0; i < time_bins; i ++){
      
      //std::cout << i << " " << time_correlation[i] << " " << time_count[i] << "\n";
      time_correlation[i] /= (1.*time_count[i]);
      outfile << i * time_binwidth << " " << time_correlation[i] << " " << overall_average << " " << squareaverage << "\n";
   }
   outfile.close();
   
   // Output distance correlation
   
   suffix = "_distance.txt";
   filename = pathstring + suffix;
   outfile.open (filename.c_str(),std::ios::out);
   for(i = 0; i < distance_bins; i ++){
      distance_correlation[i] /= (1.*distance_count[i]);
      outfile << i * distance_binwidth << " " << distance_correlation[i] << " " << " " << overall_average << " " << squareaverage << "\n";
   }
   outfile.close();
}

void initialize_network_Eulerian(double ****bias, double *****weight, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, int x_width, int y_width, double agree_value, double (*activation_function_inverse)(double), double (*activation_function_inverse_derivative)(double))
{
   
   // Create a chain of neurons that enforces that output = input at agree value and first derivative = 1
   
   int chain_index;
   double myweight = activation_function_inverse_derivative(agree_value);
   double mybias = activation_function_inverse(agree_value) - myweight * agree_value;
   for(int i = y_start; i < y_end; i ++){
      for(int j = x_start; j < x_end; j ++){
         for(int k = 1; k < number_total_layers - 1; k ++){
            if(k - 1 == 0){
               
               // Chain index of input layer corresponds to central pixel
               
               chain_index = (y_width - 1)/2*x_width + (x_width - 1)/2;
            }
            else{
               
               // Chain index of hidden layer is first index
               
               chain_index = 0;
            }
            bias[i][j][k][0] = mybias;
            for(int m = 0; m < number_neurons[k - 1]; m ++){
               if(m == chain_index){
                  weight[i][j][k][0][m] = myweight;
               }
               else{
                  weight[i][j][k][0][m] = 0;
               }
            }
            for(int l = 1; l < number_neurons[k]; l ++){
               bias[i][j][k][l] = 0;
               for(int m = 0; m < number_neurons[k - 1]; m ++){
                  weight[i][j][k][l][m] = 0;
               }
            }
         }
         
         // All output neurons receive same weight and bias
         // (Eulerian persistence for entire future interval)
         
         for(int l = 0; l < number_neurons[number_total_layers - 1]; l ++){
            bias[i][j][number_total_layers - 1][l] = mybias;
            weight[i][j][number_total_layers - 1][l][0] = myweight;
            for(int m = 0; m < number_neurons[number_total_layers - 2]; m ++){
               weight[i][j][number_total_layers - 1][l][m] = 0;
            }
         }
      }
   }
}

void feed_forward_and_backpropagate(float ***data, double min_dBZ, double max_dBZ, int y_start, int y_end, int x_start, int x_end, double *minutes, int counter, int time_depth, int time_depth_past, int time_depth_future, double interval, int y_width, int x_width, int number_total_layers, int *number_neurons, double **activation, double **weighted_input, double ****bias, double *****weight, double **error, double ***cost, double ***cost_persistent, double ****dcost_dbias, double *****dcost_dweight, long long int *pbatch_count, double (*activation_function)(double), double (*activation_function_derivative)(double), double (*cost_function)(double, double), double (*cost_derivative)(double, double))
{
   if(counter < time_depth_past + time_depth_future - 1){
      return;
   }
   int time_slot, i, j, k, l, m, y_slot;
   
   // Check that times are one-to-one with expected spacing
   
   double difference, observation;
   int number_intervals;
   for(i = 0; i < time_depth_past; i ++){
      difference = minutes[(counter - time_depth_future + 1) % time_depth] - minutes[(counter - time_depth_future - i) % time_depth];
      number_intervals = (int) (difference/interval + 0.5);
      if(number_intervals != i + 1){
         //std::cout << "past difference " << difference << " != (" << i << " + 1) x " << interval << std::endl;
         return;
      }
   }
   for(i = 0; i < time_depth_future; i ++){
      difference = minutes[(counter - time_depth_future + 1 + i) % time_depth] - minutes[(counter - time_depth_future + 1) % time_depth];
      number_intervals = (int) (difference/interval + 0.5);
      if(number_intervals != i){
         //std::cout << "future difference " << difference << " != " << i << " x " << interval << std::endl;
         return;
      }
   }
   
   int y_buffer = (y_width - 1) / 2;
   int x_buffer = (x_width - 1) / 2;
   for(i = y_start; i < y_end; i ++){
      for(j = x_start; j < x_end; j ++){
         
         // Assign normalized inputs to activation of input layer
         
         for(k = 0; k < time_depth_past; k ++){
            
            // Data runs forward in time but weight indexing runs backward in time
            
            time_slot = (counter - time_depth_future - k) % time_depth;
            for(l = 0; l < y_width; l ++){
               y_slot = i - y_buffer + l;
               for(m = 0; m < x_width; m ++){
                  activation[0][k*y_width*x_width + l*x_width + m] = (data[time_slot][y_slot][j - x_buffer + m] - min_dBZ)/(max_dBZ - min_dBZ);
               }
            }
         }
         
         // Feed forward to hidden and output layers
         
         for(k = 1; k < number_total_layers; k ++){
            for(l = 0; l < number_neurons[k]; l ++){
               weighted_input[k][l] = bias[i][j][k][l];
               for(m = 0; m < number_neurons[k - 1]; m ++){
                  weighted_input[k][l] += activation[k-1][m]*weight[i][j][k][l][m];
               }
               
               // Apply activation function
               
               activation[k][l] = activation_function(weighted_input[k][l]);
            }
         }
         
         // Calculate error in output layer from observations
         
         for(l = 0; l < time_depth_future; l ++){
            time_slot = (counter - time_depth_future + 1 + l) % time_depth;
            observation = (data[time_slot][i][j] - min_dBZ)/(max_dBZ - min_dBZ);
            
            // Calculate cost function to monitor
            
            cost[i][j][l] += cost_function(observation, activation[number_total_layers - 1][l]);
            //std::cout << "cost " << cost[i][j][l] << "\n";
            
            // Calculate cost for Eulerian persistence
            
            cost_persistent[i][j][l] += cost_function(observation, activation[0][k*y_width*x_width + (y_width - 1)/2*x_width + (x_width - 1)/2]);
            
            // Calculate "error" in output layer (actually dcost / d(weighted input))
            
            error[number_total_layers - 1][l] = cost_derivative(observation, activation[number_total_layers - 1][l]) * activation_function_derivative(weighted_input[number_total_layers - 1][l]);
            dcost_dbias[i][j][number_total_layers - 1][l] += error[number_total_layers - 1][l];
            for(m = 0; m < number_neurons[number_total_layers - 2]; m ++){
               dcost_dweight[i][j][number_total_layers - 1][l][m] += error[number_total_layers - 1][l] * activation[number_total_layers - 2][m];
            }
         }
         
         // Backpropagate error
         
         for(k = number_total_layers - 2; k >= 1; k --){
            for(l = 0; l < number_neurons[k]; l ++){
               error[k][l] = 0;
               for(m = 0; m < number_neurons[k + 1]; m ++){
                  error[k][l] += (weight[i][j][k + 1][m][l] * error[k + 1][m]);
               }
               error[k][l] *= activation_function_derivative(weighted_input[k][l]);
               dcost_dbias[i][j][k][l] += error[k][l];
               for(m = 0; m < number_neurons[k - 1]; m ++){
                  dcost_dweight[i][j][k][l][m] += error[k][l] * activation[k - 1][m];
               }
            }
         }
      }
   }
   (*pbatch_count) = (*pbatch_count) + 1;
   //std::cout << "batch count now " << (*pbatch_count) << "\n";
}

void feed_forward_validate(float ***data, double min_dBZ, double max_dBZ, int y_start, int y_end, int x_start, int x_end, double *minutes, int counter, int time_depth, int time_depth_past, int time_depth_future, double interval, int y_width, int x_width, int number_total_layers, int *number_neurons, double **activation, double **weighted_input, double ****bias, double *****weight, double ***cost, double ***cost_persistent, long long int *pbatch_count, double (*activation_function)(double), double (*activation_function_derivative)(double), double (*cost_function)(double, double), int epoch, double ***forecast_observation_correlation, double ***forecast_observation_correlation_persistent, double ***forecast_squareaverage, double ***forecast_persistent_squareaverage, double ***observation_squareaverage)
{
   if(counter < time_depth_past + time_depth_future - 1){
      return;
   }
   int time_slot, i, j, k, l, m, y_slot;
   
   // Check that times are one-to-one with expected spacing
   
   double difference, observation, observation0;
   int number_intervals;
   for(i = 0; i < time_depth_past; i ++){
      difference = minutes[(counter - time_depth_future + 1) % time_depth] - minutes[(counter - time_depth_future - i) % time_depth];
      number_intervals = (int) (difference/interval + 0.5);
      if(number_intervals != i + 1){
         return;
      }
   }
   for(i = 0; i < time_depth_future; i ++){
      difference = minutes[(counter - time_depth_future + 1 + i) % time_depth] - minutes[(counter - time_depth_future + 1) % time_depth];
      number_intervals = (int) (difference/interval + 0.5);
      if(number_intervals != i){
         return;
      }
   }   
   int y_buffer = (y_width - 1) / 2;
   int x_buffer = (x_width - 1) / 2;
   for(i = y_start; i < y_end; i ++){
      for(j = x_start; j < x_end; j ++){
         
         // Assign normalized inputs to activation of input layer
         
         for(k = 0; k < time_depth_past; k ++){
            
            // Data runs forward in time but weight indexing runs backward in time
            
            time_slot = (counter - time_depth_future - k) % time_depth;
            for(l = 0; l < y_width; l ++){
               y_slot = i - y_buffer + l;
               for(m = 0; m < x_width; m ++){
                  activation[0][k*y_width*x_width + l*x_width + m] = (data[time_slot][y_slot][j - x_buffer + m] - min_dBZ)/(max_dBZ - min_dBZ);
               }
            }
         }
         
         // Feed forward to hidden and output layers
         
         for(k = 1; k < number_total_layers; k ++){
            for(l = 0; l < number_neurons[k]; l ++){
               weighted_input[k][l] = bias[i][j][k][l];
               for(m = 0; m < number_neurons[k - 1]; m ++){
                  weighted_input[k][l] += activation[k-1][m]*weight[i][j][k][l][m];
               }
               
               // Apply activation function
               
               activation[k][l] = activation_function(weighted_input[k][l]);
            }
         }
         
         // Calculate averages
         
         time_slot = (counter - time_depth_future) % time_depth;
         observation0 = (data[time_slot][i][j] - min_dBZ)/(max_dBZ - min_dBZ);
         for(l = 0; l < time_depth_future; l ++){
            time_slot = (counter - time_depth_future + 1 + l) % time_depth;
            observation = (data[time_slot][i][j] - min_dBZ)/(max_dBZ - min_dBZ);
            
            // Calculate cost
            
            cost[i][j][l] += cost_function(observation, activation[number_total_layers - 1][l]);
            
            // Calculate forecast-observation correlation
            
            forecast_observation_correlation[i][j][l] += (observation * activation[number_total_layers - 1][l]);
            forecast_squareaverage[i][j][l] += (activation[number_total_layers - 1][l] * activation[number_total_layers - 1][l]);
            observation_squareaverage[i][j][l] += (observation * observation);
            if(epoch == 0){
            
               // Calculate cost for Eulerian persistence
            
               cost_persistent[i][j][l] += cost_function(observation, activation[0][k*y_width*x_width + (y_width - 1)/2*x_width + (x_width - 1)/2]);
               
               // Calculate time correlation for Eulerian persistence
               
               forecast_observation_correlation_persistent[i][j][l] += (observation * observation0);
               forecast_persistent_squareaverage[i][j][l] += (observation0 * observation0);
            }
         }
      }
   }
   (*pbatch_count) = (*pbatch_count) + 1;
}

void feed_forward_generalize(float ***data, double min_dBZ, double max_dBZ, int y_start, int y_end, int x_start, int x_end, double *minutes, int counter, int time_depth, int time_depth_past, int time_depth_future, double interval, int y_width, int x_width, int number_total_layers, int *number_neurons, double **activation, double **weighted_input, double **bias_average, double ***weight_average, double ***cost, double ***cost_persistent, long long int *pbatch_count, double (*activation_function)(double), double (*activation_function_derivative)(double), double (*cost_function)(double, double), int epoch, double ***forecast_observation_correlation, double ***forecast_observation_correlation_persistent, double ***forecast_squareaverage, double ***forecast_persistent_squareaverage, double ***observation_squareaverage)
{
   if(counter < time_depth_past + time_depth_future - 1){
      return;
   }
   int time_slot, i, j, k, l, m, y_slot;
   
   // Check that times are one-to-one with expected spacing
   
   double difference, observation, observation0;
   int number_intervals;
   for(i = 0; i < time_depth_past; i ++){
      difference = minutes[(counter - time_depth_future + 1) % time_depth] - minutes[(counter - time_depth_future - i) % time_depth];
      number_intervals = (int) (difference/interval + 0.5);
      if(number_intervals != i + 1){
         return;
      }
   }
   for(i = 0; i < time_depth_future; i ++){
      difference = minutes[(counter - time_depth_future + 1 + i) % time_depth] - minutes[(counter - time_depth_future + 1) % time_depth];
      number_intervals = (int) (difference/interval + 0.5);
      if(number_intervals != i){
         return;
      }
   }
   int y_buffer = (y_width - 1) / 2;
   int x_buffer = (x_width - 1) / 2;
   for(i = y_start; i < y_end; i ++){
      for(j = x_start; j < x_end; j ++){
         
         // Assign normalized inputs to activation of input layer
         
         for(k = 0; k < time_depth_past; k ++){

            // Data runs forward in time but weight indexing runs backward in time

            time_slot = (counter - time_depth_future - k) % time_depth;
            for(l = 0; l < y_width; l ++){
               y_slot = i - y_buffer + l;
               for(m = 0; m < x_width; m ++){
                  activation[0][k*y_width*x_width + l*x_width + m] = (data[time_slot][y_slot][j - x_buffer + m] - min_dBZ)/(max_dBZ - min_dBZ);
               }
            }
         }
         
         // Feed forward to hidden and output layers
         
         for(k = 1; k < number_total_layers; k ++){
            for(l = 0; l < number_neurons[k]; l ++){
               weighted_input[k][l] = bias_average[k][l];
               for(m = 0; m < number_neurons[k - 1]; m ++){
                  weighted_input[k][l] += activation[k-1][m]*weight_average[k][l][m];
               }
               
               // Apply activation function
               
               activation[k][l] = activation_function(weighted_input[k][l]);
            }
         }
         
         // Calculate averages
         
         time_slot = (counter - time_depth_future) % time_depth;
         observation0 = (data[time_slot][i][j] - min_dBZ)/(max_dBZ - min_dBZ);
         for(l = 0; l < time_depth_future; l ++){
            time_slot = (counter - time_depth_future + 1 + l) % time_depth;
            observation = (data[time_slot][i][j] - min_dBZ)/(max_dBZ - min_dBZ);
            
            // Calculate cost
            
            cost[i][j][l] += cost_function(observation, activation[number_total_layers - 1][l]);
            
            // Calculate forecast-observation correlation
            
            forecast_observation_correlation[i][j][l] += (observation * activation[number_total_layers - 1][l]);
            forecast_squareaverage[i][j][l] += (activation[number_total_layers - 1][l] * activation[number_total_layers - 1][l]);
            observation_squareaverage[i][j][l] += (observation * observation);
            if(epoch == 0){
               
               // Calculate cost for Eulerian persistence
               
               cost_persistent[i][j][l] += cost_function(observation, activation[0][k*y_width*x_width + (y_width - 1)/2*x_width + (x_width - 1)/2]);
               
               // Calculate time correlation for Eulerian persistence
               
               forecast_observation_correlation_persistent[i][j][l] += (observation * observation0);
               forecast_persistent_squareaverage[i][j][l] += (observation0 * observation0);
            }
         }
      }
   }
   (*pbatch_count) = (*pbatch_count) + 1;
}

void increment_and_output_neural_network(std::string output_directory, int epoch, time_t start_cputime, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ****bias, double *****weight, double ***cost, double ***cost_persistent, double ****dcost_dbias, double *****dcost_dweight, double learning_rate, long long int *pbatch_count, int *pdo_delete)
{
   std::ofstream outfile;
   std::string suffix;
   std::string filename;
   time_t relative_cputime;
   time(&relative_cputime);
   relative_cputime -= start_cputime;
   int i, j, k, l, m;
   double average, variance, count;
   
   for(k = 1; k < number_total_layers; k ++){
      
      // Output dcost_dbias for all pixels and neurons
      
      average = variance = count = 0;
      std::stringstream ss;
      ss << k;
      suffix = "/dcost_dbias_layer" + ss.str() + "_individual.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         outfile << "epoch cputime";
         for(i = y_start; i < y_end; i ++){
            for(j = x_start; j < x_end; j ++){
               for(l = 0; l < number_neurons[k]; l ++){
                  outfile << " " << i << "," << j << "," << l;
               }
            }
         }
         outfile << "\n";
      }
      outfile << epoch << " " << relative_cputime;
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[k]; l ++){
               dcost_dbias[i][j][k][l] /= (*pbatch_count);
               outfile << " " << dcost_dbias[i][j][k][l];
               average += dcost_dbias[i][j][k][l];
               variance += dcost_dbias[i][j][k][l] * dcost_dbias[i][j][k][l];
               count += 1;
               
               // Increment bias in proportion to dcost/dbias (gradient descent)
               
               bias[i][j][k][l] -= (learning_rate * dcost_dbias[i][j][k][l]);
               dcost_dbias[i][j][k][l] = 0;
            }
         }
      }
      outfile << "\n";
      outfile.close();
      
      // Output average dcost_dbias
      
      suffix = "/dcost_dbias_layer" + ss.str() + "_average.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         outfile << "epoch cputime average stddev\n";
      }
      average /= count;
      variance = sqrt(variance / count - average * average);
      outfile << epoch << " " << relative_cputime << " " << average << " " << variance << "\n";
      outfile.close();
      
      // Output dcost_dweight for all pixels and connections
      
      average = variance = count = 0;
      suffix = "/dcost_dweight_layer" + ss.str() + "_individual.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         outfile << "epoch cputime";
         for(i = y_start; i < y_end; i ++){
            for(j = x_start; j < x_end; j ++){
               for(l = 0; l < number_neurons[k]; l ++){
                  for(m = 0; m < number_neurons[k - 1]; m ++){
                     outfile << " " << i << "," << j << "," << l << "," << m;
                  }
               }
            }
         }
         outfile << "\n";
      }
      outfile << epoch << " " << relative_cputime;
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[k]; l ++){
               for(m = 0; m < number_neurons[k - 1]; m ++){
                  dcost_dweight[i][j][k][l][m] /= (*pbatch_count);
                  outfile << " " << dcost_dweight[i][j][k][l][m];
                  average += dcost_dweight[i][j][k][l][m];
                  variance += dcost_dweight[i][j][k][l][m] * dcost_dweight[i][j][k][l][m];
                  count += 1;
                  
                  // Increment weight in proportion to dcost/dweight (gradient descent)
                  
                  weight[i][j][k][l][m] -= (learning_rate * dcost_dweight[i][j][k][l][m]);
                  dcost_dweight[i][j][k][l][m] = 0;
               }
            }
         }
      }
      outfile << "\n";
      outfile.close();
      
      // Output average dcost_dweight
      
      suffix = "/dcost_dweight_layer" + ss.str() + "_average.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         outfile << "epoch cputime average stddev\n";
      }
      average /= count;
      variance = sqrt(variance / count - average * average);
      outfile << epoch << " " << relative_cputime << " " << average << " " << variance << "\n";
      outfile.close();
      
      // Output bias for all pixels and neurons
      
      average = variance = count = 0;
      suffix = "/bias_layer" + ss.str() + "_individual.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         outfile << "epoch cputime";
         for(i = y_start; i < y_end; i ++){
            for(j = x_start; j < x_end; j ++){
               for(l = 0; l < number_neurons[k]; l ++){
                  outfile << " " << i << "," << j << "," << l;
               }
            }
         }
         outfile << "\n";
      }
      outfile << epoch << " " << relative_cputime;
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[k]; l ++){
               outfile << " " << bias[i][j][k][l];
               average += bias[i][j][k][l];
               variance += bias[i][j][k][l] * bias[i][j][k][l];
               count += 1;
            }
         }
      }
      outfile << "\n";
      outfile.close();
      
      // Output average bias
      
      suffix = "/bias_layer" + ss.str() + "_average.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         outfile << "epoch cputime average stddev\n";
      }
      average /= count;
      variance = sqrt(variance / count - average * average);
      outfile << epoch << " " << relative_cputime << " " << average << " " << variance << "\n";
      outfile.close();
      
      // Output weight for all pixels and connections
      
      average = variance = count = 0;
      suffix = "/weight_layer" + ss.str() + "_individual.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         outfile << "epoch cputime";
         for(i = y_start; i < y_end; i ++){
            for(j = x_start; j < x_end; j ++){
               for(l = 0; l < number_neurons[k]; l ++){
                  for(m = 0; m < number_neurons[k - 1]; m ++){
                     outfile << " " << i << "," << j << "," << l << "," << m;
                  }
               }
            }
         }
         outfile << "\n";
      }
      outfile << epoch << " " << relative_cputime;
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[k]; l ++){
               for(m = 0; m < number_neurons[k - 1]; m ++){
                  outfile << " " << weight[i][j][k][l][m];
                  average += weight[i][j][k][l][m];
                  variance += weight[i][j][k][l][m] * weight[i][j][k][l][m];
                  count += 1;
               }
            }
         }
      }
      outfile << "\n";
      outfile.close();
      
      // Output average weight
      
      suffix = "/weight_layer" + ss.str() + "_average.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         outfile << "epoch cputime average stddev\n";
      }
      average /= count;
      variance = sqrt(variance / count - average * average);
      outfile << epoch << " " << relative_cputime << " " << average << " " << variance << "\n";
      outfile.close();
   }
   
   // Output cost for all pixels and output neurons
   
   average = variance = count = 0;
   suffix = "/cost_individual.txt";
   filename = output_directory + suffix;
   if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
   else outfile.open (filename.c_str(),std::ios::app);
   if((*pdo_delete) == 1){
      outfile << "epoch cputime";
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
               outfile << " " << i << "," << j << "," << l;
            }
         }
      }
      outfile << "\n";
   }
   outfile << epoch << " " << relative_cputime;
   for(i = y_start; i < y_end; i ++){
      for(j = x_start; j < x_end; j ++){
         for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
            cost[i][j][l] /= (*pbatch_count);
            outfile << " " << cost[i][j][l];
            average += cost[i][j][l];
            variance += cost[i][j][l] * cost[i][j][l];
            count += 1;
            cost[i][j][l] = 0;
         }
      }
   }
   outfile << "\n";
   outfile.close();
   
   // Output average cost
   
   suffix = "/cost_average.txt";
   filename = output_directory + suffix;
   if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
   else outfile.open (filename.c_str(),std::ios::app);
   if((*pdo_delete) == 1){
      outfile << "epoch cputime average stddev\n";
   }
   average /= count;
   variance = sqrt(variance / count - average * average);
   outfile << epoch << " " << relative_cputime << " " << average << " " << variance << "\n";
   outfile.close();
   
   if(epoch==0){
      
      // Output Eulerian persistence cost for all pixels and output neurons
      
      average = variance = count = 0;
      suffix = "/cost_persistent_individual.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         for(i = y_start; i < y_end; i ++){
            for(j = x_start; j < x_end; j ++){
               for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
                  outfile << i << "," << j << "," << l << " ";
               }
            }
         }
         outfile << "\n";
      }
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
               cost_persistent[i][j][l] /= (*pbatch_count);
               outfile << cost_persistent[i][j][l] << " ";
               average += cost_persistent[i][j][l];
               variance += cost_persistent[i][j][l] * cost_persistent[i][j][l];
               count += 1;
               cost_persistent[i][j][l] = 0;
            }
         }
      }
      outfile << "\n";
      outfile.close();
      
      // Output average cost for Eulerian persistence
      
      suffix = "/cost_persistent_average.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         outfile << "average stddev\n";
      }
      average /= count;
      variance = sqrt(variance / count - average * average);
      outfile << average << " " << variance << "\n";
      outfile.close();
   }
   *pbatch_count = 0;
}

void increment_neural_network(int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ****bias, double *****weight, double ****dcost_dbias, double *****dcost_dweight, double learning_rate, long long int *pbatch_count, long long int *poutput_batch_count)
{
   int i, j, k, l, m;
   double average, variance, count;
      
   // Gradient descent
   
   for(k = 1; k < number_total_layers; k ++){
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[k]; l ++){
               dcost_dbias[i][j][k][l] /= (*pbatch_count);
               bias[i][j][k][l] -= (learning_rate * dcost_dbias[i][j][k][l]);
               dcost_dbias[i][j][k][l] = 0;
               for(m = 0; m < number_neurons[k - 1]; m ++){
                  dcost_dweight[i][j][k][l][m] /= (*pbatch_count);
                  weight[i][j][k][l][m] -= (learning_rate * dcost_dweight[i][j][k][l][m]);
                  dcost_dweight[i][j][k][l][m] = 0;
               }
            }
         }
      }
   }
   *poutput_batch_count += (*pbatch_count);
   *pbatch_count = 0;
}

void output_neural_network(std::string output_directory, int epoch, time_t start_cputime, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ****bias, double *****weight, double ***cost, long long int *pbatch_count, long long int *poverall_batch_count, int *pdo_delete)
{
   std::ofstream outfile;
   std::string suffix;
   std::string filename;
   time_t relative_cputime;
   time(&relative_cputime);
   relative_cputime -= start_cputime;
   int i, j, k, l, m;
   double average, variance, count;
   
   for(k = 1; k < number_total_layers; k ++){
      std::stringstream ss;
      ss << k;
      
      // Output bias for all pixels and neurons
      
      average = variance = count = 0;
      suffix = "/bias_layer" + ss.str() + "_individual.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         outfile << "count epoch cputime";
         for(i = y_start; i < y_end; i ++){
            for(j = x_start; j < x_end; j ++){
               for(l = 0; l < number_neurons[k]; l ++){
                  outfile << " " << i << "," << j << "," << l;
               }
            }
         }
         outfile << "\n";
      }
      outfile << (*poverall_batch_count) + (*pbatch_count) << " " << epoch << " " << relative_cputime;
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[k]; l ++){
               outfile << " " << bias[i][j][k][l];
               average += bias[i][j][k][l];
               variance += bias[i][j][k][l] * bias[i][j][k][l];
               count += 1;
            }
         }
      }
      outfile << "\n";
      outfile.close();
      
      // Output average bias
      
      suffix = "/bias_layer" + ss.str() + "_average.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         outfile << "count epoch cputime average stddev\n";
      }
      average /= count;
      variance = sqrt(variance / count - average * average);
      outfile << (*poverall_batch_count) + (*pbatch_count) << " " << epoch << " " << relative_cputime << " " << average << " " << variance << "\n";
      outfile.close();
      
      // Output weight for all pixels and connections
      
      average = variance = count = 0;
      suffix = "/weight_layer" + ss.str() + "_individual.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         outfile << "count epoch cputime";
         for(i = y_start; i < y_end; i ++){
            for(j = x_start; j < x_end; j ++){
               for(l = 0; l < number_neurons[k]; l ++){
                  for(m = 0; m < number_neurons[k - 1]; m ++){
                     outfile << " " << i << "," << j << "," << l << "," << m;
                  }
               }
            }
         }
         outfile << "\n";
      }
      outfile << (*poverall_batch_count) + (*pbatch_count) << " " << epoch << " " << relative_cputime;
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[k]; l ++){
               for(m = 0; m < number_neurons[k - 1]; m ++){
                  outfile << " " << weight[i][j][k][l][m];
                  average += weight[i][j][k][l][m];
                  variance += weight[i][j][k][l][m] * weight[i][j][k][l][m];
                  count += 1;
               }
            }
         }
      }
      outfile << "\n";
      outfile.close();
      
      // Output average weight
      
      suffix = "/weight_layer" + ss.str() + "_average.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         outfile << "count epoch cputime average stddev\n";
      }
      average /= count;
      variance = sqrt(variance / count - average * average);
      outfile << (*poverall_batch_count) + (*pbatch_count) << " " << epoch << " " << relative_cputime << " " << average << " " << variance << "\n";
      outfile.close();
   }
   
   // Output cost for all pixels and output neurons
   
   average = variance = count = 0;
   suffix = "/cost_individual.txt";
   filename = output_directory + suffix;
   if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
   else outfile.open (filename.c_str(),std::ios::app);
   if((*pdo_delete) == 1){
      outfile << "count epoch cputime";
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
               outfile << " " << i << "," << j << "," << l;
            }
         }
      }
      outfile << "\n";
   }
   outfile << (*poverall_batch_count) + (*pbatch_count) << " " << epoch << " " << relative_cputime;
   for(i = y_start; i < y_end; i ++){
      for(j = x_start; j < x_end; j ++){
         for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
            //std::cout << "cost = " << cost[i][j][l] << " / " << (*pbatch_count) << "\n";
            cost[i][j][l] /= (*pbatch_count);
            outfile << " " << cost[i][j][l];
            average += cost[i][j][l];
            variance += cost[i][j][l] * cost[i][j][l];
            count += 1;
            cost[i][j][l] = 0;
         }
      }
   }
   outfile << "\n";
   outfile.close();
   
   // Output average cost
   
   suffix = "/cost_average.txt";
   filename = output_directory + suffix;
   if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
   else outfile.open (filename.c_str(),std::ios::app);
   if((*pdo_delete) == 1){
      outfile << "count epoch cputime average stddev\n";
   }
   average /= count;
   variance = sqrt(variance / count - average * average);
   outfile << (*poverall_batch_count) + (*pbatch_count) << " " << epoch << " " << relative_cputime << " " << average << " " << variance << "\n";
   outfile.close();
   *poverall_batch_count += (*pbatch_count);
   *pbatch_count = 0;
}

void input_neural_network(std::string input_directory, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double **bias_average, double ***weight_average)
{
   std::ifstream infile;
   std::string suffix;
   std::string filename;
   std::string line;
   std::string lastline;
   std::string item;
   int i, j, k, l, m;
   
   for(k = 1; k < number_total_layers; k ++){
      for(l = 0; l < number_neurons[k]; l ++){
         bias_average[k][l] = 0;
         for(m = 0; m < number_neurons[k - 1]; m ++){
            weight_average[k][l][m] = 0;
         }
      }
      std::stringstream ss;
      ss << k;
      
      // Input bias
      
      suffix = "/bias_layer" + ss.str() + "_individual.txt";
      filename = input_directory + suffix;
      infile.open (filename.c_str(),std::ios::in);
      
      // Skip to last line
      
      while (std::getline(infile, line))
      {
         lastline = line;
      }
      
      std::istringstream iss(lastline);
      
      // Skip three words (count, epoch, cputime)
      
      for(i = 0; i < 3; i ++){
         std::getline(iss, item, ' ');
      }
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[k]; l ++){
               std::getline(iss, item, ' ');
               bias_average[k][l] += atof(item.c_str());
            }
         }
      }
      for(l = 0; l < number_neurons[k]; l ++){
         bias_average[k][l] /= ((y_end - y_start)*(x_end - x_start));
      }
      infile.close();
      
      // Input weight
      
      suffix = "/weight_layer" + ss.str() + "_individual.txt";
      filename = input_directory + suffix;
      infile.open (filename.c_str(),std::ios::in);
      
      // Skip to last line
      
      while (std::getline(infile, line))
      {
         lastline = line;
      }
      
      std::istringstream iss2(lastline);
      
      // Skip three words (count, epoch, cputime)
      
      for(i = 0; i < 3; i ++){
         std::getline(iss2, item, ' ');
      }
            
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[k]; l ++){
               for(m = 0; m < number_neurons[k - 1]; m ++){
                  std::getline(iss2, item, ' ');
                  weight_average[k][l][m] += atof(item.c_str());
               }
            }
         }
      }
      for(l = 0; l < number_neurons[k]; l ++){
         for(m = 0; m < number_neurons[k - 1]; m ++){
            weight_average[k][l][m] /= ((y_end - y_start)*(x_end - x_start));
         }
      }
      infile.close();
   }
}

void output_persistent_cost(std::string output_directory, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ***cost_persistent, long long int batch_count)
{
   std::ofstream outfile;
   std::string suffix;
   std::string filename;
   int i, j, k, l;
   double average, variance, count;
   
   // Output Eulerian persistence cost for all pixels and output neurons
   
   average = variance = count = 0;
   suffix = "/cost_persistent_individual.txt";
   filename = output_directory + suffix;
   outfile.open (filename.c_str(),std::ios::out);
   for(i = y_start; i < y_end; i ++){
      for(j = x_start; j < x_end; j ++){
         for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
            outfile << i << "," << j << "," << l << " ";
         }
      }
   }
   outfile << "\n";
   for(i = y_start; i < y_end; i ++){
      for(j = x_start; j < x_end; j ++){
         for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
            cost_persistent[i][j][l] /= (batch_count);
            outfile << cost_persistent[i][j][l] << " ";
            average += cost_persistent[i][j][l];
            variance += cost_persistent[i][j][l] * cost_persistent[i][j][l];
            count += 1;
            cost_persistent[i][j][l] = 0;
         }
      }
   }
   outfile << "\n";
   outfile.close();
   
   // Output average cost for Eulerian persistence
   
   suffix = "/cost_persistent_average.txt";
   filename = output_directory + suffix;
   outfile.open (filename.c_str(),std::ios::out);
   outfile << "average stddev\n";
   average /= count;
   variance = sqrt(variance / count - average * average);
   outfile << average << " " << variance << "\n";
   outfile.close();
}

void output_validation_cost(std::string output_directory, int epoch, time_t start_cputime, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ***cost, double ***cost_persistent, double ***forecast_observation_correlation, double ***forecast_observation_correlation_persistent, double ***forecast_squareaverage, double ***forecast_persistent_squareaverage, double ***observation_squareaverage, long long int *pbatch_count, int *pdo_delete)
{
   std::ofstream outfile;
   std::string suffix;
   std::string filename;
   time_t relative_cputime;
   time(&relative_cputime);
   relative_cputime -= start_cputime;
   int i, j, k, l, m;
   double average, variance, count;
      
   // Output cost for all pixels and output neurons
   
   average = variance = count = 0;
   suffix = "/cost_validate_individual.txt";
   filename = output_directory + suffix;
   if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
   else outfile.open (filename.c_str(),std::ios::app);
   if((*pdo_delete) == 1){
      outfile << "epoch cputime";
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
               outfile << " " << i << "," << j << "," << l;
            }
         }
      }
      outfile << "\n";
   }
   outfile << epoch << " " << relative_cputime;
   for(i = y_start; i < y_end; i ++){
      for(j = x_start; j < x_end; j ++){
         for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
            cost[i][j][l] /= (*pbatch_count);
            outfile << " " << cost[i][j][l];
            average += cost[i][j][l];
            variance += cost[i][j][l] * cost[i][j][l];
            count += 1;
            cost[i][j][l] = 0;
         }
      }
   }
   outfile << "\n";
   outfile.close();
   
   // Output average cost
   
   suffix = "/cost_validate_average.txt";
   filename = output_directory + suffix;
   if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
   else outfile.open (filename.c_str(),std::ios::app);
   if((*pdo_delete) == 1){
      outfile << "epoch cputime average stddev\n";
   }
   average /= count;
   variance = sqrt(variance / count - average * average);
   outfile << epoch << " " << relative_cputime << " " << average << " " << variance << "\n";
   outfile.close();
   
   // Output normalized forecast-observation correlation for all pixels and output neurons
   
   average = variance = count = 0;
   suffix = "/correlation_validate_individual.txt";
   filename = output_directory + suffix;
   if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
   else outfile.open (filename.c_str(),std::ios::app);
   if((*pdo_delete) == 1){
      outfile << "epoch cputime";
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
               outfile << " " << i << "," << j << "," << l;
            }
         }
      }
      outfile << "\n";
   }
   outfile << epoch << " " << relative_cputime;
   for(i = y_start; i < y_end; i ++){
      for(j = x_start; j < x_end; j ++){
         for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
            //forecast_observation_correlation[i][j][l] /= sqrt(squareaverage[i][j] * forecast_squareaverage[i][j][l]);
            forecast_observation_correlation[i][j][l] /= sqrt(observation_squareaverage[i][j][l] * forecast_squareaverage[i][j][l]);
            outfile << " " << forecast_observation_correlation[i][j][l];
            average += forecast_observation_correlation[i][j][l];
            variance += forecast_observation_correlation[i][j][l] * forecast_observation_correlation[i][j][l];
            count += 1;
            //forecast_observation_correlation[i][j][l] = 0;
            forecast_squareaverage[i][j][l] = 0;
            if(epoch > 0) observation_squareaverage[i][j][l] = 0;
         }
         //if(epoch > 0) squareaverage[i][j] = 0;
      }
   }
   outfile << "\n";
   outfile.close();
   
   // Output correlation time series
   
   std::string suffix2 = ".txt";
   std::stringstream ss;
   ss << epoch;
   suffix = "/correlation_validate_individual_epoch";
   filename = output_directory + suffix + ss.str() + suffix2;
   outfile.open (filename.c_str(),std::ios::out);
   for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            outfile << forecast_observation_correlation[i][j][l] << " ";
            forecast_observation_correlation[i][j][l] = 0;
         }
      }
      outfile << "\n";
   }
   outfile.close();
   
   // Output average normalized forecast-observation correlation
   
   suffix = "/correlation_validate_average.txt";
   filename = output_directory + suffix;
   if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
   else outfile.open (filename.c_str(),std::ios::app);
   if((*pdo_delete) == 1){
      outfile << "epoch cputime average stddev\n";
   }
   average /= count;
   variance = sqrt(variance / count - average * average);
   outfile << epoch << " " << relative_cputime << " " << average << " " << variance << "\n";
   outfile.close();
   
   if(epoch==0){
      
      // Output Eulerian persistence cost for all pixels and output neurons
      
      average = variance = count = 0;
      suffix = "/cost_validate_persistent_individual.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         for(i = y_start; i < y_end; i ++){
            for(j = x_start; j < x_end; j ++){
               for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
                  outfile << i << "," << j << "," << l << " ";
               }
            }
         }
         outfile << "\n";
      }
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
               cost_persistent[i][j][l] /= (*pbatch_count);
               outfile << cost_persistent[i][j][l] << " ";
               average += cost_persistent[i][j][l];
               variance += cost_persistent[i][j][l] * cost_persistent[i][j][l];
               count += 1;
               cost_persistent[i][j][l] = 0;
            }
         }
      }
      outfile << "\n";
      outfile.close();
      
      // Output average cost for Eulerian persistence
      
      suffix = "/cost_validate_persistent_average.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         outfile << "average stddev\n";
      }
      average /= count;
      variance = sqrt(variance / count - average * average);
      outfile << average << " " << variance << "\n";
      outfile.close();

      // Output normalized persistent forecast-observation correlation for all pixels and output neurons
      
      average = variance = count = 0;
      suffix = "/correlation_validate_persistent_individual.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         for(i = y_start; i < y_end; i ++){
            for(j = x_start; j < x_end; j ++){
               for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
                  outfile << i << "," << j << "," << l<< " " ;
               }
            }
         }
         outfile << "\n";
      }
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
               //std::cout << "output " << i << " " << j << " " << l << " " <<forecast_observation_correlation_persistent[i][j][l] << " " << squareaverage[i][j] << "\n";
               //if(l ==1){
                  //std::cout << "output " << i << " " << j << " " << l << " " <<forecast_observation_correlation_persistent[i][j][l] << " " << forecast_persistent_squareaverage[i][j][l] << " " << observation_squareaverage[i][j][l] << " " << forecast_observation_correlation_persistent[i][j][l] / sqrt(forecast_persistent_squareaverage[i][j][l] * observation_squareaverage[i][j][l]) << " " << (*pbatch_count) << "\n";
                  //exit(1);
               //}
               //forecast_observation_correlation_persistent[i][j][l] /= squareaverage[i][j];
               forecast_observation_correlation_persistent[i][j][l] /= sqrt(forecast_persistent_squareaverage[i][j][l] * observation_squareaverage[i][j][l]);
               //forecast_observation_correlation_persistent[i][j][l] /= sqrt(squareaverage[i][j] * observation_squareaverage[i][j][l]);
               outfile << forecast_observation_correlation_persistent[i][j][l] << " ";
               average += forecast_observation_correlation_persistent[i][j][l];
               variance += forecast_observation_correlation_persistent[i][j][l] * forecast_observation_correlation_persistent[i][j][l];
               count += 1;
               //forecast_observation_correlation_persistent[i][j][l] = 0;
               forecast_persistent_squareaverage[i][j][l] = 0;
               observation_squareaverage[i][j][l] = 0;
            }
            //exit(1);
            //squareaverage[i][j] = 0;
         }
      }
      outfile << "\n";
      outfile.close();
      
      // Output persistent correlation time series
      
      suffix = "/correlation_validate_persistent_individual_timeseries.txt";
      filename = output_directory + suffix;
      outfile.open (filename.c_str(),std::ios::out);
      for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
         for(i = y_start; i < y_end; i ++){
            for(j = x_start; j < x_end; j ++){
               outfile << forecast_observation_correlation_persistent[i][j][l] << " ";
               forecast_observation_correlation_persistent[i][j][l] = 0;
            }
         }
         outfile << "\n";
      }
      outfile.close();

      // Output average normalized forecast-observation correlation
      
      suffix = "/correlation_validate_persistent_average.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         outfile << "average stddev\n";
      }
      average /= count;
      variance = sqrt(variance / count - average * average);
      outfile << average << " " << variance << "\n";
      outfile.close();
   }
   *pbatch_count = 0;
}

void output_generalization_cost(std::string output_directory, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ***cost, double ***cost_persistent, double ***forecast_observation_correlation, double ***forecast_observation_correlation_persistent, double ***forecast_squareaverage, double ***forecast_persistent_squareaverage, double ***observation_squareaverage, long long int *pbatch_count)
{
   std::ofstream outfile;
   std::string suffix;
   std::string extension = ".txt";
   std::stringstream ss;
   std::string filename;
   int i, j, k, l, m;
   double **timeaverage = new double*[y_end - y_start];
   for(i = y_start; i < y_end; i ++) timeaverage[i - y_start] = new double[x_end - x_start];
   double *spaceaverage = new double[number_neurons[number_total_layers - 1]];
   
   // Output cost map for each forecast time
   
   for(i = 0; i < y_end - y_start; i ++){
      for(j = 0; j < x_end - x_start; j ++){
         timeaverage[i][j] = 0;
      }
   }
   for(i = 0; i < number_neurons[number_total_layers - 1]; i ++){
      spaceaverage[i] = 0;
   }
   suffix = "/cost_map_";
   for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
      ss << l;
      filename = output_directory + suffix + ss.str() + extension;
      ss.str("");
      outfile.open (filename.c_str(),std::ios::out);
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            cost[i][j][l] /= (*pbatch_count);
            outfile << cost[i][j][l] << " ";
            timeaverage[i - y_start][j - x_start] += cost[i][j][l];
            spaceaverage[l] += cost[i][j][l];
         }
         outfile << "\n";
      }
      outfile.close();
   }
   
   // Output cost time average map
   
   suffix = "/cost_timeaverage_map.txt";
   filename = output_directory + suffix;
   outfile.open (filename.c_str(),std::ios::out);
   for(i = 0; i < y_end - y_start; i ++){
      for(j = 0; j < x_end - x_start; j ++){
         timeaverage[i][j] /= (number_neurons[number_total_layers - 1]);
         outfile << timeaverage[i][j] << " ";
      }
      outfile << "\n";
   }
   outfile.close();
   
   // Output cost time series, averaged over space
   
   suffix = "/cost_timeseries.txt";
   filename = output_directory + suffix;
   outfile.open (filename.c_str(),std::ios::out);
   for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
      spaceaverage[l] /= ((x_end - x_start)*(y_end - y_start));
      outfile << spaceaverage[l] << "\n";
   }
   outfile.close();
   
   // Output forecast-observation correlation map for each forecast time
   
   for(i = 0; i < y_end - y_start; i ++){
      for(j = 0; j < x_end - x_start; j ++){
         timeaverage[i][j] = 0;
      }
   }
   for(i = 0; i < number_neurons[number_total_layers - 1]; i ++){
      spaceaverage[i] = 0;
   }
   suffix = "/correlation_map_";
   for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
      ss << l;
      filename = output_directory + suffix + ss.str() + extension;
      ss.str("");
      outfile.open (filename.c_str(),std::ios::out);
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            forecast_observation_correlation[i][j][l] /= sqrt(observation_squareaverage[i][j][l] * forecast_squareaverage[i][j][l]);
            outfile << forecast_observation_correlation[i][j][l] << " ";
            timeaverage[i - y_start][j - x_start] += forecast_observation_correlation[i][j][l];
            spaceaverage[l] += forecast_observation_correlation[i][j][l];
         }
         outfile << "\n";
      }
      outfile.close();
   }
   
   // Output forecast-observation correlation time average map
   
   suffix = "/correlation_timeaverage_map.txt";
   filename = output_directory + suffix;
   outfile.open (filename.c_str(),std::ios::out);
   for(i = 0; i < y_end - y_start; i ++){
      for(j = 0; j < x_end - x_start; j ++){
         timeaverage[i][j] /= (number_neurons[number_total_layers - 1]);
         outfile << timeaverage[i][j] << " ";
      }
      outfile << "\n";
   }
   outfile.close();
   
   // Output forecast-observation correlation time series, averaged over space
   
   suffix = "/correlation_timeseries.txt";
   filename = output_directory + suffix;
   outfile.open (filename.c_str(),std::ios::out);
   for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
      spaceaverage[l] /= ((x_end - x_start)*(y_end - y_start));
      outfile << spaceaverage[l] << "\n";
   }
   outfile.close();
   
   // Output Eulerian persistence cost map for each forecast time
   
   for(i = 0; i < y_end - y_start; i ++){
      for(j = 0; j < x_end - x_start; j ++){
         timeaverage[i][j] = 0;
      }
   }
   for(i = 0; i < number_neurons[number_total_layers - 1]; i ++){
      spaceaverage[i] = 0;
   }
   suffix = "/cost_persistent_map_";
   for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
      ss << l;
      filename = output_directory + suffix + ss.str() + extension;
      ss.str("");
      outfile.open (filename.c_str(),std::ios::out);
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            cost_persistent[i][j][l] /= (*pbatch_count);
            outfile << cost[i][j][l] << " ";
            timeaverage[i - y_start][j - x_start] += cost_persistent[i][j][l];
            spaceaverage[l] += cost_persistent[i][j][l];
         }
         outfile << "\n";
      }
      outfile.close();
   }
   
   // Output Eulerian persistence cost time average map
   
   suffix = "/cost_persistent_timeaverage_map.txt";
   filename = output_directory + suffix;
   outfile.open (filename.c_str(),std::ios::out);
   for(i = 0; i < y_end - y_start; i ++){
      for(j = 0; j < x_end - x_start; j ++){
         timeaverage[i][j] /= (number_neurons[number_total_layers - 1]);
         outfile << timeaverage[i][j] << " ";
      }
      outfile << "\n";
   }
   outfile.close();
   
   // Output Eulerian persistence cost time series, averaged over space
   
   suffix = "/cost_persistent_timeseries.txt";
   filename = output_directory + suffix;
   outfile.open (filename.c_str(),std::ios::out);
   for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
      spaceaverage[l] /= ((x_end - x_start)*(y_end - y_start));
      outfile << spaceaverage[l] << "\n";
   }
   outfile.close();
   
   // Output Eulerian persistence forecast-observation correlation map for each forecast time
   
   for(i = 0; i < y_end - y_start; i ++){
      for(j = 0; j < x_end - x_start; j ++){
         timeaverage[i][j] = 0;
      }
   }
   for(i = 0; i < number_neurons[number_total_layers - 1]; i ++){
      spaceaverage[i] = 0;
   }
   suffix = "/correlation_persistent_map_";
   for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
      ss << l;
      filename = output_directory + suffix + ss.str() + extension;
      ss.str("");
      outfile.open (filename.c_str(),std::ios::out);
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            forecast_observation_correlation_persistent[i][j][l] /= sqrt(forecast_persistent_squareaverage[i][j][l] * observation_squareaverage[i][j][l]);
            outfile << forecast_observation_correlation_persistent[i][j][l] << " ";
            timeaverage[i - y_start][j - x_start] += forecast_observation_correlation_persistent[i][j][l];
            spaceaverage[l] += forecast_observation_correlation_persistent[i][j][l];
         }
         outfile << "\n";
      }
      outfile.close();
   }
   
   // Output Eulerian persistence forecast-observation correlation time average map
   
   suffix = "/correlation_persistent_timeaverage_map.txt";
   filename = output_directory + suffix;
   outfile.open (filename.c_str(),std::ios::out);
   for(i = 0; i < y_end - y_start; i ++){
      for(j = 0; j < x_end - x_start; j ++){
         timeaverage[i][j] /= (number_neurons[number_total_layers - 1]);
         outfile << timeaverage[i][j] << " ";
      }
      outfile << "\n";
   }
   outfile.close();
   
   // Output Eulerian persistence forecast-observation correlation time series, averaged over space
   
   suffix = "/correlation_persistent_timeseries.txt";
   filename = output_directory + suffix;
   outfile.open (filename.c_str(),std::ios::out);
   for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
      spaceaverage[l] /= ((x_end - x_start)*(y_end - y_start));
      outfile << spaceaverage[l] << "\n";
   }
   outfile.close();

   for(i = 0; i < y_end - y_start; i ++){
      delete[] timeaverage[i];
   }
   delete[] timeaverage;
   delete[] spaceaverage;
   *pbatch_count = 0;
}

void output_validation_cost_training_count(std::string output_directory, int epoch, time_t start_cputime, int y_start, int y_end, int x_start, int x_end, int number_total_layers, int *number_neurons, double ***cost, double ***cost_persistent, double ***forecast_observation_correlation, double ***forecast_observation_correlation_persistent, double ***forecast_squareaverage, double ***forecast_persistent_squareaverage, double ***observation_squareaverage, long long int *pbatch_count, int *pdo_delete, long long int training_batch_count)
{
   std::ofstream outfile;
   std::string suffix;
   std::string filename;
   time_t relative_cputime;
   time(&relative_cputime);
   relative_cputime -= start_cputime;
   int i, j, k, l, m;
   double average, variance, count;
   
   // Output cost for all pixels and output neurons
   
   average = variance = count = 0;
   suffix = "/cost_validate_individual.txt";
   filename = output_directory + suffix;
   if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
   else outfile.open (filename.c_str(),std::ios::app);
   if((*pdo_delete) == 1){
      outfile << "count epoch cputime";
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
               outfile << " " << i << "," << j << "," << l;
            }
         }
      }
      outfile << "\n";
   }
   outfile << training_batch_count << " " << epoch << " " << relative_cputime;
   for(i = y_start; i < y_end; i ++){
      for(j = x_start; j < x_end; j ++){
         for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
            cost[i][j][l] /= (*pbatch_count);
            outfile << " " << cost[i][j][l];
            average += cost[i][j][l];
            variance += cost[i][j][l] * cost[i][j][l];
            count += 1;
            cost[i][j][l] = 0;
         }
      }
   }
   outfile << "\n";
   outfile.close();
   
   // Output average cost
   
   suffix = "/cost_validate_average.txt";
   filename = output_directory + suffix;
   if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
   else outfile.open (filename.c_str(),std::ios::app);
   if((*pdo_delete) == 1){
      outfile << "count epoch cputime average stddev\n";
   }
   average /= count;
   variance = sqrt(variance / count - average * average);
   outfile << training_batch_count << " " << epoch << " " << relative_cputime << " " << average << " " << variance << "\n";
   outfile.close();
   
   // Output normalized forecast-observation correlation for all pixels and output neurons
   
   average = variance = count = 0;
   suffix = "/correlation_validate_individual.txt";
   filename = output_directory + suffix;
   if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
   else outfile.open (filename.c_str(),std::ios::app);
   if((*pdo_delete) == 1){
      outfile << "count epoch cputime";
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
               outfile << " " << i << "," << j << "," << l;
            }
         }
      }
      outfile << "\n";
   }
   outfile << training_batch_count << " " << epoch << " " << relative_cputime;
   for(i = y_start; i < y_end; i ++){
      for(j = x_start; j < x_end; j ++){
         for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
            //forecast_observation_correlation[i][j][l] /= sqrt(squareaverage[i][j] * forecast_squareaverage[i][j][l]);
            forecast_observation_correlation[i][j][l] /= sqrt(observation_squareaverage[i][j][l] * forecast_squareaverage[i][j][l]);
            outfile << " " << forecast_observation_correlation[i][j][l];
            average += forecast_observation_correlation[i][j][l];
            variance += forecast_observation_correlation[i][j][l] * forecast_observation_correlation[i][j][l];
            count += 1;
            //forecast_observation_correlation[i][j][l] = 0;
            forecast_squareaverage[i][j][l] = 0;
            if(epoch > 0) observation_squareaverage[i][j][l] = 0;
         }
         //if(epoch > 0) squareaverage[i][j] = 0;
      }
   }
   outfile << "\n";
   outfile.close();
   
   // Output correlation time series
   
   std::string suffix2 = ".txt";
   std::stringstream ss;
   ss << epoch;
   suffix = "/correlation_validate_individual_epoch";
   filename = output_directory + suffix + ss.str() + suffix2;
   outfile.open (filename.c_str(),std::ios::out);
   for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            outfile << forecast_observation_correlation[i][j][l] << " ";
            forecast_observation_correlation[i][j][l] = 0;
         }
      }
      outfile << "\n";
   }
   outfile.close();
   
   // Output average normalized forecast-observation correlation
   
   suffix = "/correlation_validate_average.txt";
   filename = output_directory + suffix;
   if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
   else outfile.open (filename.c_str(),std::ios::app);
   if((*pdo_delete) == 1){
      outfile << "count epoch cputime average stddev\n";
   }
   average /= count;
   variance = sqrt(variance / count - average * average);
   outfile << training_batch_count << " " << epoch << " " << relative_cputime << " " << average << " " << variance << "\n";
   outfile.close();
   
   
   if(epoch==0){
   
      // Output Eulerian persistence cost for all pixels and output neurons
      
      average = variance = count = 0;
      suffix = "/cost_validate_persistent_individual.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         for(i = y_start; i < y_end; i ++){
            for(j = x_start; j < x_end; j ++){
               for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
                  outfile << i << "," << j << "," << l << " ";
               }
            }
         }
         outfile << "\n";
      }
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
               cost_persistent[i][j][l] /= (*pbatch_count);
               outfile << cost_persistent[i][j][l] << " ";
               average += cost_persistent[i][j][l];
               variance += cost_persistent[i][j][l] * cost_persistent[i][j][l];
               count += 1;
               cost_persistent[i][j][l] = 0;
            }
         }
      }
      outfile << "\n";
      outfile.close();
      
      // Output average cost for Eulerian persistence
      
      suffix = "/cost_validate_persistent_average.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         outfile << "average stddev\n";
      }
      average /= count;
      variance = sqrt(variance / count - average * average);
      outfile << average << " " << variance << "\n";
      outfile.close();
      
      // Output normalized persistent forecast-observation correlation for all pixels and output neurons
      
      average = variance = count = 0;
      suffix = "/correlation_validate_persistent_individual.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         for(i = y_start; i < y_end; i ++){
            for(j = x_start; j < x_end; j ++){
               for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
                  outfile << i << "," << j << "," << l<< " " ;
               }
            }
         }
         outfile << "\n";
      }
      for(i = y_start; i < y_end; i ++){
         for(j = x_start; j < x_end; j ++){
            for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
               //std::cout << "output " << i << " " << j << " " << l << " " <<forecast_observation_correlation_persistent[i][j][l] << " " << squareaverage[i][j] << "\n";
               //if(l ==1){
               //std::cout << "output " << i << " " << j << " " << l << " " <<forecast_observation_correlation_persistent[i][j][l] << " " << forecast_persistent_squareaverage[i][j][l] << " " << observation_squareaverage[i][j][l] << " " << forecast_observation_correlation_persistent[i][j][l] / sqrt(forecast_persistent_squareaverage[i][j][l] * observation_squareaverage[i][j][l]) << " " << (*pbatch_count) << "\n";
               //exit(1);
               //}
               //forecast_observation_correlation_persistent[i][j][l] /= squareaverage[i][j];
               forecast_observation_correlation_persistent[i][j][l] /= sqrt(forecast_persistent_squareaverage[i][j][l] * observation_squareaverage[i][j][l]);
               //forecast_observation_correlation_persistent[i][j][l] /= sqrt(squareaverage[i][j] * observation_squareaverage[i][j][l]);
               outfile << forecast_observation_correlation_persistent[i][j][l] << " ";
               average += forecast_observation_correlation_persistent[i][j][l];
               variance += forecast_observation_correlation_persistent[i][j][l] * forecast_observation_correlation_persistent[i][j][l];
               count += 1;
               //forecast_observation_correlation_persistent[i][j][l] = 0;
               forecast_persistent_squareaverage[i][j][l] = 0;
               observation_squareaverage[i][j][l] = 0;
            }
            //exit(1);
            //squareaverage[i][j] = 0;
         }
      }
      outfile << "\n";
      outfile.close();
      
      // Output persistent correlation time series
      
      suffix = "/correlation_validate_persistent_individual_timeseries.txt";
      filename = output_directory + suffix;
      outfile.open (filename.c_str(),std::ios::out);
      for(l = 0; l < number_neurons[number_total_layers - 1]; l ++){
         for(i = y_start; i < y_end; i ++){
            for(j = x_start; j < x_end; j ++){
               outfile << forecast_observation_correlation_persistent[i][j][l] << " ";
               forecast_observation_correlation_persistent[i][j][l] = 0;
            }
         }
         outfile << "\n";
      }
      outfile.close();
      
      // Output average normalized forecast-observation correlation
      
      suffix = "/correlation_validate_persistent_average.txt";
      filename = output_directory + suffix;
      if((*pdo_delete) == 1) outfile.open (filename.c_str(),std::ios::out);
      else outfile.open (filename.c_str(),std::ios::app);
      if((*pdo_delete) == 1){
         outfile << "average stddev\n";
      }
      average /= count;
      variance = sqrt(variance / count - average * average);
      outfile << average << " " << variance << "\n";
      outfile.close();
   }
   *pbatch_count = 0;
}

double sigmoid_function(double z)
{
   return 1/(1 + exp(-z));
}

double sigmoid_function_derivative(double z)
{
   return exp(-z)/pow(1 + exp(-z), 2);
}

double inverse_sigmoid_function(double z)
{
   return -log(1/z - 1);
}

double inverse_sigmoid_function_derivative(double z)
{
   return 1/(z + z*z);
}

double quadratic_function(double observation, double forecast)
{
   return pow(forecast - observation, 2)/2;
}

double quadratic_function_derivative(double observation, double forecast)
{
   return forecast - observation;
}

double cross_entropy_function(double observation, double forecast)
{
   return -observation * log(forecast) + (observation - 1) * log(1 - forecast);
}

double cross_entropy_function_derivative(double observation, double forecast)
{
   if((forecast==0)||(forecast==1)){
      std::cout << "forecast = " << forecast << " in cross_entropy_function_derivative!\n";
      exit(1);
   }
   return -observation / forecast + (1 - observation) / (1 - forecast);
}

