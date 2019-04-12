#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "stb/image.h"
#include "stb/image_write.h"


int main(int argc, char** argv) {

    // parse input parameters
    if(argc != 3) {
      printf("Usage: %s input.png output.png\n", argv[0]);
      return EXIT_FAILURE;
    }

    char* input_file_name = argv[1];
    char* output_file_name = argv[2];


    // load input file
    printf("Loading input file %s ..\n", input_file_name);
    int width, height, components;
    unsigned char *data = stbi_load(input_file_name, &width, &height, &components, 0);
    printf("Loaded image of size %dx%d with %d components.\n", width,height,components);

    // start the timer
    double start_time = now();

    // ------ Analyze Image ------

    // compute min/max/avg of each component
    unsigned char min_val[components];
    unsigned char max_val[components];
    unsigned char avg_val[components];

    // an auxilary array for computing the average
    unsigned long long sum[components];

    // initialize
    for(int c = 0; c<components; c++) {
      min_val[c] = 255;
      max_val[c] = 0;
      sum[c] = 0;
    }

    // compute min/max/sum
    for(int x=0; x<width; ++x) {
      for(int y=0; y<height; ++y) {
        for(int c=0; c<components; ++c) {
          unsigned char val = data[c + x*components + y*width*components];
          if (val < min_val[c]) min_val[c] = val;
          if (val > max_val[c]) max_val[c] = val;
          sum[c] += val;
        }
      }
    }

    // compute average and multiplicative factors
    float min_fac[components];
    float max_fac[components];
    for(int c=0; c<components; ++c) {
      avg_val[c] = sum[c]/((unsigned long long)width*height);
      min_fac[c] = (float)avg_val[c] / (float)(avg_val[c] - min_val[c]);
      max_fac[c] = (255.0f-(float)avg_val[c]) / (float)(max_val[c] - avg_val[c]);
      printf("\tComponent %1u: %3u / %3u / %3u * %3.2f / %3.2f\n", c, min_val[c], avg_val[c], max_val[c], min_fac[c], max_fac[c]);
    }

    // ------ Adjust Image ------

    for(int x=0; x<width; ++x) {
      for(int y=0; y<height; ++y) {
        for(int c=0; c<components; ++c) {
          int index = c + x*components + y*width*components;
          unsigned char val = data[index];
          float v = (float)(val - avg_val[c]);
          v *= (val < avg_val[c]) ? min_fac[c] : max_fac[c];
          data[index] = (unsigned char)(v + avg_val[c]);
        }
      }
    }

    printf("Done, took %.1f ms\n", (now() - start_time)*1000.0);

    // ------ Store Image ------

    printf("Writing output image %s ...\n", output_file_name);
    stbi_write_png(output_file_name,width,height,components,data,width*components);
    stbi_image_free(data);

    printf("Done!\n");

    // done
    return EXIT_SUCCESS;
}
