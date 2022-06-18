#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>

#include "opencv2/core.hpp"

#include "ostromoukhov.hpp"

Ostromoukhov::Ostromoukhov(){
    for (int i=0; i<256; i++){
        coefs_tab[i*4] /= coefs_tab[i*4+3];
        coefs_tab[i*4+1] /= coefs_tab[i*4+3];
        coefs_tab[i*4+2] /= coefs_tab[i*4+3];
    }
}

cv::Mat Ostromoukhov::process(cv::Mat& ct){
    assert(ct.type() == CV_8UC1);
    cv::Mat ht(ct.size(), CV_8UC1);

    const double threshold = 127.5;
    double* carry_line_0 = (double*)calloc(ct.cols+2, sizeof(double));
    double* carry_line_1 = (double*)calloc(ct.cols+2, sizeof(double));

    for (int h=0; h<ht.rows; h++){
        int w_start, w_stop, w_step;
        if ((h & 0x1) == 0){
            // to right
            w_start = 0; w_stop = ht.cols; w_step = 1;
        }else{
            // to left
            w_start = ht.cols - 1; w_stop = -1; w_step = -1;
        }

        for (int w=w_start; w!=w_stop; w+=w_step){
            uint8_t ct_value = ct.at<uint8_t>(h, w);
            double ct_absorb = (double)ct_value + carry_line_0[w+1];
            double error;
            if (ct_absorb > threshold){
                error = ct_absorb - 255.;
                ht.at<uint8_t>(h, w) = 255;
            }else{
                error = ct_absorb;
                ht.at<uint8_t>(h, w) = 0;
            }

            double d10 = coefs_tab[ct_value*4];
            double d11 = coefs_tab[ct_value*4+1];
            double d01 = coefs_tab[ct_value*4+2];

            carry_line_0[w+1+w_step] += error * d10;
            carry_line_1[w+1] += error * d01;
            carry_line_1[w+1-w_step] += error * d11;
        }
        std::swap(carry_line_0, carry_line_1);
        memset(carry_line_1, 0, (ht.cols+2)*sizeof(double));
    }

    free(carry_line_0);
    free(carry_line_1);

    return ht;
}
