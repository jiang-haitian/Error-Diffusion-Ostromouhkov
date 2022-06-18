#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"

#include "ostromoukhov.hpp"

int main(int argc, char* argv[]){
    if (argc != 2 && argc != 3){
        std::cout << "Usage:" << std::endl
            << "error_diffusion INPUT_IMG [OUTPUT_IMG]" << std::endl;
        return -1;
    }

    cv::Mat ct = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    Ostromoukhov halftoner;

    cv::Mat ht = halftoner.process(ct);

    if (argc == 3){
        cv::imwrite(argv[2], ht);
    }else{
        cv::imshow("ht", ht); 
        cv::waitKey(0);
    }

    return 0;
}
