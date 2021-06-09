#include <opencv2/opencv.hpp>
#include <iostream>
#include "Matrix.h"

using namespace std;

int main()
{
    // Matrix<double> a({{6.000, 5.500, -1.000 },
    //                   {5.500, 1.000, -2.000},
    //                   {-1.000, -2.000, -3.000}});
    // // double *eigenvalues = a.eigenvalues();
    // // // numpy: 9.6026536, -1.55750606, -4.04514754
    // // cout << "Eigenvalues: " << endl;
    // // for (int i = 0; i < 3; i++)
    // // {
    // //     cout << *(eigenvalues + i) << endl;
    // // }
    // // Matrix<double>* eigenvectors = a.eigenvectors();
    // // cout << "Eigenvectors: " << endl;
    // // for (int i = 0; i < 3; i++)
    // // {
    // //     (*(eigenvectors + i)).printMatrix();
    // // }

    // auto result = a.eigenValueAndEigenVector();
    // cout << "Eigenvalue: " << endl;
    // cout << result.first << endl;
    // cout << "Eigenvector: " << endl;
    // result.second.printMatrix();

    Matrix<float> a(5, 10);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 10; j++) {
            a[i][j] = i * j;
        }
    }
    cv::Mat* cvMat = a.toOpenCVMat(CV_32F);
    cout << *cvMat << endl;
    return 0;
}