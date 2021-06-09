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

    Matrix<double> a({{0.5, 0.75, 0.5},
                    {1.0, 0.5, 0.75},
                    {0.25, 0.25, 0.25}});
    auto aResult = a.QR_decomposition();
    // ASSERT_TRUE(aResult.first * aResult.second == a);
    return 0;
}