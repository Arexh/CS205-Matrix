#include <opencv2/opencv.hpp>
#include <iostream>
#include "Matrix.h"
#include "SparseMatrix.h"

using namespace std;

int main()
{
    SparseMatrix<double> a({{0.5, 0.75, 0.5},
                      {1.0, 0.5, 0.75},
                      {0.25, 0.25, 0.25}});
    SparseMatrix<double> b(a);
    // auto aResult = a.QR_decomposition();
    // SparseMatrix<double> r = aResult.first * aResult.second;
    // r.printMatrix();
    return 0;
}