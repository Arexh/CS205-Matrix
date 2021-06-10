#include <opencv2/opencv.hpp>
#include <iostream>
#include "Matrix.h"
#include "SparseMatrix.h"

using namespace std;

int main()
{
    // SparseMatrix<double> b({{1, 2},
    //                         {-1, -3}});
    // Matrix<double> cc({{1, 2},
    //                 {-1, -3}});
    // SparseMatrix<double> pp({{9, 23, 10},
    //                         {2, 1, 0},
    //                         {0, 7, 5}});
    // SparseMatrix<double> bb({{5, 75, 5},
    //                         {0, 1, 0},
    //                         {25, 25, 2}});
    // // SparseMatrix<double> a({{0.5, 0.75, 0.5},
    // //                         {1.0, 0.5, 0.75},
    // //                         {0.25, 0.25, 0.25}});
    // // SparseMatrix<double> b(a);
    // // b.printMatrix();
    // b.printMatrix();
    // SparseMatrix<double> e = b;
    // SparseMatrix<double> oo = e.Inverse();
    // oo.printMatrix();
    // cout << oo.col_mean(0) << endl;
    // cc.Inverse().printMatrix();

    SparseMatrix<int> a;
    a.printMatrix();
    // SparseMatrix<double> r = aResult.first * aResult.second;
    // r.printMatrix();
    return 0;
}