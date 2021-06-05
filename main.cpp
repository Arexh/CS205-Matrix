#include <iostream>
#include "Matrix.h"
#include "Complex.h"
using namespace std;
int main()
{
    int row, col, row1, col1;
    cin >> row >> col >> row1 >> col1;
    Matrix<double> a(row, col);
    Matrix<Complex> b(row1, col1);
    //Matrix<int> b(a);
    for (int i = 0; i < a.m_row; i++) {
        for (int j = 0; j < a.m_col; j++) {
            // a[i][j] =i+j;
            ////a[i][j] = Complex(i, j);
             cin>>a[i][j];
        }
    }

     for (int i = 0; i < b.m_row; i++) {
         for (int j = 0; j < b.m_col; j++) {
             b[i][j] = Complex(i,j);
             // b[i][j] = Complex(i, j);
             //cin>>b[i][j];
         }
     }


     //(a * b).printMatrix();
    //cout << (a * b)[0][0] << endl;
    cout << "\n";

    // a.printMatrix();
    // cout << "\n";

    // b.printMatrix();

    // // a = b;
     //a.printMatrix();
    //a.LU_factor_U().printMatrix();
    //a.LU_factor_L().printMatrix();
    //((a.LU_factor_L())*(a.LU_factor_U())).printMatrix();

    // (~a).printMatrix();
     //Complex tt(a.min(1, 0));
    Matrix<double> l(a.LDU_factor_L());
    Matrix<double> d(a.LDU_factor_D());
    Matrix<double> u(a.LDU_factor_U());
    l.printMatrix();
    d.printMatrix();
    u.printMatrix();
    (l *d* u).printMatrix();


    return 0;
}