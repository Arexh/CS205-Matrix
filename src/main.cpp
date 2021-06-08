#include <iostream>
#include "Matrix.h"

using namespace std;

int main()
{
   Matrix<double> a({{8.662634278267483, 2.3440981169711796, 3.414158790068152, 9.819959485632891, 9.812414578216162},
                      {4.8096369839436495, 7.743133259609277, 9.871217856632036, 7.100783013043249, 8.127838524397976},
                      {1.3468248609110365, 1.3120774834063536, 9.607366488550678, 2.852679282078192, 8.087038227451359},
                      {7.556075051454403, 5.80117852857823, 3.550189544341768, 3.7807047754393994, 7.934423413357392},
                      {2.866445996919499, 7.125441061546031, 4.53141730712106, 4.297092147605687, 2.5126585000174146}});
    pair<Matrix<double>, Matrix<double>> qr = a.QR_decomposition();
    cout << "Ori matrix: " << endl;
    a.printMatrix();
    cout << "Q matrix: " << endl;
    qr.first.printMatrix();
    cout << "R matrix: " << endl;
    qr.second.printMatrix();
    Matrix<double> qrm = qr.first * qr.second;
    cout << "QR matrix: " << endl;
    qrm.printMatrix();
    cout << "Ori == QR: " << (qrm == a) << endl;
    return 0;
}