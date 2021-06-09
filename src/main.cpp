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

    // Matrix<double> a({{0.5, 0.75, 0.5},
    //                 {1.0, 0.5, 0.75},
    //                 {0.25, 0.25, 0.25}});
    // Matrix<double> input(a);
    // int m_row = 3;
    // int m_col = 3;
    // vector<Matrix<double>> plist;
    // for (int j = 0; j < m_row - 1; j++)
    // {
    //     cout << "HERE1" << endl;
    //     Matrix<double> a1(1, m_row - j);
    //     Matrix<double> b1(1, m_row - j);

    //     for (int i = j; i < m_row; i++)
    //     {
    //         a1[0][i - j] = input[i][j];
    //         b1[0][i - j] = static_cast<double>(0.0);
    //     }
    //     b1[0][0] = static_cast<double>(1.0);

    //     double a1norm = a1.norm();

    //     double sgn = -1;
    //     if (a1[0][0] < static_cast<double>(0.0))
    //     {
    //         sgn = 1;
    //     }

    //     Matrix<double> temp = b1 * sgn * a1norm;
    //     Matrix<double> u = a1 - temp;
    //     Matrix<double> n = u.normalized();
    //     Matrix<double> nTrans = n.Transposition();
    //     Matrix<double> I(m_row - j, m_row - j);
    //     I.SetIdentity();
    //     Matrix<double> temp1 = n * static_cast<double>(2.0);
    //     Matrix<double> temp2 = nTrans * temp1;
    //     Matrix<double> Ptemp = I - temp2;

    //     Matrix<double> P(m_row, m_col);
    //     P.SetIdentity();

    //     for (int x = j; x < m_row; x++)
    //     {
    //         for (int y = j; y < m_col; y++)
    //         {
    //             P[x][y] = Ptemp[x - j][y - j];
    //         }
    //     }

    //     plist.push_back(P);
    //     input = P * input;
    // }

    // Matrix<double> qMat = plist[0];
    // for (int i = 1; i < m_row - 1; i++)
    // {
    //     Matrix<double> temp3 = plist[i].Transposition();
    //     qMat = qMat * temp3;
    // }

    // int numElements = plist.size();
    // Matrix<double> rMat = plist[numElements - 1];
    // for (int i = (numElements - 2); i >= 0; i--)
    // {
    //     rMat = rMat * plist[i];
    // }
    // rMat = rMat * a;
    // ASSERT_TRUE(aResult.first * aResult.second == a);
    Matrix<double> a({{0.5, 0.75, 0.5},
                      {1.0, 0.5, 0.75},
                      {0.25, 0.25, 0.25}});
    auto aResult = a.QR_decomposition();
    Matrix<double> r = aResult.first * aResult.second;
    r.printMatrix();
    return 0;
}