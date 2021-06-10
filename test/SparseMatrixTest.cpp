#include <opencv2/opencv.hpp>
#include "gtest/gtest.h"
#include "SparseMatrix.h"
#include <complex>

using namespace std;

template <typename T>
void assertSparseMatrixEqual(SparseMatrix<T>& one, SparseMatrix<T>& two) {
    ASSERT_EQ(one.m_row, two.m_row);
    ASSERT_EQ(one.m_col, two.m_col);
    for (int i = 0; i < one.m_row; i++) {
        for (int j = 0; j < one.m_col; j++) {
            ASSERT_EQ(one[i][j], two[i][j]);
        }
    }
}

TEST(SparseMatrixTest, SparseMatrixAdd)
{
    SparseMatrix<int> a({{1, 2, 3, 4},
                   {5, 6, 7, 8}});
    SparseMatrix<int> b({{8, 7, 6, 5},
                   {4, 3, 2, 1}});
    SparseMatrix<int> c = a + b;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            ASSERT_EQ(c[i][j], 9);
        }
    }

    SparseMatrix<complex<double>> aa({{(1.0 + 2i), (3.0 + 4i), (5.0 + 6i), (7.0 + 8i)},
                                {(8.0 + 7i), (6.0 + 5i), (4.0 + 3i), (2.0 + 1i)}});
    SparseMatrix<complex<double>> bb({{(8.0 + 7i), (6.0 + 5i), (4.0 + 3i), (2.0 + 1i)},
                                {(1.0 + 2i), (3.0 + 4i), (5.0 + 6i), (7.0 + 8i)}});
    SparseMatrix<complex<double>> cc = aa + bb;
    for (int i = 0; i < cc.m_row; i++)
    {
        for (int j = 0; j < cc.m_row; j++)
        {
            ASSERT_EQ(cc[i][j], 9.0 + 9i);
        }
    }
}

TEST(SparseMatrixTest, SparseMatrixMinus)
{
    SparseMatrix<int> a({{1, 2, 3, 4},
                   {5, 6, 7, 8}});
    SparseMatrix<int> b({{2, 3, 4, 5},
                   {6, 7, 8, 9}});
    SparseMatrix<int> c = a - b;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            ASSERT_EQ(c[i][j], -1);
        }
    }

    SparseMatrix<complex<double>> aa({{(1.0 + 2i), (3.0 + 4i)},
                                {(5.0 + 6i), (7.0 + 8i)}});
    SparseMatrix<complex<double>> bb({{(2.0 + 3i), (4.0 + 5i)},
                                {(6.0 + 7i), (8.0 + 9i)}});
    SparseMatrix<complex<double>> cc = aa - bb;
    for (int i = 0; i < cc.m_row; i++)
    {
        for (int j = 0; j < cc.m_row; j++)
        {
            ASSERT_EQ(-1.0 - 1i, (aa - bb)[i][j]);
        }
    }
}

TEST(SparseMatrixTest, SparseMatrixMultiply)
{
    SparseMatrix<int> a({{1, 1},
                   {2, 2}});
    SparseMatrix<int> b({{-2, -2},
                   {1, 1}});
    SparseMatrix<int> c({{-1, -1},
                   {-2, -2}});
    for (int i = 0; i < c.m_row; i++)
    {
        for (int j = 0; j < c.m_row; j++)
        {
            ASSERT_EQ(c[i][j], (a * b)[i][j]);
        }
    }

    SparseMatrix<complex<int>> aa({{complex<int>(1, 1), complex<int>(1, 1)},
                             {complex<int>(2, 2), complex<int>(2, 2)}});
    SparseMatrix<complex<int>> bb({{complex<int>(-2, -2), complex<int>(-2, -2)},
                             {complex<int>(1, 1), complex<int>(1, 1)}});
    (aa * bb).printMatrix();
    SparseMatrix<complex<int>> mul({{complex<int>(0, -2), complex<int>(0, -2)},
                              {complex<int>(0, -4), complex<int>(0, -4)}});
    for (int i = 0; i < mul.m_row; i++)
    {
        for (int j = 0; j < mul.m_row; j++)
        {
            ASSERT_EQ(mul[i][j], (aa * bb)[i][j]);
        }
    }
}

TEST(SparseMatrixTest, DefautConstructorTest)
{
    SparseMatrix<int> a;
    a.printMatrix();
    SparseMatrix<complex<double>> *b = new SparseMatrix<complex<double>>;
    b->printMatrix();
    delete b;
    SparseMatrix<double> c(3, 3);
    c.printMatrix();
    SparseMatrix<complex<int>> d(5, 5);
    d.printMatrix();
    SparseMatrix<complex<double>> *e = new SparseMatrix<complex<double>>(4, 4);
    e->printMatrix();
    delete e;
}

TEST(SparseMatrixTest, CopyConstructorTest)
{
    int i, j;
    SparseMatrix<int> a({{1, 2, 3, 4},
                   {5, 6, 7, 8}});
    SparseMatrix<int> aa(a);
    for (i = 0; i < a.m_row; i++)
        for (j = 0; j < a.m_col; j++)
            ASSERT_EQ(aa[i][j], a[i][j]);

    ASSERT_NE(&a, &aa);
    //////////////////
    SparseMatrix<complex<double>> b({{(1.0 + 2i), (3.0 + 4i), (5.0 + 6i), (7.0 + 8i)},
                               {(8.0 + 7i), (6.0 + 5i), (4.0 + 3i), (2.0 + 1i)}});
    SparseMatrix<complex<double>> bb(b);
    for (i = 0; i < b.m_row; i++)
        for (j = 0; j < b.m_col; j++)
            ASSERT_EQ(bb[i][j], b[i][j]);

    ASSERT_NE(&b, &bb);
    /////////////////
    SparseMatrix<complex<int>> c({{complex<int>(1, 2), complex<int>(3, 4)},
                            {complex<int>(5, 6), complex<int>(7, 8)}});
    SparseMatrix<complex<int>> cc(c);
    for (i = 0; i < c.m_row; i++)
        for (j = 0; j < c.m_col; j++)
            ASSERT_EQ(cc[i][j], c[i][j]);

    ASSERT_NE(&c, &cc);
}

TEST(SparseMatrixTest, row_col_negative)
{
    SparseMatrix<complex<int>> a(-3, -1);
    SparseMatrix<int> b(0, -5);
    SparseMatrix<double> c(-5, 0);
    SparseMatrix<complex<float>> d(1, 0);
    //锟斤拷飧达拷乒锟斤拷锟斤拷锟�
    SparseMatrix<complex<int>> aa(a);
    SparseMatrix<int> bb(b);

    ASSERT_GE(a.m_row, 0) << "m_row is negetive";
    ASSERT_GE(a.m_col, 0) << "m_col is negetive";
    ASSERT_GE(b.m_row, 0) << "m_row is negetive";
    ASSERT_GE(b.m_col, 0) << "m_col is negetive";
    ASSERT_GE(c.m_row, 0) << "m_row is negetive";
    ASSERT_GE(c.m_col, 0) << "m_col is negetive";
    ASSERT_GE(d.m_row, 0) << "m_row is negetive";
    ASSERT_GE(d.m_col, 0) << "m_col is negetive";
    ASSERT_GE(aa.m_row, 0) << "m_row is negetive";
    ASSERT_GE(aa.m_col, 0) << "m_col is negetive";
    ASSERT_GE(bb.m_row, 0) << "m_row is negetive";
    ASSERT_GE(bb.m_col, 0) << "m_col is negetive";
}

TEST(SparseMatrixTest, AssignmentOperator)
{
    int i, j;
    SparseMatrix<complex<int>> a({{complex<int>(1, 2), complex<int>(3, 4)},
                            {complex<int>(5, 6), complex<int>(7, 8)}});
    SparseMatrix<complex<int>> aa = a;
    for (i = 0; i < a.m_row; i++)
        for (j = 0; j < a.m_col; j++)
            ASSERT_EQ(aa[i][j], a[i][j]);
    ASSERT_NE(&a, &aa);

    SparseMatrix<double> b({{1, 2, 3, 4},
                      {5, 6, 7, 8}});
    SparseMatrix<double> bb = b;
    for (i = 0; i < b.m_row; i++)
        for (j = 0; j < b.m_col; j++)
            ASSERT_EQ(bb[i][j], b[i][j]);
    ASSERT_NE(&b, &bb);
}

TEST(SparseMatrixTest, ComplexConjugate)
{
    int i, j;
    SparseMatrix<complex<int>> c({{complex<int>(1, 2), complex<int>(3, 4)},
                            {complex<int>(5, 6), complex<int>(7, 8)}});
    SparseMatrix<complex<int>> cc({{complex<int>(1, -2), complex<int>(3, -4)},
                             {complex<int>(5, -6), complex<int>(7, -8)}});
    for (i = 0; i < cc.m_row; i++)
        for (j = 0; j < cc.m_col; j++)
            ASSERT_EQ(conj(cc[i][j]), c[i][j]);
}

TEST(SparseMatrixTest, trace)
{
    SparseMatrix<double> a({{1, 1, 1, 1},
                      {2, 2, 2, 2},
                      {1, 2, 3, 4},
                      {1, 5, 6, 9}});
    ASSERT_EQ(15, a.trace());

    SparseMatrix<double> b({{5.5, 2}});
    ASSERT_EQ(0, b.trace());

    SparseMatrix<complex<double>> d({{complex<double>(1, 1), complex<double>(1, 1)},
                               {complex<double>(1, -1), complex<double>(1, 1)}});
    ASSERT_EQ(complex<double>(2, 2), d.trace());
}

TEST(SparseMatrixTest, determinant)
{
    SparseMatrix<double> a({{1, 1, 1, 1},
                      {2, 2, 2, 2},
                      {1, 2, 3, 4},
                      {1, 5, 6, 9}});
    ASSERT_EQ(0, a.determinant());

    vector<vector<double>> v{{5.5}};
    SparseMatrix<double> b(v);
    ASSERT_EQ(5.5, b.determinant());

    SparseMatrix<double> c({{1, 1},
                      {2, 3}});
    ASSERT_EQ(1, c.determinant());

    SparseMatrix<complex<double>> d({{complex<double>(1, 1), complex<double>(1, 1)},
                               {complex<double>(1, -1), complex<double>(1, 1)}});
    ASSERT_EQ(complex<double>(-2, 2), d.determinant());
}

TEST(SparseMatrixTest, Inverse)
{
    vector<vector<double>> v{{4}};
    SparseMatrix<double> b(v);
    (b.Inverse()).printMatrix();
    (b.Inverse() * b).printMatrix();

    SparseMatrix<double> c({{1, -1},
                      {1, 1}});
    (c.Inverse()).printMatrix();
    (c.Inverse() * c).printMatrix();

    SparseMatrix<double> d({{1, 0, 0},
                      {0, 2, 0},
                      {0, 0, 4}});
    (d.Inverse()).printMatrix();
    (d.Inverse() * d).printMatrix();

    SparseMatrix<complex<double>> e({{complex<double>(1, 1), complex<double>(1, 1)},
                               {complex<double>(1, -1), complex<double>(1, 1)}});
    e.Inverse().printMatrix();
    (e.Inverse() * e).printMatrix();
}

TEST(SparseMatrixTest, Max_Min_Sum_Mean_Test)
{
    SparseMatrix<double> a({{5, 2, 1, 1},
                      {2, 3, 4, 5},
                      {1, 2, 3, 4},
                      {1, 5, 6, 9}});
    double sum = (double)(5 + 2 + 1 + 1 + 2 + 3 + 4 + 5 + 1 + 2 + 3 + 4 + 1 + 5 + 6 + 9);
    double mean = ((5 + 2 + 1 + 1 + 2 + 3 + 4 + 5 + 1 + 2 + 3 + 4 + 1 + 5 + 6 + 9) / 16.0);
    ASSERT_EQ(sum, a.sum());
    ASSERT_EQ(9, a.max());
    ASSERT_EQ(1, a.min());
    ASSERT_EQ(mean, a.mean());

    double row_sum = (double)(5 + 2 + 1 + 1);
    double row_mean = (double)(5 + 2 + 1 + 1) / 4.0;
    ASSERT_EQ(row_sum, a.row_sum(0));
    ASSERT_EQ(5, a.row_max(0));
    ASSERT_EQ(1, a.row_min(0));
    ASSERT_EQ(row_mean, a.row_mean(0));

    double col_sum = (double)(2 + 3 + 2 + 5);
    double col_mean = (double)(2 + 3 + 2 + 5) / 4.0;
    ASSERT_EQ(col_sum, a.col_sum(1));
    ASSERT_EQ(5, a.col_max(1));
    ASSERT_EQ(2, a.col_min(1));
    ASSERT_EQ(col_mean, a.col_mean(1));
}

TEST(SparseMatrixTest, reshape)
{
    SparseMatrix<double> a({{5, 2, 1, 1, 2},
                      {2, 3, 4, 5, 3},
                      {1, 2, 3, 4, 4},
                      {1, 5, 6, 9, 5}});
    a.reshape(2, 10).printMatrix();
    a.reshape(3, 4).printMatrix();
}

TEST(SparseMatrixTest, slice)
{
    SparseMatrix<double> a({{5, 2, 1, 1, 2},
                      {2, 3, 4, 5, 3},
                      {1, 2, 3, 4, 4},
                      {1, 5, 6, 9, 5}});
    a.slice(1, 3, 2, 4).printMatrix();
    a.slice(-1, -1, 5, 5).printMatrix();
    a.slice(2, 2, 1, 4).printMatrix(); 
}

TEST(SparseMatrixTest, QR_decomposition)
{
    // 3 x 3 SparseMatrix
    SparseMatrix<double> a({{0.5, 0.75, 0.5},
                      {1.0, 0.5, 0.75},
                      {0.25, 0.25, 0.25}});
    auto aResult = a.QR_decomposition();
    ASSERT_TRUE(aResult.first * aResult.second == a);

    // 4 x 4 SparseMatrix
    SparseMatrix<double> b({{1.0, 5.0, 3.0, 4.0},
                      {7.0, 8.0, 2.0, 9.0},
                      {7.0, 3.0, 2.0, 1.0},
                      {9.0, 3.0, 5.0, 7.0}});
    auto bResult = b.QR_decomposition();
    ASSERT_TRUE((bResult.first * bResult.second) == b);

    // 5 x 5 SparseMatrix
    SparseMatrix<double> c({{2, 6, 4, 6, 8},
                      {6, 7, 9, 7, 9},
                      {2, 3, 6, 3, 5},
                      {6, 1, 1, 5, 5},
                      {3, 5, 6, 5, 6}});
    auto cResult = c.QR_decomposition();
    ASSERT_TRUE((cResult.first * cResult.second) == c);

    // 5 x 5 SparseMatrix
    SparseMatrix<double> d({{8.662634278267483, 2.3440981169711796, 3.414158790068152, 9.819959485632891, 9.812414578216162},
                      {4.8096369839436495, 7.743133259609277, 9.871217856632036, 7.100783013043249, 8.127838524397976},
                      {1.3468248609110365, 1.3120774834063536, 9.607366488550678, 2.852679282078192, 8.087038227451359},
                      {7.556075051454403, 5.80117852857823, 3.550189544341768, 3.7807047754393994, 7.934423413357392},
                      {2.866445996919499, 7.125441061546031, 4.53141730712106, 4.297092147605687, 2.5126585000174146}});
    auto dResult = d.QR_decomposition();
    ASSERT_TRUE((dResult.first * dResult.second) == d);
}

TEST(SparseMatrixTest, From_OpenCV_Mat)
{
    // 6 x 2 zeros SparseMatrix
    {
        const int aRow = 6, aCol = 2;
        cv::Mat a = cv::Mat::zeros(aRow, aCol, CV_8UC1);
        SparseMatrix<uchar> aMatix = SparseMatrix<uchar>::fromOpenCV(a);
        ASSERT_EQ(aMatix.m_row, aRow);
        ASSERT_EQ(aMatix.m_col, aCol);
        for (int i = 0; i < aRow; i++) {
            for (int j = 0; j < aCol; j++) {
                ASSERT_EQ(aMatix[i][j], a.at<uchar>(i, j));
                ASSERT_EQ(aMatix[i][j], 0);
            }
        }
    }
    // 4 x 4 ones SparseMatrix
    {
        const int aRow = 4, aCol = 4;
        cv::Mat a = cv::Mat::ones(aRow, aCol, CV_8UC1);
        SparseMatrix<uchar> aMatix = SparseMatrix<uchar>::fromOpenCV(a);
        ASSERT_EQ(aMatix.m_row, aRow);
        ASSERT_EQ(aMatix.m_col, aCol);
        for (int i = 0; i < aRow; i++) {
            for (int j = 0; j < aCol; j++) {
                ASSERT_EQ(aMatix[i][j], a.at<uchar>(i, j));
                ASSERT_EQ(aMatix[i][j], 1);
            }
        }
    }
    // 10 x 10 float random value SparseMatrix
    {
        const int aRow = 10, aCol = 10;
        double low = -500.0;
        double high = +500.0;
        cv::Mat a = cv::Mat(aRow, aCol, CV_32F);
        randu(a, cv::Scalar(low), cv::Scalar(high));
        SparseMatrix<float> aMatix = SparseMatrix<float>::fromOpenCV(a);
        ASSERT_EQ(aMatix.m_row, aRow);
        ASSERT_EQ(aMatix.m_col, aCol);
        for (int i = 0; i < aRow; i++) {
            for (int j = 0; j < aCol; j++) {
                ASSERT_EQ(aMatix[i][j], a.at<float>(i, j));
            }
        }
    }
}

TEST(SparseMatrixTest, To_OpenCV_Mat)
{
    // 5 x 10 int SparseMatrix
    {
        const int aRow = 5, aCol = 10;
        SparseMatrix<int> aSparseMatrix(aRow, aCol);
        int cnt = 0;
        for (int i = 0; i < aRow; i++) {
            for (int j = 0; j < aCol; j++) {
                aSparseMatrix[i][j] = cnt++;
            }
        }
        cv::Mat* cvMat = aSparseMatrix.toOpenCVMat(CV_32S);
        SparseMatrix<int> bSparseMatrix = SparseMatrix<int>::fromOpenCV(*cvMat);
        for (int i = 0; i < aRow; i++) {
            for (int j = 0; j < aCol; j++) {
                ASSERT_EQ(aSparseMatrix[i][j], bSparseMatrix[i][j]);
            }
        }
    }
    // 20 x 10 float random value SparseMatrix
    {
        const int aRow = 20, aCol = 10;
        double low = -500.0;
        double high = +500.0;
        cv::Mat a = cv::Mat(aRow, aCol, CV_32F);
        randu(a, cv::Scalar(low), cv::Scalar(high));
        SparseMatrix<float> aSparseMatrix = SparseMatrix<float>::fromOpenCV(a);
        cv::Mat* cvMat = aSparseMatrix.toOpenCVMat(CV_32S);
        SparseMatrix<float> bSparseMatrix = SparseMatrix<float>::fromOpenCV(*cvMat);
        for (int i = 0; i < aRow; i++) {
            for (int j = 0; j < aCol; j++) {
                ASSERT_EQ(aSparseMatrix[i][j], bSparseMatrix[i][j]);
            }
        }
    }
}

TEST(SparseMatrixTest, conv2D)
{
    SparseMatrix<int> inputDim4({
        {5, 7, 1, 2},
        {4, 3, 9, 0},
        {8, 7, 6, 1},
        {4, 2, 0, 7}
    });
    SparseMatrix<int> kernelDim3({
        {1, 0, 1},
        {1, 1, 1},
        {2, 1, 0}
    });
    // 4 x 4 input, 3 x 3 kernel, 1 stride, valid
    {
        
        SparseMatrix<int> result = SparseMatrix<int>::conv2D(inputDim4, kernelDim3, 1, false);
        SparseMatrix<int> expected({
            {45, 41},
            {44, 21}
        });
        assertSparseMatrixEqual(result, expected);
    }
    // 4 x 4 input, 3 x 3 kernel, 1 stride, same
    {
        
        SparseMatrix<int> result = SparseMatrix<int>::conv2D(inputDim4, kernelDim3, 1, true);
        SparseMatrix<int> expected({
            {16, 24, 25, 21},
            {22, 45, 41, 23},
            {22, 44, 21, 23},
            {13, 20, 17, 13}
        });
        assertSparseMatrixEqual(result, expected);
    }
    // 4 x 4 input, 3 x 3 kernel, 2 stride, valid
    {
        
        SparseMatrix<int> result = SparseMatrix<int>::conv2D(inputDim4, kernelDim3, 2, false);
        vector<vector<int>> v {
            {45}
        };
        SparseMatrix<int> expected(v);
        assertSparseMatrixEqual(result, expected);
    }
    // 4 x 4 input, 3 x 3 kernel, 2 stride, same
    {
        
        SparseMatrix<int> result = SparseMatrix<int>::conv2D(inputDim4, kernelDim3, 2, true);
        vector<vector<int>> v {
            {16, 25},
            {22, 21}
        };
        SparseMatrix<int> expected(v);
        assertSparseMatrixEqual(result, expected);
    }
}