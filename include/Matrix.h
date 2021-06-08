#pragma once
#include <vector>
#include <iostream>
#include <cassert>
#include <iomanip>
#include <math.h>
#include <complex>
#include <typeinfo>
#include <random>

using namespace std;
using std::setw;

const double EQ_THRESHOLD = 1e-10;

template <class T>
class Matrix : public std::vector<std::vector<T>>
{
public:
    int m_row;
    int m_col;
    explicit Matrix() :Matrix(0, 0) {};
    Matrix(int row, int col);
    Matrix(vector<vector<T>> arr);
    Matrix(const Matrix& a);
    Matrix<T> operator=(const Matrix<T>& m1); //深拷贝
    bool operator==(Matrix& m1);     //矩阵相同时true
    bool operator!=(Matrix& m1);     //矩阵不相同时true
    bool is_size_equal(const Matrix& m1);
    bool is_square();
    bool is_zero();
    Matrix<T> operator+(Matrix& m1);
    Matrix<T> operator-(Matrix<T>& m1);
    Matrix<T> operator*(Matrix<T>& m1);
    Matrix<T> operator*(T a);
    Matrix<T> operator/(double a);

    Matrix<T> operator+=(Matrix& m1);
    Matrix<T> operator-=(Matrix& m1);
    Matrix<T> operator*=(Matrix& m1);
    Matrix<T> operator*=(int a);
    Matrix<T> operator/=(int a);
    Matrix<T> operator^(Matrix& m1); //矩阵按位置相乘
    Matrix<T> conju();           //取共轭矩阵

    Matrix<T> dot(Matrix& m1);
    Matrix<T> cross(Matrix& m1);
    Matrix<T> Transposition();
    Matrix<T> toTransposition();

    T determinant();
    T all_sort(int a[], int now, int length, T& determinant);

    T trace();
    Matrix<T> LU_factor_U();
    Matrix<T> LU_factor_L();
    Matrix<T> LDU_factor_L();
    Matrix<T> LDU_factor_D();
    Matrix<T> LDU_factor_U();

    Matrix<T> Inverse();
    Matrix<T> reshape(int r, int c);
    Matrix<T> slice(int r1, int r2, int c1, int c2);


    T sum();
    T mean();
    T max();
    T min();
    T row_max(int row);
    T row_min(int row);
    T row_sum(int row);
    T row_mean(int row);

    T col_max(int col);
    T col_min(int col);
    T col_sum(int col);
    T col_mean(int col);

    void printMatrix();
    pair<Matrix<T>, Matrix<T>> QR_decomposition();
    T norm();
    Matrix<T> normalized();
    void SetIdentity();

    T* eigenvalues(int max_iter=10e3);
    Matrix<T> eigenvector(T eigenvalue, int max_iter=10e3);
    Matrix<T>* eigenvectors(int max_itr=10e3);
    bool isUpperTri();
    bool isCloseEnough(T a, T b, double threshold = EQ_THRESHOLD);
    pair<T, Matrix<T>> eigenValueAndEigenVector(int max_itr=10e3);
};

template <typename T>
Matrix<T>::Matrix(int row, int col)
{
    if (row <= 0 || col <= 0) {
        cout << "You input negative row/col num" << endl;
        this->m_row = 0;
        this->m_col = 0;
    }
    else {
        this->m_row = row;
        this->m_col = col;
        this->resize(row);
        typename std::vector<std::vector<T>>::iterator iter;
        for (iter = this->begin(); iter < this->end(); iter++)
        {
            iter->resize(col);
        }
    }
}

template <class T>
Matrix<T>::Matrix(vector<vector<T>> arr)
{
    int row = arr.size();
    int col = arr[0].size();
    this->m_row = row;
    this->m_col = col;
    this->resize(row);
    typename std::vector<std::vector<T>>::iterator iter;
    for (iter = this->begin(); iter < this->end(); iter++)
    {
        iter->resize(col);
    }
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            (*this)[i][j] = arr[i][j];
        }
    }
}

template <class T>
Matrix<T>::Matrix(const Matrix& a)
{
    this->m_row = a.m_row;
    this->m_col = a.m_col;
    this->resize(m_row);
    typename std::vector<std::vector<T>>::iterator iter;
    for (iter = this->begin(); iter < this->end(); iter++)
    {
        iter->resize(m_col);
    }
    for (int i = 0; i < a.m_row; i++)
    {
        for (int j = 0; j < a.m_col; j++)
        {
            (*this)[i][j] = a[i][j];
        }
    }
}

template <class T>
void Matrix<T>::printMatrix()
{
    printf("\n---------row:%d,col:%d-----------\n", m_row, m_col);
    for (int i = 0; i < m_row; i++)
    {
        for (int j = 0; j < m_col; j++)
        {
            cout << setprecision(2) << std::setw(7) << (*this)[i][j] << " ";
        }
        printf("\n");
    }
    printf("---------------------------------\n");
}

template <class T>
Matrix<T> Matrix<T>::operator=(const Matrix<T>& m1)
{
    this->clear();
    this->resize(m1.m_row);
    this->m_col = m1.m_col;
    this->m_row = m1.m_row;
    typename std::vector<std::vector<T>>::iterator iter;
    for (iter = this->begin(); iter < this->end(); iter++)
    {
        iter->resize(m1.m_col);
    }

    int i, j;
    for (i = 0; i < m1.m_row; i++)
    {
        for (j = 0; j < m1.m_col; j++)
        {
            (*this)[i][j] = m1[i][j];
        }
    }
    return (*this);
}

template <class T>
bool Matrix<T>::operator==(Matrix& m1)
{
    if (this->m_row != m1.m_row || this->m_col != m1.m_col) return false;
    for (int i = 0; i < m_row; i++) {
        for (int j = 0; j < m_col; j++) {
            if (!isCloseEnough((*this)[i][j], m1[i][j])) return false;
        }
    }
    return true;
}

template <class T>
bool Matrix<T>::operator!=(Matrix& m1)
{
    int i = 0, j = 0;
    bool isSame = false;
    if (this->m_row != m1.m_row || this->m_col != m1.m_col)
    {
        isSame = true;
    }

    while (!isSame && i < this->m_row)
    {
        while (j < this->m_col)
        {
            if ((*this)[i][j] != m1[i][j])
            {
                isSame = true;
                break;
            }
            j++;
        }
        i++;
    }
    return isSame;
}

template <class T>
bool Matrix<T>::is_size_equal(const Matrix& m1) {
    if (this->m_row == m1.m_row && this->m_col == m1.m_col) return true;
    else return false;
}

template<class T>
bool Matrix<T>::is_square()
{
    if (this->m_row == this->m_col) return true;
    else return false;
}

template<class T>
bool Matrix<T>::is_zero()
{
    T tmp; tmp = 0;
    if (this->determinant() = tmp) return true;
    else return false;
}

template <class T>
Matrix<T> Matrix<T>::operator+(Matrix& m1)
{
    assert(is_size_equal(m1) && !this->empty());
    Matrix<T> tmp(this->m_row, this->m_col);
    int i, j;
    for (i = 0; i < m_row; i++)
    {
        for (j = 0; j < m_col; j++)
        {
            tmp[i][j] = (*this)[i][j] + m1[i][j];
        }
    }
    return tmp;
}

template <class T>
Matrix<T> Matrix<T>::operator-(Matrix<T>& m1)
{
    assert(is_size_equal(m1) && !this->empty());
    Matrix<T> tmp(this->m_row, this->m_col);
    int i, j;
    for (i = 0; i < m_row; i++)
    {
        for (j = 0; j < m_col; j++)
        {
            tmp[i][j] = (*this)[i][j] - m1[i][j];
        }
    }
    return tmp;
}

template <class T>
Matrix<T> Matrix<T>::operator*(Matrix<T>& m1)
{
    assert(this->m_col == m1.m_row && !this->empty());
    Matrix<T> tmp(this->m_row, m1.m_col);
    int i, j, k;
    for (i = 0; i < tmp.m_row; i++)
    {
        for (j = 0; j < tmp.m_col; j++)
        {
            for (k = 0; k < this->m_col; k++)
            {
                tmp[i][j] += (*this)[i][k] * m1[k][j];
            }
        }
    }
    return tmp;
}

template <class T>
Matrix<T> Matrix<T>::operator*(T a)
{
    assert(!this->empty());
    Matrix<T> tmp(this->m_row, this->m_col);
    int i, j;
    for (i = 0; i < this->m_row; i++)
    {
        for (j = 0; j < this->m_col; j++)
        {
            tmp[i][j] = (*this)[i][j] * a;
        }
    }
    return tmp;
}

template <class T>
Matrix<T> Matrix<T>::operator/(double a)
{
    assert(!this->empty());
    Matrix<T> tmp(this->m_row, this->m_col);
    int i, j;
    for (i = 0; i < this->m_row; i++)
    {
        for (j = 0; j < this->m_col; j++)
        {
            tmp[i][j] = (*this)[i][j] / a;
        }
    }
    return tmp;
}

template <class T>
Matrix<T> Matrix<T>::operator+=(Matrix& m1)
{
    assert(is_size_equal(m1) && !this->empty());
    for (int i = 0; i < this->m_row; i++)
    {
        for (int j = 0; j < this->m_col; j++)
        {
            (*this)[i][j] += m1[i][j];
        }
    }
    return (*this);
}

template <class T>
Matrix<T> Matrix<T>::operator-=(Matrix& m1)
{
    assert(is_size_equal(m1) && !this->empty());
    for (int i = 0; i < this->m_row; i++)
    {
        for (int j = 0; j < this->m_col; j++)
        {
            (*this)[i][j] -= m1[i][j];
        }
    }
    return (*this);
}

template <class T>
Matrix<T> Matrix<T>::operator*=(Matrix& m1)
{
    assert(this->m_col == m1.m_row && !this->empty());
    Matrix<T> tmp(*this);
    this->clear();
    this->resize(m_row);
    this->m_col = m1.m_col;
    typename std::vector<std::vector<T>>::iterator iter;
    for (iter = this->begin(); iter < this->end(); iter++)
    {
        iter->resize(m1.m_col);
    }
    for (int i = 0; i < this->m_row; i++)
    {
        for (int j = 0; j < this->m_col; j++)
        {
            for (int k = 0; k < m1.m_row; k++)
            {
                (*this)[i][j] += tmp[i][k] * m1[k][j];
            }
        }
    }
    return (*this);
}

template <class T>
Matrix<T> Matrix<T>::operator*=(int a)
{
    assert(!this->empty());
    int i, j;
    for (i = 0; i < this->m_row; i++)
    {
        for (j = 0; j < this->m_col; j++)
        {
            (*this)[i][j] = (*this)[i][j] * a;
        }
    }
    return (*this);
}

template <class T>
Matrix<T> Matrix<T>::operator/=(int a)
{
    int i, j;
    for (i = 0; i < this->m_row; i++)
    {
        for (j = 0; j < this->m_col; j++)
        {
            (*this)[i][j] = (*this)[i][j] / a;
        }
    }
    return (*this);
}

template <class T>
Matrix<T> Matrix<T>::operator^(Matrix<T>& m1)
{
    assert(is_size_equal(m1) && !this->empty());
    int i, j;
    for (i = 0; i < this->m_row; i++)
    {
        for (j = 0; j < this->m_col; j++)
        {
            (*this)[i][j] *= m1[i][j];
        }
    }
    return (*this);
}

template <class T>
Matrix<T> Matrix<T>::conju()
{
    assert(!this->empty());
    Matrix<T> tmp(*this);
    int i, j;
    for (i = 0; i < this->m_row; i++)
        for (j = 0; j < this->m_col; j++) {
            tmp[i][j] = conj(tmp[i][j]);
        }
    return tmp;

}
//template< >
// template<class T >
// Matrix<complex<T>> Matrix<complex<T>>::operator ~() {
//     int i, j;
//     Matrix<Complex<T>> tmp(*this);
//     for (i = 0; i < tmp.m_row; i++) {
//         for (j = 0; j < tmp.m_col; j++) {
//             tmp[i][j] = ~tmp[i][j];
//         }
//     }
//     return tmp;
// }

template <class T>
Matrix<T> Matrix<T>::dot(Matrix<T>& m1)
{
    return (*this) ^ m1;
}

template <class T>
Matrix<T> Matrix<T>::cross(Matrix<T>& m1)
{
    return (*this) * m1;
}

template <class T>
Matrix<T> Matrix<T>::Transposition()
{
    Matrix<T> tmp(this->m_col, this->m_row);
    int i, j;
    for (i = 0; i < tmp.m_row; i++)
    {
        for (j = 0; j < tmp.m_col; j++)
        {
            tmp[i][j] = (*this)[j][i];
        }
    }
    return tmp;
}

template <class T>
Matrix<T> Matrix<T>::toTransposition()
{
    Matrix<T> tmp(*this);
    this->clear();
    this->resize(this->m_col);
    typename std::vector<std::vector<T>>::iterator iter;
    for (iter = this->begin(); iter < this->end(); iter++)
    {
        iter->resize(this->m_row);
    }
    int a = this->m_col;
    this->m_col = this->m_row;
    this->m_row = a;

    int i, j;
    for (i = 0; i < this->m_row; i++)
    {
        for (j = 0; j < this->m_col; j++)
        {
            (*this)[i][j] = tmp[j][i];
        }
    }
    return (*this);
}

template <class T>
T Matrix<T>::determinant()
{
    assert(m_col == m_row);
    int length = m_col, now = 0;
    T d;
    d = 0;
    int* permutation = new int[length];
    //初始化全排列数组
    for (int i = 0; i < length; i++)
    {
        permutation[i] = i;
    }

    d = this->all_sort(permutation, now, length, d);
    delete[] permutation;

    return d;
}

template <class T>
T Matrix<T>::all_sort(int a[], int now, int length, T& determinant)
{
    if (now == length - 1)
    {
        T tmp;
        tmp = 1;
        int pow = 0;
        for (int i = 0; i < length; i++)
        {
            tmp *= (*this)[i][a[i]];
        }

        for (int i = 1; i < length; i++)
            for (int j = 0; j < i; j++)
                if (a[j] > a[i])
                    pow++;

        if (pow % 2 == 0)
            determinant += tmp;
        else
            determinant -= tmp;
        //将-1的幂化为0,tmp初始化
        pow = 0;
        tmp = 1;
    }
    for (int i = now; i < length; i++)
    {
        int tmp = a[now];
        a[now] = a[i];
        a[i] = tmp;
        all_sort(a, now + 1, length, determinant);
        tmp = a[now];
        a[now] = a[i];
        a[i] = tmp;
    }
    return determinant;
}

template <class T>
T Matrix<T>::trace()
{
    if (this->is_square()) {
        T trace = static_cast<T>(0);
        for (int i = 0; i < m_col; i++)
        {
            trace += (*this)[i][i];
        }
        return trace;
    }
    else { return static_cast<T>(0);}
}

template <class T>
Matrix<T> Matrix<T>::LU_factor_U()
{
    assert(m_col == m_row);
    int n = m_col;
    T sum;
    sum = 0;
    Matrix<T> l(n, n);
    Matrix<T> u(n, n);

    for (int i = 0; i < n; i++) //初始化矩阵L和矩阵U
        for (int j = 0; j < n; j++)
        {
            u[i][j] = 0;
            if (i == j)
                l[i][j] = 1;
        }

    for (int i = 0; i < n; i++)
    {
        T sum;
        sum = 0;
        for (int j = i; j < n; j++)
        {
            for (int k = 0; k <= i - 1; k++)
                sum += l[i][k] * u[k][j];
            u[i][j] = (*this)[i][j] - sum; //计算矩阵U
            sum = 0;
        }

        for (int x = i + 1; x < n; x++)
        {
            for (int k = 0; k <= i - 1; k++)
                sum += l[x][k] * u[k][i];
            l[x][i] = ((*this)[x][i] - sum) / u[i][i]; //计算矩阵L
            sum = 0;
        }
    }
    return u;
}

template <class T>
Matrix<T> Matrix<T>::LU_factor_L()
{
    assert(m_col == m_row);
    //需要判断行列式是否为0
    int n = m_col;
    T sum;
    sum = 0;
    Matrix<T> l(n, n);
    Matrix<T> u(n, n);

    for (int i = 0; i < n; i++) //初始化矩阵L和矩阵U
        for (int j = 0; j < n; j++)
        {
            u[i][j] = 0;
            if (i == j)
                l[i][j] = 1;
        }

    for (int i = 0; i < n; i++)
    {
        T sum;
        sum = 0;
        for (int j = i; j < n; j++)
        {
            for (int k = 0; k <= i - 1; k++)
                sum += l[i][k] * u[k][j];
            u[i][j] = (*this)[i][j] - sum; //计算矩阵U
            sum = 0;
        }

        for (int x = i + 1; x < n; x++)
        {
            for (int k = 0; k <= i - 1; k++)
                sum += l[x][k] * u[k][i];
            l[x][i] = ((*this)[x][i] - sum) / u[i][i]; //计算矩阵L
            sum = 0;
        }
    }
    return l;
}

template <class T>
Matrix<T> Matrix<T>::LDU_factor_L()
{
    Matrix<T> l(this->LU_factor_L());
    return l;
}

template <class T>
Matrix<T> Matrix<T>::LDU_factor_D()
{
    assert(this->m_row == this->m_col);
    Matrix<T> tmp(this->LU_factor_U());
    Matrix<T> d(this->m_row, this->m_col);
    for (int i = 0; i < m_row; i++)
    {
        d[i][i] = tmp[i][i];
    }
    return d;
}

template <class T>
Matrix<T> Matrix<T>::LDU_factor_U()
{
    assert(this->m_row == this->m_col);
    Matrix<T> u(this->LU_factor_U());
    for (int i = 0; i < m_row; i++)
    {
        for (int j = i; j < m_col; j++)
        {
            u[i][j] /= this->LU_factor_U()[i][i];
        }
    }
    return u;
}

template<class T>
Matrix<T> Matrix<T>::Inverse() {
    T deter = this->determinant();
    assert(this->is_square());
    T tmp1; tmp1 = 0;
    assert(deter != tmp1);
    if(this->m_row==1)
    {
        vector<vector<T>> v { {static_cast<T>(1) / (*this)[0][0]} };
        Matrix<T> tmp(v);
        return tmp;
    }
    else {
        int i, j, k, m, tt = this->m_row, n = tt - 1;
        Matrix<T> inverse(tt, tt);
        Matrix<T> tmp(n, n);
        for (i = 0; i < tt; i++)
        {

            for (j = 0; j < tt; j++)
            {
                for (k = 0; k < n; k++)
                    for (m = 0; m < n; m++)
                        tmp[k][m] = (*this)[k >= i ? k + 1 : k][m >= j ? m + 1 : m];

                T a = tmp.determinant();
                if ((i + j) % 2 == 1) { a = -a; };
                T b = a / (this->determinant());
                inverse[j][i] = b;
            }
        }
        return inverse;
    }
    
}

template<class T>
Matrix<T> Matrix<T>::reshape(int r, int c)
{
    if (this->m_row * this->m_col != r * c) {
        //cout << "ReshapeError:Not The Same Szie" << __FILE__ << __LINE__ << end;
        return (*this);
    }
    else {
        Matrix<T> ans(r, c);
        int i, j, x = 0, y = 0;
        for (i = 0; i < this->m_row; i++)
        {
            for (j = 0; j < this->m_col; j++)
            {
                ans[x][y] = (*this)[i][j];
                y++;
                if (y == c) {
                    x++;
                    y = 0;
                }
            }
        }
        return ans;
    }
}

template <class T>
Matrix<T> Matrix<T>::slice(int r1, int r2, int c1, int c2)
{
    // assert(r1)
    if (r1 > r2) { int tmp = r1; r1 = r2; r2 = tmp; }
    if (c1 > c2) { int tmp = c1; c1 = c2; c2 = tmp; }
    if (r1 < 0)
    {
        if (r2 < 0) { r2 = 0; }
        r1 = 0;
    }
    if (r2 >= this->m_row) {
        if (r1 >= this->m_row) { r1 = this->m_row - 1; }
        r2 = this->m_row - 1;
    }
    if (c1 < 0)
    {
        if (c2 < 0) { c2 = 0; }
        c1 = 0;
    }
    if (c2 >= this->m_col)
    {
        if (c1 >= this->m_col) { c1 = this->m_col - 1; }
        c2 = this->m_col - 1;
    }

    Matrix<T> tmp(r2 - r1 + 1, c2 - c1 + 1);
    for (int i = r1; i <= r2; i++) {
        for (int j = c1; j <= c2; j++) {
            tmp[i - r1][j - c1] = (*this)[i][j];
        }
    }
    return tmp;

}

template <class T>
T Matrix<T>::sum()
{
    T sum; sum = 0;
    for (int i = 0; i < m_row; i++)
    {
        for (int j = 0; j < m_col; j++)
        {
            sum += (*this)[i][j];
        }
    }
    return (sum);
}

template <class T>
T Matrix<T>::mean()
{
    T total; total = this->m_row * this->m_col;
    return (this->sum() / total);
}

template <class T>
T Matrix<T>::max()
{
    int k = 0, m = 0, i, j;
    for (i = 0; i < this->m_row; i++)
        for (j = 0; j < this->m_col; j++)
            if ((*this)[i][j] > (*this)[k][m]) {
                k = i; m = j;
            }

    return (*this)[k][m];
}

template <class T>
T Matrix<T>::min()
{
    int k = 0, m = 0, i = 0, j = 0;
    for (i = 0; i < this->m_row; i++) {
        for (j = 0; j < this->m_col; j++) {
            if ((*this)[k][m] > (*this)[i][j])
            {
                k = i;
                m = j;
            }
        }
    }
    return (*this)[k][m];
}
template <typename T>
T Matrix<T>::row_max(int row)
{
    assert(row >= 0 && row < this->m_row);
    int k = 0;
    for (int i = 0; i < this->m_col; i++) {
        if ((*this)[row][k] < (*this)[row][i])
            k = i;
    }
    return (*this)[row][k];
}
template <typename T>
T Matrix<T>::row_min(int row)
{
    assert(row >= 0 && row < this->m_row);
    int k = 0;
    for (int i = 0; i < this->m_col; i++)
        if ((*this)[row][k] > (*this)[row][i])
            k = i;

    return (*this)[row][k];
}
template <typename T>
T Matrix<T>::row_sum(int row)
{
    assert(row >= 0 && row < this->m_row);
    T row_sum; row_sum = 0;
    for (int i = 0; i < this->m_col; i++) {
        row_sum += (*this)[row][i];
    }
    return row_sum;
}
template <typename T>
T Matrix<T>::row_mean(int row)
{
    assert(row >= 0 && row < this->m_row);
    T total; total = (this->m_col);
    return this->row_sum(row) / total;
}
template <typename T>
T Matrix<T>::col_max(int col) {
    assert(col >= 0 && col < this->m_col);
    int k = 0;
    for (int i = 0; i < this->m_row; i++)
        if ((*this)[k][col] < (*this)[i][col])
            k = i;

    return (*this)[k][col];
}
template <typename T>
T Matrix<T>::col_min(int col) {
    assert(col >= 0 && col < this->m_col);
    int k = 0;
    for (int i = 0; i < this->m_row; i++)
        if ((*this)[k][col] > (*this)[i][col])
            k = i;

    return (*this)[k][col];
}
template <typename T>
T Matrix<T>::col_sum(int col)
{
    assert(col >= 0 && col < this->m_col);
    T col_sum; col_sum = 0;
    for (int i = 0; i < this->m_row; i++) {
        col_sum += (*this)[i][col];
    }
    return col_sum;
}
template <typename T>
T Matrix<T>::col_mean(int col) {
    assert(col >= 0 && col < this->m_col);
    T total; total = this->m_row;
    return this->col_sum(col) / total;
}

// from: https://github.com/QuantitativeBytes/qbLinAlg/blob/main/qbQR.h
template <typename T>
pair<Matrix<T>, Matrix<T>> Matrix<T>::QR_decomposition()
{
    Matrix<T> input = *this;
    vector<Matrix<T>> plist;
    for (int j = 0; j < m_row - 1; j++) {
        Matrix<T> a1(1, m_row - j);
        Matrix<T> b1(1, m_row - j);

        for (int i = j; i < m_row; i++) {
            a1[0][i - j] = input[i][j];
            b1[0][i - j] = static_cast<T>(0.0);
        }
        b1[0][0] = static_cast<T>(1.0);

        T a1norm = a1.norm();
        
        int sgn = -1;
        if (a1[0][0] < static_cast<T>(0.0)) {
            sgn = 1;
        }

        Matrix<T> temp = b1 * sgn * a1norm;
        Matrix<T> u = a1 - temp;
        Matrix<T> n = u.normalized();
        Matrix<T> nTrans = n.Transposition();
        Matrix<T> I (m_row - j, m_row - j);
        I.SetIdentity();

        Matrix<T> temp1 = n * static_cast<T>(2.0);
        Matrix<T> temp2 = nTrans * temp1;
        Matrix<T> Ptemp = I - temp2;

        Matrix<T> P (m_row, m_col);
        P.SetIdentity();

        for (int x = j; x < m_row; x++) {
            for (int y = j; y < m_col; y++) {
                P[x][y] = Ptemp[x - j][y - j];
            }
        }

        plist.push_back(P);
        input = P * input;
    }

    Matrix<T> qMat = plist[0];
    for (int i = 1; i < m_row - 1; i++) {
        Matrix<T> temp3 = plist[i].Transposition();
        qMat = qMat * temp3;
    }

    int numElements = plist.size();
    Matrix<T> rMat = plist[numElements - 1];
    for (int i = (numElements - 2); i >= 0; i--) {
        rMat = rMat * plist[i];
    }
    rMat = rMat * (*this);

    return pair<Matrix<T>, Matrix<T>>(qMat, rMat);
}

template <typename T>
T Matrix<T>::norm()
{
    T cumulativeSum = static_cast<T>(0.0);
    for (int i = 0; i < m_row; i++) {
        for (int j = 0; j < m_col; j++) {
            cumulativeSum += (*this)[i][j] * (*this)[i][j];
        }
    }
    return sqrt(cumulativeSum);
}

template <typename T>
Matrix<T> Matrix<T>::normalized()
{
    T norm = this->norm();
    Matrix<T> copy(*this);
    return copy * (static_cast<T>(1.0) / norm);
}

template <typename T>
void Matrix<T>::SetIdentity() {
    for (int i = 0 ; i < m_row; i++) {
        for (int j = 0; j < m_col; j++) {
            if (i == j) {
                (*this)[i][j] = static_cast<T>(1.0);
            } else {
                (*this)[i][j] = static_cast<T>(0.0);
            }
        }
    }
}

// from: https://github.com/QuantitativeBytes/qbLinAlg/blob/main/qbEIG.h
// only work for symmetric metrices
template <typename T>
T* Matrix<T>::eigenvalues(int max_iter) {
    Matrix<T> A = (*this);
    Matrix<T> identityMatrix (m_row, m_col);
    identityMatrix.SetIdentity();

    for (int i = 0; i < max_iter; i++) {
        auto qrResult = A.QR_decomposition();
        A = qrResult.second * qrResult.first;
        if (A.isUpperTri()) break;
    }
    
    T *eigenvalues = new T[m_row];
    for (int i = 0; i < m_row; i++) {
        eigenvalues[i] = A[i][i];
    }
    return eigenvalues;
}

template <typename T>
bool Matrix<T>::isCloseEnough(T a, T b, double threshold) {
    return abs(a - b) < static_cast<T>(threshold);
}

template <typename T>
bool Matrix<T>::isUpperTri() {
    T cumulativeSum = static_cast<T>(0);
    for (int i = 1; i < m_row; i++) {
        for (int j = 0; j < i; j++) {
            cumulativeSum += (*this)[i][j];
        }
    }
    return isCloseEnough(cumulativeSum, static_cast<T>(0));
}

// from: https://github.com/QuantitativeBytes/qbLinAlg/blob/main/qbEIG.h
template <typename T>
Matrix<T> Matrix<T>::eigenvector(T eigenvalue, int max_iter) {
    Matrix<T> A = (*this);
    random_device myRandomDevice;
    mt19937 myRandomGenerator(myRandomDevice());
    uniform_int_distribution<int> myDistribution(1.0, 10.0);
    
    Matrix<T> identityMatrix(m_row, m_col);
    identityMatrix.SetIdentity();

    Matrix<T> v(m_row, 1);
    for (int i = 0; i < m_row; i++) {
        v[i][0] = static_cast<T>(myDistribution(myRandomGenerator));
    }

    T deltaThreshold = static_cast<T>(EQ_THRESHOLD);
    T delta = static_cast<T>(1e-6);
    Matrix<T> preVector(m_row, 1);
    Matrix<T> tempMatrix(m_row, m_row);
    
    for (int i = 0; i < max_iter; i++) {
        preVector = v;
        Matrix<T> temp = identityMatrix * eigenvalue;
        tempMatrix = A - temp;
        tempMatrix = tempMatrix.Inverse();
        v = tempMatrix * v;
        v = v.normalized();

        delta = (v - preVector).norm();
        if (delta > deltaThreshold) break;
    }
    return v;
}

template <typename T>
Matrix<T>* Matrix<T>::eigenvectors(int max_itr) {
    Matrix<T> * eigenvectors = new Matrix<T>[m_row];
    T * eigenvalues = this->eigenvalues();
    for (int i = 0; i < m_row; i++) {
        eigenvectors[i] = this->eigenvector(*(eigenvalues + i));
    }
    return eigenvectors;
}

// from: https://github.com/QuantitativeBytes/qbLinAlg/blob/main/qbEIG.h
template <typename T>
pair<T, Matrix<T>> Matrix<T>::eigenValueAndEigenVector(int max_itr) {
    T eigenvalue;
    Matrix<T> inputMatrix = (*this);
    random_device myRandomDevice;
    mt19937 myRandomGenerator(myRandomDevice());
    uniform_int_distribution<int> myDistribution(1.0, 10.0);
    Matrix<T> identityMatrix(m_row, m_col);
    identityMatrix.SetIdentity();
    
    Matrix<T> v(m_row, 1);
    for (int i = 0; i < m_row; i++) {
        v[i][0] = static_cast<T>(static_cast<T>(myDistribution(myRandomGenerator)));
    }
    Matrix<T> v1(m_row, 1);
    for (int i = 0; i < max_itr; i++) {
        v1 = inputMatrix * v;
        v1 = v1.normalized();
        v = v1;
    }
    T cumulativeSum = static_cast<T>(0.0);
    for (int i = 1; i < m_row; i++) {
        cumulativeSum += inputMatrix[0][i] * v1[i][0];
    }
    eigenvalue = (cumulativeSum / v1[0][0]) + inputMatrix[0][0];
    return pair<T, Matrix<T>>(eigenvalue, v1);
}