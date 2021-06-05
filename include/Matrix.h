#pragma once
#include <vector>
#include <iostream>
#include <cassert>
#include <iomanip>
#include <math.h>
#include <complex>

using namespace std;
using std::setw;

template <class T>
class Matrix : public std::vector<std::vector<T>>
{
public:
    int m_row;
    int m_col;
    Matrix(int row, int col);
    Matrix(vector<vector<T>> arr);
    Matrix(const Matrix &a);
    Matrix<T> operator=(Matrix &m1); //深拷贝
    bool operator==(Matrix &m1);     //矩阵相同时true
    bool operator!=(Matrix &m1);     //矩阵不相同时true
    Matrix<T> operator+(Matrix &m1);
    Matrix<T> operator-(Matrix &m1);
    Matrix<T> operator*(Matrix &m1);
    Matrix<T> operator*(int a);
    Matrix<T> operator/(double a);

    Matrix<T> operator+=(Matrix &m1);
    Matrix<T> operator-=(Matrix &m1);
    Matrix<T> operator*=(Matrix &m1);
    Matrix<T> operator*=(int a);
    Matrix<T> operator/=(int a);
    Matrix<T> operator^(Matrix &m1); //矩阵按位置相乘
    Matrix<T> operator~();           //取共轭矩阵

    Matrix<T> dot(Matrix &m1);   //未实现
    Matrix<T> cross(Matrix &m1); //未实现
    Matrix<T> Transposition();
    Matrix<T> toTransposition();

    T determinant();
    T all_sort(int a[], int now, int length, T &determinant);

    T trace();
    Matrix<T> LU_factor_U();
    Matrix<T> LU_factor_L();
    Matrix<T> LDU_factor_L();
    Matrix<T> LDU_factor_D();
    Matrix<T> LDU_factor_U();

    T sum(int row = 0, int col = 0);
    T mean(int row = 0, int col = 0);
    T max(int row = 0, int col = 0);
    T min(int row = 0, int col = 0);

    void printMatrix();
};

template <typename T>
Matrix<T>::Matrix(int row, int col)
{
    this->m_row = row;
    this->m_col = col;
    this->resize(row);
    typename std::vector<std::vector<T>>::iterator iter;
    for (iter = this->begin(); iter < this->end(); iter++)
    {
        iter->resize(col);
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
Matrix<T>::Matrix(const Matrix &a)
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
            cout << std::setw(7) << (*this)[i][j] << " ";
        }
        printf("\n");
    }
    printf("------------------------\n");
}

template <class T>
Matrix<T> Matrix<T>::operator=(Matrix &m1)
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
bool Matrix<T>::operator==(Matrix &m1)
{
    int i = 0, j = 0;
    bool isSame = true;
    if (this->m_row != m1.m_row || this->m_col != m1.m_col)
    {
        isSame = false;
    }
    else
    {
        isSame = true;
    }
    while (isSame && i < this->m_row)
    {
        while (j < this->m_col)
        {
            if ((*this)[i][j] != m1[i][j])
            {
                isSame = false;
                break;
            }
            j++;
        }
        i++;
    }
    return isSame;
}

template <class T>
bool Matrix<T>::operator!=(Matrix &m1)
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
Matrix<T> Matrix<T>::operator+(Matrix &m1)
{
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
Matrix<T> Matrix<T>::operator-(Matrix &m1)
{
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
Matrix<T> Matrix<T>::operator*(Matrix &m1)
{
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
Matrix<T> Matrix<T>::operator*(int a)
{
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
Matrix<T> Matrix<T>::operator+=(Matrix &m1)
{
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
Matrix<T> Matrix<T>::operator-=(Matrix &m1)
{
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
Matrix<T> Matrix<T>::operator*=(Matrix &m1)
{
    //this->clear();
    //this->resize(m_row);
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
Matrix<T> Matrix<T>::operator^(Matrix<T> &m1)
{
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
Matrix<T> Matrix<T>::operator~()
{
    return (*this);
}

// template< >
// Matrix<std::complex> Matrix<st>::operator ~() {
//     int i, j;
//     Matrix<Complex> tmp(*this);
//     for (i = 0; i < tmp.m_row; i++) {
//         for (j = 0; j < tmp.m_col; j++) {
//             tmp[i][j] = ~tmp[i][j];
//         }
//     }
//     return tmp;
// }

template <class T>
Matrix<T> Matrix<T>::dot(Matrix<T> &m1)
{
    int i, j;
    Matrix<T> tmp(*this);
    for (i = 0; i < tmp.m_row; i++)
    {
        for (j = 0; j < tmp.m_col; j++)
        {
            tmp[i][j] *= m1[i][j];
        }
    }
    return tmp;
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
    int *permutation = new int[length];
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
T Matrix<T>::all_sort(int a[], int now, int length, T &determinant)
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
        swap(a[now], a[i]);
        all_sort(a, now + 1, length, determinant);
        swap(a[now], a[i]);
    }
    return determinant;
}

template <class T>
T Matrix<T>::trace()
{
    assert(m_col == m_row);
    T trace;
    trace = 0;
    for (int i = 0; i < m_col; i++)
    {
        trace += (*this)[i][i];
    }
    return trace;
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

template <class T>
T Matrix<T>::sum(int row, int col)
{
    assert(0 <= row && row <= m_row);
    assert(0 <= col && col <= m_col);

    T sum;
    sum = 0;

    if (row == 0 && col == 0)
    {
        for (int i = 0; i < m_row; i++)
        {
            for (int j = 0; j < m_col; j++)
            {
                sum += (*this)[i][j];
            }
        }
    }
    else if (row == 0 && col != 0)
    {
        for (int i = 0; i < m_row; i++)
        {
            sum += (*this)[i][col - 1];
        }
    }

    else if (row != 0 && col == 0)
    {
        for (int i = 0; i < m_col; i++)
        {
            sum += (*this)[row - 1][i];
        }
    }
    else
    {
        sum = (*this)[row - 1][col - 1];
    }
    return (sum);
}

template <class T>
T Matrix<T>::mean(int row, int col)
{
    assert(0 <= row && row <= m_row);
    assert(0 <= col && col <= m_col);
    if (row == 0 && col == 0)
    {
        return (this->sum() / m_col * m_row);
    }
    else if (row != 0 && col == 0)
    {
        return (this->sum(row, 0) / m_col);
    }
    else if (row == 0 && col != 0)
    {
        return (this->sum(0, col) / m_row);
    }
    else
    {
        return (*this)[row - 1][col - 1];
    }
}

template <class T>
T Matrix<T>::max(int row, int col)
{
    assert(0 <= row && row <= m_row);
    assert(0 <= col && col <= m_col);
    int k = 0, m = 0, i, j;
    if (row == 0 && col == 0)
    {
        for (i = 0; i < m_row; i++)
        {
            for (j = 0; j < m_col; j++)
            {
                if ((*this)[i][j] > (*this)[k][m])
                {
                    k = i;
                    m = j;
                }
            }
        }
        return (*this)[k][m];
    }
    else if (row != 0 && col == 0)
    {
        k = row - 1;
        for (i = 0; i < m_col; i++)
        {
            if ((*this)[k][i] > (*this)[k][m])
            {
                m = i;
            }
        }
        return (*this)[k][m];
    }
    else if (row == 0 && col != 0)
    {
        m = col - 1;
        for (i = 0; i < m_row; i++)
        {
            if ((*this)[i][m] > (*this)[k][m])
            {
                k = i;
            }
        }
        return (*this)[k][m];
    }
    else
    {
        return (*this)[row - 1][col - 1];
    }
}

template <class T>
T Matrix<T>::min(int row, int col)
{
    assert(0 <= row && row <= m_row);
    assert(0 <= col && col <= m_col);
    int k = 0, m = 0, i, j;
    if (row == 0 && col == 0)
    {
        for (i = 0; i < m_row; i++)
        {
            for (j = 0; j < m_col; j++)
            {
                if ((*this)[i][j] < (*this)[k][m])
                {
                    k = i;
                    m = j;
                }
            }
        }
        return (*this)[k][m];
    }
    else if (row != 0 && col == 0)
    {
        k = row - 1;
        for (i = 0; i < m_col; i++)
        {
            if ((*this)[k][i] < (*this)[k][m])
            {
                m = i;
            }
        }
        return (*this)[k][m];
    }
    else if (row == 0 && col != 0)
    {
        m = col - 1;
        for (i = 0; i < m_row; i++)
        {
            if ((*this)[i][m] < (*this)[k][m])
            {
                k = i;
            }
        }
        return (*this)[k][m];
    }
    else
    {
        return (*this)[row - 1][col - 1];
    }
}