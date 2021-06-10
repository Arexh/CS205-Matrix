#pragma once
#include <opencv2/opencv.hpp>
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

#define EQ_THRESHOLD 1e-10

template <typename T>
class Matrix
{
private:
    vector<vector<T>>* data;
public:
    int m_row;
    int m_col;

    Matrix() : Matrix(0, 0){};
    Matrix(int row, int col);
    Matrix(int row, int col, const T* data);
    Matrix(const vector<vector<T>>& arr);
    Matrix(const Matrix<T> &a);

    ~Matrix();

    bool operator==(const Matrix<T> &m1) const;
    bool operator!=(const Matrix<T> &m1) const;

    bool is_size_equal(const Matrix<T> &m1) const;
    bool is_square() const;
    bool is_zero() const;

    Matrix<T> operator=(const Matrix<T> &m1);

    vector<T>& operator[](int idx) const;

    template <typename U>
    friend Matrix<U> operator+(const Matrix<U>& l, const Matrix<U>& r);
    template <typename U>
    friend Matrix<U> operator+(const U& l, const Matrix<U>& r);
    template <typename U>
    friend Matrix<U> operator+(const Matrix<U>& l, const U& r);

    template <typename U>
    friend Matrix<U> operator-(const Matrix<U>& l, const Matrix<U>& r);
    template <typename U>
    friend Matrix<U> operator-(const U& l, const Matrix<U>& r);
    template <typename U>
    friend Matrix<U> operator-(const Matrix<U>& l, const U& r);

    template <typename U>
    friend Matrix<U> operator*(const Matrix<U>& l, const Matrix<U>& r);
    template <typename U>
    friend Matrix<U> operator*(const U& l, const Matrix<U>& r);
    template <typename U>
    friend Matrix<U> operator*(const Matrix<U>& l, const U& r);
    
    template <typename U>
    friend Matrix<U> operator/(const U& l, const Matrix<U>& r);
    template <typename U>
    friend Matrix<U> operator/(const Matrix<U>& l, const U& r);
    template <typename U>
    friend Matrix<U> operator^(const Matrix<U>& l, const Matrix<U>& r);

    Matrix<T> operator-() const;

    Matrix<T> operator+=(const Matrix<T> &m1);
    Matrix<T> operator-=(const Matrix<T> &m1);
    Matrix<T> operator*=(const Matrix<T> &m1);
    Matrix<T> operator*=(const T& a);
    Matrix<T> operator/=(const T& a);

    Matrix<T> conju();

    Matrix<T> & dot(const Matrix<T> &m1) const;
    Matrix<T> cross(const Matrix<T> &m1) const;
    Matrix<T> Transposition() const;
    Matrix<T> toTransposition();

    T determinant() const;
    T trace() const;

    Matrix<T> LU_factor_U() const;
    Matrix<T> LU_factor_L() const;
    Matrix<T> LDU_factor_L() const;
    Matrix<T> LDU_factor_D() const;
    Matrix<T> LDU_factor_U() const;

    Matrix<T> Inverse() const;
    Matrix<T> reshape(int r, int c) const;
    Matrix<T> slice(int r1, int r2, int c1, int c2) const;

    T sum() const;
    T mean() const;
    T max() const;
    T min() const;
    T row_max(int row) const;
    T row_min(int row) const;
    T row_sum(int row) const;
    T row_mean(int row) const;

    T col_max(int col) const;
    T col_min(int col) const;
    T col_sum(int col) const;
    T col_mean(int col) const;

    void printMatrix() const;

    pair<Matrix<T>, Matrix<T>> QR_decomposition() const;
    T norm() const;
    Matrix<T> normalized() const;
    void SetIdentity();

    T *eigenvalues(int max_iter = 10e3);
    Matrix<T> eigenvector(T eigenvalue, int max_iter = 10e3);
    Matrix<T> *eigenvectors(int max_itr = 10e3);
    bool isUpperTri();
    static bool isCloseEnough(T a, T b, double threshold = EQ_THRESHOLD);
    pair<T, Matrix<T>> eigenValueAndEigenVector(int max_itr = 10e3);
    T** toArray();
    cv::Mat* toOpenCVMat(int type);

    static Matrix<T> fromOpenCV(const cv::Mat &cvMat);
    static Matrix<T> conv2D(const Matrix<T> &input, const Matrix<T> &kernel, int stride=1, bool same_padding=true);

private:
    T all_sort(int a[], int now, int length, T &determinant) const;
    void printMatrixInt() const;
};

template <typename T>
Matrix<T>::~Matrix()
{
    if (data)
        delete data;
    data = nullptr;
}

template <typename T>
Matrix<T>::Matrix(int row, int col)
{
    if (row <= 0 || col <= 0)
    {
        cout << "You input negative row/col num" << endl;
        m_row = 0;
        m_col = 0;
        data = nullptr;
    }
    else
    {
        m_row = row;
        m_col = col;
        data = new vector<vector<T>>();
        data->resize(row);
        typename vector<vector<T>>::iterator iter;
        for (iter = data->begin(); iter < data->end(); iter++)
        {
            iter->resize(col);
        }
    }
}

template <typename T>
Matrix<T>::Matrix(const vector<vector<T>>& arr)
{
    int row = arr.size();
    int col = arr[0].size();
    m_row = row;
    m_col = col;
    data = new vector<vector<T>>();
    data->resize(row);
    typename vector<vector<T>>::iterator iter;
    for (iter = data->begin(); iter < data->end(); iter++)
    {
        iter->resize(col);
    }
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            (*data)[i][j] = arr[i][j];
        }
    }
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T> &a)
{
    m_row = a.m_row;
    m_col = a.m_col;
    data = new vector<vector<T>>();
    data->resize(m_row);
    typename vector<vector<T>>::iterator iter;
    for (iter = data->begin(); iter < data->end(); iter++)
    {
        iter->resize(m_col);
    }
    for (int i = 0; i < a.m_row; i++)
    {
        for (int j = 0; j < a.m_col; j++)
        {
            (*data)[i][j] = (*a.data)[i][j];
        }
    }
}

template <typename T>
void Matrix<T>::printMatrix() const
{
    printf("\n---------row:%d,col:%d-----------\n", m_row, m_col);
    for (int i = 0; i < m_row; i++)
    {
        for (int j = 0; j < m_col; j++)
        {
            cout << std::setw(7) << (*data)[i][j] << " ";
        }
        printf("\n");
    }
    printf("---------------------------------\n");
}

template <typename T>
void Matrix<T>::printMatrixInt() const
{
    printf("\n---------row:%d,col:%d-----------\n", m_row, m_col);
    for (int i = 0; i < m_row; i++)
    {
        for (int j = 0; j < m_col; j++)
        {
            cout << setprecision(2) << std::setw(7) << ((int) (*data)[i][j]) << " ";
        }
        printf("\n");
    }
    printf("---------------------------------\n");
}

template <>
void Matrix<char>::printMatrix() const
{
    printMatrixInt();
}

template <>
void Matrix<uchar>::printMatrix() const
{
    printMatrixInt();
}

template <typename T>
vector<T>& Matrix<T>::operator[](int idx) const
{
    return (*data)[idx];
}

template <typename T>
Matrix<T> Matrix<T>::operator=(const Matrix<T> &m1)
{
    if (this != &m1) {
        m_row = m1.m_row;
        m_col = m1.m_col;

        if (data)
            delete data;
        
        data = new vector<vector<T>>(m1.m_row);

        typename vector<vector<T>>::iterator iter;
        for (iter = data->begin(); iter < data->end(); iter++)
            iter->resize(m1.m_col);

        for (int i = 0; i < m1.m_row; i++)
            for (int j = 0; j < m1.m_col; j++)
                (*data)[i][j] = m1[i][j];
    }
    return (*this);
}

template <typename T>
bool Matrix<T>::operator==(const Matrix<T> &m1) const
{
    if (m_row != m1.m_row || m_col != m1.m_col)
        return false;

    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            if (!isCloseEnough((*data)[i][j], m1[i][j]))
                return false;

    return true;
}

template <typename T>
bool Matrix<T>::operator!=(const Matrix<T> &m1) const
{
    return !(*this == m1);
}

template <typename T>
bool Matrix<T>::is_size_equal(const Matrix<T> &m1) const
{
    return m_row == m1.m_row && m_col == m1.m_col;
}

template <typename T>
bool Matrix<T>::is_square() const
{
    return m_row == m_col;
}

template <typename T>
bool Matrix<T>::is_zero() const
{
    return determinant() ==  static_cast<T>(0);
}

template <typename T>
Matrix<T> Matrix<T>::operator-() const
{
    Matrix<T> result(*this);
    
    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            result[i][j] = -result[i][j];
    
    return result;
}

template <typename T>
Matrix<T> operator+(const Matrix<T>& l, const Matrix<T>& r)
{
    Matrix<T> result(l);

    for (int i = 0; i < l.m_row; i++)
        for (int j = 0; j < l.m_col; j++)
            result[i][j] += r[i][j];

    return result;
}

template <typename T>
Matrix<T> operator+(const Matrix<T>& l, const T& r)
{
    Matrix<T> result(l);

    for (int i = 0; i < result.m_row; i++)
        for (int j = 0; j < result.m_col; j++)
            result[i][j] += r;

    return result;
}

template <typename T>
Matrix<T> operator+(const T& l, const Matrix<T>& r)
{
    return r + l;
}

template <typename T>
Matrix<T> operator-(const Matrix<T>& l, const Matrix<T>& r)
{
    Matrix<T> result(l);

    for (int i = 0; i < result.m_row; i++)
        for (int j = 0; j < result.m_col; j++)
            result[i][j] -= r[i][j];

    return result;
}

template <typename T>
Matrix<T> operator-(const Matrix<T>& l, const T& r)
{
    Matrix<T> result(l);

    for (int i = 0; i < result.m_row; i++)
        for (int j = 0; j < result.m_col; j++)
            result[i][j] -= r;

    return result;
}

template <typename T>
Matrix<T> operator-(const T& l, const Matrix<T>& r)
{
    return r.Matrix<T>::operator-() + l;
}

template <typename T>
Matrix<T> operator*(const Matrix<T>& l, const Matrix<T>& r)
{
    Matrix<T> result(l.m_row, r.m_col);

    for (int i = 0; i < result.m_row; i++)
        for (int j = 0; j < result.m_col; j++)
            for (int k = 0; k < l.m_col; k++)
                result[i][j] += l[i][k] * r[k][j];

    return result;
}

template <typename T>
Matrix<T> operator*(const Matrix<T>& l, const T& r)
{
    Matrix<T> result(l);

    for (int i = 0; i < result.m_row; i++)
        for (int j = 0; j < result.m_col; j++)
            result[i][j] *= r;

    return result;
}

template <typename T>
Matrix<T> operator*(const T& l, const Matrix<T>& r)
{
    return r * l;
}

template <typename T>
Matrix<T> operator/(const Matrix<T>& l, const T& r)
{
    Matrix<T> result(l);

    for (int i = 0; i < result.m_row; i++)
        for (int j = 0; j < result.m_col; j++)
            result[i][j] /= r;

    return result;
}

template <typename T>
Matrix<T> operator/(const T& l, const Matrix<T>& r)
{
    Matrix<T> result(r);

    for (int i = 0; i < result.m_row; i++)
        for (int j = 0; j < result.m_col; j++)
            result[i][j] = l / result[i][j];

    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator+=(const Matrix<T>& m1)
{
    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            (*data)[i][j] += m1[i][j];

    return (*this);
}

template <typename T>
Matrix<T> Matrix<T>::operator-=(const Matrix<T> &m1)
{
    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            (*data)[i][j] -= m1[i][j];

    return (*this);
}

template <typename T>
Matrix<T> Matrix<T>::operator*=(const Matrix<T> &m1)
{
    Matrix<T> result = (*this) * m1;
    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            (*data)[i][j] = result[i][j];

    return (*this);
}

template <typename T>
Matrix<T> Matrix<T>::operator*=(const T& m1)
{
    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            (*data)[i][j] *= m1;

    return (*this);
}

template <typename T>
Matrix<T> Matrix<T>::operator/=(const T& a)
{
    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            (*data)[i][j] = (*data)[i][j] / a;

    return (*this);
}

template <typename T>
Matrix<T> operator^(const Matrix<T> &l, const Matrix<T> &r)
{
    Matrix<T> result(l);

    for (int i = 0; i < result.m_row; i++)
        for (int j = 0; j < result.m_col; j++)
            result[i][j] *= r[i][j];

    return result;
}

template <typename T>
Matrix<T> Matrix<T>::conju()
{
    Matrix<T> result(*this);

    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            result[i][j] = conj(result[i][j]);

    return result;
}

template <typename T>
Matrix<T> & Matrix<T>::dot(const Matrix<T> &m1) const
{
    return (*this) ^ m1;
}

template <typename T>
Matrix<T> Matrix<T>::cross(const Matrix<T> &m1) const
{
    return (*this) * m1;
}

template <typename T>
Matrix<T> Matrix<T>::Transposition() const
{
    Matrix<T> result(m_col, m_row);

    for (int i = 0; i < result.m_row; i++)
        for (int j = 0; j < result.m_col; j++)
            result[i][j] = (*data)[j][i];

    return result;
}

template <typename T>
Matrix<T> Matrix<T>::toTransposition()
{
    // Matrix<T> result = Transposition();
    
    // for (int i = 0; i < m_row; i++)
    //     for (int j = 0; j < m_col; j++)
    //         (*data)[i][j] = result[i][j];

    return (*this);
}

template <typename T>
T Matrix<T>::determinant() const
{
    assert(m_col == m_row);
    int length = m_col, now = 0;
    T d;
    d = 0;
    int *permutation = new int[length];
    for (int i = 0; i < length; i++)
    {
        permutation[i] = i;
    }

    d = all_sort(permutation, now, length, d);
    delete[] permutation;

    return d;
}

template <typename T>
T Matrix<T>::all_sort(int a[], int now, int length, T &determinant) const
{
    if (now == length - 1)
    {
        T tmp;
        tmp = 1;
        int pow = 0;
        for (int i = 0; i < length; i++)
        {
            tmp *= (*data)[i][a[i]];
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

template <typename T>
T Matrix<T>::trace() const
{
    if (is_square())
    {
        T trace = static_cast<T>(0);

        for (int i = 0; i < m_col; i++)
            trace += (*data)[i][i];

        return trace;
    }
    else
    {
        return static_cast<T>(0);
    }
}

template <typename T>
Matrix<T> Matrix<T>::LU_factor_U() const
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
            u[i][j] = (*data)[i][j] - sum; //计算矩阵U
            sum = 0;
        }

        for (int x = i + 1; x < n; x++)
        {
            for (int k = 0; k <= i - 1; k++)
                sum += l[x][k] * u[k][i];
            l[x][i] = ((*data)[x][i] - sum) / u[i][i]; //计算矩阵L
            sum = 0;
        }
    }
    return u;
}

template <typename T>
Matrix<T> Matrix<T>::LU_factor_L() const
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
            u[i][j] = (*data)[i][j] - sum; //计算矩阵U
            sum = 0;
        }

        for (int x = i + 1; x < n; x++)
        {
            for (int k = 0; k <= i - 1; k++)
                sum += l[x][k] * u[k][i];
            l[x][i] = ((*data)[x][i] - sum) / u[i][i]; //计算矩阵L
            sum = 0;
        }
    }
    return l;
}

template <typename T>
Matrix<T> Matrix<T>::LDU_factor_L() const
{
    Matrix<T> l(this->LU_factor_L());
    return l;
}

template <typename T>
Matrix<T> Matrix<T>::LDU_factor_D() const
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

template <typename T>
Matrix<T> Matrix<T>::LDU_factor_U() const
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

template <typename T>
Matrix<T> Matrix<T>::Inverse() const
{
    T deter = this->determinant();
    assert(this->is_square());
    T tmp1;
    tmp1 = 0;
    assert(deter != tmp1);
    if (this->m_row == 1)
    {
        vector<vector<T>> v{{static_cast<T>(1) / (*data)[0][0]}};
        Matrix<T> tmp(v);
        return tmp;
    }
    else
    {
        int i, j, k, m, tt = this->m_row, n = tt - 1;
        Matrix<T> inverse(tt, tt);
        Matrix<T> tmp(n, n);
        for (i = 0; i < tt; i++)
        {

            for (j = 0; j < tt; j++)
            {
                for (k = 0; k < n; k++)
                    for (m = 0; m < n; m++)
                        tmp[k][m] = (*data)[k >= i ? k + 1 : k][m >= j ? m + 1 : m];

                T a = tmp.determinant();
                if ((i + j) % 2 == 1)
                {
                    a = -a;
                };
                T b = a / (this->determinant());
                inverse[j][i] = b;
            }
        }
        return inverse;
    }
}

template <typename T>
Matrix<T> Matrix<T>::reshape(int r, int c) const
{
    if (this->m_row * this->m_col != r * c)
    {
        //cout << "ReshapeError:Not The Same Szie" << __FILE__ << __LINE__ << end;
        return (*this);
    }
    else
    {
        Matrix<T> ans(r, c);
        int i, j, x = 0, y = 0;
        for (i = 0; i < this->m_row; i++)
        {
            for (j = 0; j < this->m_col; j++)
            {
                ans[x][y] = (*data)[i][j];
                y++;
                if (y == c)
                {
                    x++;
                    y = 0;
                }
            }
        }
        return ans;
    }
}

template <typename T>
Matrix<T> Matrix<T>::slice(int r1, int r2, int c1, int c2) const
{
    // assert(r1)
    if (r1 > r2)
    {
        int tmp = r1;
        r1 = r2;
        r2 = tmp;
    }
    if (c1 > c2)
    {
        int tmp = c1;
        c1 = c2;
        c2 = tmp;
    }
    if (r1 < 0)
    {
        if (r2 < 0)
        {
            r2 = 0;
        }
        r1 = 0;
    }
    if (r2 >= this->m_row)
    {
        if (r1 >= this->m_row)
        {
            r1 = this->m_row - 1;
        }
        r2 = this->m_row - 1;
    }
    if (c1 < 0)
    {
        if (c2 < 0)
        {
            c2 = 0;
        }
        c1 = 0;
    }
    if (c2 >= this->m_col)
    {
        if (c1 >= this->m_col)
        {
            c1 = this->m_col - 1;
        }
        c2 = this->m_col - 1;
    }

    Matrix<T> tmp(r2 - r1 + 1, c2 - c1 + 1);
    for (int i = r1; i <= r2; i++)
    {
        for (int j = c1; j <= c2; j++)
        {
            tmp[i - r1][j - c1] = (*data)[i][j];
        }
    }
    return tmp;
}

template <typename T>
T Matrix<T>::sum() const
{
    T sum;
    sum = 0;
    for (int i = 0; i < m_row; i++)
    {
        for (int j = 0; j < m_col; j++)
        {
            sum += (*data)[i][j];
        }
    }
    return (sum);
}

template <typename T>
T Matrix<T>::mean() const
{
    T total;
    total = this->m_row * this->m_col;
    return (this->sum() / total);
}

template <typename T>
T Matrix<T>::max() const
{
    int k = 0, m = 0, i, j;
    for (i = 0; i < this->m_row; i++)
        for (j = 0; j < this->m_col; j++)
            if ((*data)[i][j] > (*data)[k][m])
            {
                k = i;
                m = j;
            }

    return (*data)[k][m];
}

template <typename T>
T Matrix<T>::min() const
{
    int k = 0, m = 0, i = 0, j = 0;
    for (i = 0; i < this->m_row; i++)
    {
        for (j = 0; j < this->m_col; j++)
        {
            if ((*data)[k][m] > (*data)[i][j])
            {
                k = i;
                m = j;
            }
        }
    }
    return (*data)[k][m];
}

template <typename T>
T Matrix<T>::row_max(int row) const
{
    assert(row >= 0 && row < this->m_row);
    int k = 0;
    for (int i = 0; i < this->m_col; i++)
    {
        if ((*data)[row][k] < (*data)[row][i])
            k = i;
    }
    return (*data)[row][k];
}

template <typename T>
T Matrix<T>::row_min(int row) const
{
    assert(row >= 0 && row < this->m_row);
    int k = 0;
    for (int i = 0; i < this->m_col; i++)
        if ((*data)[row][k] > (*data)[row][i])
            k = i;

    return (*data)[row][k];
}

template <typename T>
T Matrix<T>::row_sum(int row) const
{
    assert(row >= 0 && row < this->m_row);
    T row_sum;
    row_sum = 0;
    for (int i = 0; i < this->m_col; i++)
    {
        row_sum += (*data)[row][i];
    }
    return row_sum;
}

template <typename T>
T Matrix<T>::row_mean(int row) const
{
    assert(row >= 0 && row < this->m_row);
    T total;
    total = (this->m_col);
    return this->row_sum(row) / total;
}

template <typename T>
T Matrix<T>::col_max(int col) const
{
    assert(col >= 0 && col < this->m_col);
    int k = 0;
    for (int i = 0; i < this->m_row; i++)
        if ((*data)[k][col] < (*data)[i][col])
            k = i;

    return (*data)[k][col];
}

template <typename T>
T Matrix<T>::col_min(int col) const
{
    assert(col >= 0 && col < this->m_col);
    int k = 0;
    for (int i = 0; i < this->m_row; i++)
        if ((*data)[k][col] > (*data)[i][col])
            k = i;

    return (*data)[k][col];
}

template <typename T>
T Matrix<T>::col_sum(int col) const
{
    assert(col >= 0 && col < this->m_col);
    T col_sum;
    col_sum = 0;
    for (int i = 0; i < this->m_row; i++)
    {
        col_sum += (*data)[i][col];
    }
    return col_sum;
}

template <typename T>
T Matrix<T>::col_mean(int col) const
{
    assert(col >= 0 && col < this->m_col);
    T total;
    total = this->m_row;
    return this->col_sum(col) / total;
}

// from: https://github.com/QuantitativeBytes/qbLinAlg/blob/main/qbQR.h
template <typename T>
pair<Matrix<T>, Matrix<T>> Matrix<T>::QR_decomposition() const
{
    Matrix<T> input(*this);
    vector<Matrix<T>> plist;
    for (int j = 0; j < m_row - 1; j++)
    {
        Matrix<T> a1(1, m_row - j);
        Matrix<T> b1(1, m_row - j);

        for (int i = j; i < m_row; i++)
        {
            a1[0][i - j] = input[i][j];
            b1[0][i - j] = static_cast<T>(0.0);
        }
        b1[0][0] = static_cast<T>(1.0);

        T a1norm = a1.norm();

        T sgn = -1;
        if (a1[0][0] < static_cast<T>(0.0))
        {
            sgn = 1;
        }

        Matrix<T> temp = b1 * sgn * a1norm;
        Matrix<T> u = a1 - temp;
        Matrix<T> n = u.normalized();
        Matrix<T> nTrans = n.Transposition();
        Matrix<T> I(m_row - j, m_row - j);
        I.SetIdentity();

        Matrix<T> temp1 = n * static_cast<T>(2.0);
        Matrix<T> temp2 = nTrans * temp1;
        Matrix<T> Ptemp = I - temp2;

        Matrix<T> P(m_row, m_col);
        P.SetIdentity();

        for (int x = j; x < m_row; x++)
        {
            for (int y = j; y < m_col; y++)
            {
                P[x][y] = Ptemp[x - j][y - j];
            }
        }

        plist.push_back(P);
        input = P * input;
    }

    Matrix<T> qMat = plist[0];
    for (int i = 1; i < m_row - 1; i++)
    {
        Matrix<T> temp3 = plist[i].Transposition();
        qMat = qMat * temp3;
    }

    int numElements = plist.size();
    Matrix<T> rMat = plist[numElements - 1];
    for (int i = (numElements - 2); i >= 0; i--)
    {
        rMat = rMat * plist[i];
    }
    rMat = rMat * (*this);

    return pair<Matrix<T>, Matrix<T>>(qMat, rMat);
}

template <typename T>
T Matrix<T>::norm() const
{
    T cumulativeSum = static_cast<T>(0.0);
    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            cumulativeSum += (*this)[i][j] * (*this)[i][j];

    return sqrt(cumulativeSum);
}

template <typename T>
Matrix<T> Matrix<T>::normalized() const
{
    T norm = this->norm();
    Matrix<T> copy(*this);
    return copy * (static_cast<T>(1.0) / norm);
}

template <typename T>
void Matrix<T>::SetIdentity()
{
    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            if (i == j)
                (*this)[i][j] = static_cast<T>(1.0);
            else
                (*this)[i][j] = static_cast<T>(0.0);
}

// from: https://github.com/QuantitativeBytes/qbLinAlg/blob/main/qbEIG.h
// only work for symmetric metrices
template <typename T>
T *Matrix<T>::eigenvalues(int max_iter)
{
    Matrix<T> A = (*this);
    Matrix<T> identityMatrix(m_row, m_col);
    identityMatrix.SetIdentity();

    for (int i = 0; i < max_iter; i++)
    {
        auto qrResult = A.QR_decomposition();
        A = qrResult.second * qrResult.first;
        if (A.isUpperTri())
            break;
    }

    T *eigenvalues = new T[m_row];
    for (int i = 0; i < m_row; i++)
    {
        eigenvalues[i] = A[i][i];
    }
    return eigenvalues;
}

template <typename T>
bool Matrix<T>::isCloseEnough(T a, T b, double threshold)
{
    return abs(a - b) < static_cast<T>(threshold);
}

template <>
bool Matrix<uchar>::isCloseEnough(uchar a, uchar b, double threshold)
{
    return a == b;
}

template <>
bool Matrix<char>::isCloseEnough(char a, char b, double threshold)
{
    return a == b;
}


template <typename T>
bool Matrix<T>::isUpperTri()
{
    T cumulativeSum = static_cast<T>(0);
    for (int i = 1; i < m_row; i++)
    {
        for (int j = 0; j < i; j++)
        {
            cumulativeSum += (*this)[i][j];
        }
    }
    return isCloseEnough(cumulativeSum, static_cast<T>(0));
}

// from: https://github.com/QuantitativeBytes/qbLinAlg/blob/main/qbEIG.h
template <typename T>
Matrix<T> Matrix<T>::eigenvector(T eigenvalue, int max_iter)
{
    Matrix<T> A = (*this);
    random_device myRandomDevice;
    mt19937 myRandomGenerator(myRandomDevice());
    uniform_int_distribution<int> myDistribution(1.0, 10.0);

    Matrix<T> identityMatrix(m_row, m_col);
    identityMatrix.SetIdentity();

    Matrix<T> v(m_row, 1);
    for (int i = 0; i < m_row; i++)
    {
        v[i][0] = static_cast<T>(myDistribution(myRandomGenerator));
    }

    T deltaThreshold = static_cast<T>(EQ_THRESHOLD);
    T delta = static_cast<T>(1e-6);
    Matrix<T> preVector(m_row, 1);
    Matrix<T> tempMatrix(m_row, m_row);

    for (int i = 0; i < max_iter; i++)
    {
        preVector = v;
        Matrix<T> temp = identityMatrix * eigenvalue;
        tempMatrix = A - temp;
        tempMatrix = tempMatrix.Inverse();
        v = tempMatrix * v;
        v = v.normalized();

        delta = (v - preVector).norm();
        if (delta > deltaThreshold)
            break;
    }
    return v;
}

template <typename T>
Matrix<T> *Matrix<T>::eigenvectors(int max_itr)
{
    Matrix<T> *eigenvectors = new Matrix<T>[m_row];
    T *eigenvalues = this->eigenvalues();
    for (int i = 0; i < m_row; i++)
    {
        eigenvectors[i] = this->eigenvector(*(eigenvalues + i));
    }
    return eigenvectors;
}

// from: https://github.com/QuantitativeBytes/qbLinAlg/blob/main/qbEIG.h
template <typename T>
pair<T, Matrix<T>> Matrix<T>::eigenValueAndEigenVector(int max_itr)
{
    T eigenvalue;
    Matrix<T> inputMatrix = (*this);
    random_device myRandomDevice;
    mt19937 myRandomGenerator(myRandomDevice());
    uniform_int_distribution<int> myDistribution(1.0, 10.0);
    Matrix<T> identityMatrix(m_row, m_col);
    identityMatrix.SetIdentity();

    Matrix<T> v(m_row, 1);
    for (int i = 0; i < m_row; i++)
    {
        v[i][0] = static_cast<T>(static_cast<T>(myDistribution(myRandomGenerator)));
    }
    Matrix<T> v1(m_row, 1);
    for (int i = 0; i < max_itr; i++)
    {
        v1 = inputMatrix * v;
        v1 = v1.normalized();
        v = v1;
    }
    T cumulativeSum = static_cast<T>(0.0);
    for (int i = 1; i < m_row; i++)
    {
        cumulativeSum += inputMatrix[0][i] * v1[i][0];
    }
    eigenvalue = (cumulativeSum / v1[0][0]) + inputMatrix[0][0];
    return pair<T, Matrix<T>>(eigenvalue, v1);
}

// learn from: https://stackoverflow.com/questions/26681713/convert-mat-to-array-vector-in-opencv
template <typename T>
Matrix<T> Matrix<T>::fromOpenCV(const cv::Mat &cvMat)
{
    int row = cvMat.rows;
    int col = cvMat.cols;
    Matrix<T> result(row, col);
    cv::MatConstIterator_<T> it = cvMat.begin<T>(), it_end = cvMat.end<T>();
    vector<T> array;
    if (cvMat.isContinuous())
    {
        array.assign((T *)cvMat.data, (T *)cvMat.data + cvMat.total() * cvMat.channels());
    }
    else
    {
        for (int i = 0; i < cvMat.rows; ++i)
        {
            array.insert(array.end(), cvMat.ptr<T>(i), cvMat.ptr<T>(i) + cvMat.cols * cvMat.channels());
        }
    }
    int cnt = 0;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            result[i][j] = array[cnt++];
        }
    }
    return result;
}

template <typename T>
T** Matrix<T>::toArray() {
    T** array = new T*[m_row];
    for (int i = 0; i < m_row; i++) {
        array[i] = new T[m_col];
    }
    for (int i = 0; i < m_row; i++) {
        for (int j = 0; j < m_col; j++) {
            array[i][j] = (*this)[i][j];
        }
    }
    return array;
}

template <typename T>
cv::Mat* Matrix<T>::toOpenCVMat(int type) {
    cv::Mat* cvMat = new cv::Mat(m_row, m_col, type);
    for (int i = 0; i < m_row; i++) {
        for (int j = 0; j < m_col; j++) {
            (*cvMat).at<T>(i, j) = (*this)[i][j];
        }
    }
    return cvMat;
}

template <typename T>
Matrix<T> Matrix<T>::conv2D(const Matrix<T> &input, const Matrix<T> &kernel, int stride, bool same_padding) {
    Matrix<T> inputMatrix;
    int padding = 0;
    if (same_padding) {
        padding = 1;
    }
    inputMatrix = Matrix<T>(input.m_row + padding * 2, input.m_col + padding * 2);
    for (int i = 0; i < input.m_row; i++) {
        for (int j = 0; j < input.m_col; j++) {
            inputMatrix[i + padding][j + padding] = input[i][j];
        }
    }
    if (padding == 1) {
        for (int i = 0; i < inputMatrix.m_row; i++) inputMatrix[i][0] = inputMatrix[i][inputMatrix.m_col - 1] = 0;
        for (int i = 0; i < inputMatrix.m_col; i++) inputMatrix[0][i] = inputMatrix[inputMatrix.m_row - 1][i] = 0;
    }
    int rowDim = ((input.m_row + 2 * padding - kernel.m_row) / stride) + 1;
    int colDim = ((input.m_col + 2 * padding - kernel.m_col) / stride) + 1;
    Matrix<T> result(rowDim, colDim);
    for (int i = 0; i < rowDim; i++) {
        for (int j = 0; j < colDim; j++) {
            T cumulativeSum = static_cast<T>(0);
            for (int x = 0; x < kernel.m_row; x++) {
                for (int y = 0; y < kernel.m_col; y++) {
                    cumulativeSum += kernel[x][y] * inputMatrix[x + i * stride][y + j * stride];
                }
            }
            result[i][j] = cumulativeSum;
        }
    }
    return result;
}