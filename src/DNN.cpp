#include "math.h"
#include "DNN.h"
#include "Matrix.h"
#include <vector>

const int BATCH_SIZE = 10;
int COUNT = 1;
int BATCH_COUNT = 0;
double CUR_ERROR = 0;

double getError(Matrix<double> &outputError)
{
    double res = 0;
    for (int i = 0; i < outputError.m_row; i++)
    {
        for (int j = 0; j < outputError.m_col; j++)
        {
            res += abs(outputError[i][j]);
        }
    }
    return res;
}

double DNN::sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

void DNN::train(const Matrix<double> *input, const Matrix<double> *target)
{
    for (int i = 0; i < (*input).m_row; i++)
    {
        Matrix<double> inputMat(vector<vector<double>>{(*input)[i]});
        Matrix<double> targetMat(vector<vector<double>>{(*target)[i]});

        inputMat = inputMat.Transposition();
        targetMat = targetMat.Transposition();

        Matrix<double> hiddenInput = m1 * inputMat;
        Matrix<double> hiddenOutput = hiddenInput.transform(DNN::sigmoid);

        Matrix<double> finalInput = m2 * hiddenOutput;
        Matrix<double> finalOutput = finalInput.transform(DNN::sigmoid);

        Matrix<double> outputError = targetMat - finalOutput;
        Matrix<double> hiddenError = m2.Transposition() * outputError;

        m2 += this->learningRate * ((outputError ^ finalOutput ^ (1.0 - finalOutput)) * hiddenOutput.Transposition());
        m1 += this->learningRate * ((hiddenError ^ hiddenOutput ^ (1.0 - hiddenOutput)) * inputMat.Transposition());

        if (COUNT % BATCH_SIZE == 0) {
            cout << "BATCH: " << BATCH_COUNT << ", Error: " << (CUR_ERROR / BATCH_SIZE) << endl;
            BATCH_COUNT++;
            CUR_ERROR = 0;
        }
        COUNT++;
        CUR_ERROR += getError(outputError);
    }
}

vector<int>* DNN::evaluate(const Matrix<double> *input)
{
    vector<int> *result = new vector<int>(); 
    for (int i = 0; i < (*input).m_row; i++)
    {
        Matrix<double> inputMat(vector<vector<double>>{(*input)[i]});
        inputMat = inputMat.Transposition();

        Matrix<double> hiddenInput = m1 * inputMat;
        Matrix<double> hiddenOutput = hiddenInput.transform(DNN::sigmoid);

        Matrix<double> finalInput = m2 * hiddenOutput;
        Matrix<double> finalOutput = finalInput.transform(DNN::sigmoid);
        
        int maxIdx = 0;
        for (int i = 0; i < finalOutput.m_row; i++)
            if (finalOutput[i][0] > finalOutput[maxIdx][0])
                maxIdx = i;
        result->push_back(maxIdx);
    }
    return result;
}

void DNN::initWeight()
{
    default_random_engine generatorOne;
    default_random_engine generatorTwo;

    normal_distribution<double> one(0.0, pow(m1.m_col, -0.5));
    normal_distribution<double> two(0.0, pow(m2.m_col, -0.5));

    for (int i = 0; i < m1.m_row; i++)
        for (int j = 0; j < m1.m_col; j++)
            m1[i][j] = one(generatorOne);
    
    for (int i = 0; i < m2.m_row; i++)
        for (int j = 0; j < m2.m_col; j++)
            m2[i][j] = two(generatorTwo);
}