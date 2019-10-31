#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include<stdlib.h>
using namespace cv;

Mat AddNoise(Mat src)
{
    Mat res;
    src.copyTo(res);
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
        {
            if (!(rand() % 256))
            {
                res.at<Vec3b>(y, x)[0] = 255;
                res.at<Vec3b>(y, x)[1] = 255;
                res.at<Vec3b>(y, x)[2] = 255;
            }
        }
    return res;
}

Mat ArifmeticMean(Mat src)
{
    Mat res;
    src.copyTo(res);
    int Radx = 4;
    int Rady = 4;
    for (int x = 0; x < src.cols; x++)
        for (int y = 0; y < src.rows; y++)
        {
            double resR = 1.0;
            double resG = 1.0;
            double resB = 1.0;
            for (int i = -Rady; i < Rady; i++)
                for (int j = -Radx; j < Radx; j++)
                {
                    int X = std::min(x + j, src.cols - 1);
                    X = std::max(X, 0);
                    int Y = std::min(y + i, src.rows - 1);
                    Y = std::max(Y, 0);
                    Vec3b bgr = src.at<Vec3b>(Y, X);
                    resR += bgr[2];
                    resG += bgr[1];
                    resB += bgr[0];
                }
            res.at<Vec3b>(y, x)[0] = std::min((int)(resB / (64)), 255);
            res.at<Vec3b>(y, x)[1] = std::min((int)(resG / (64)), 255);
            res.at<Vec3b>(y, x)[2] = std::min((int)(resR / (64)), 255);;
        }
    return res;
}

Mat GeometricMean(Mat src)
{
    Mat res;
    src.copyTo(res);
    int Radx = 2;
    int Rady = 2;
    for (int x = 0; x < src.cols; x++)
        for (int y = 0; y < src.rows; y++)
        {
            double resR = 1;
            double resG = 1;
            double resB = 1;
            for (int i = -Rady; i < Rady; i++)
                for (int j = -Radx; j < Radx; j++)
                {
                    int X = std::min(x + j, src.cols - 1);
                    X = std::max(X, 0);
                    int Y = std::min(y + i, src.rows - 1);
                    Y = std::max(Y, 0);
                    Vec3b bgr = src.at<Vec3b>(Y, X);
                    if(bgr[2])
                        resR *= bgr[2];
                    if (bgr[1])
                        resG *= bgr[1];
                    if (bgr[0])
                        resB *= bgr[0];
                }
            int a = pow(resB, 1.0 / (9 * Radx*Rady));
            res.at<Vec3b>(y, x)[0] = std::min(pow(resB, 1.0 / (16)), 255.0);
            res.at<Vec3b>(y, x)[1] = std::min(pow(resG, 1.0 / (16)), 255.0);
            res.at<Vec3b>(y, x)[2] = std::min(pow(resR, 1.0 / (16)), 255.0);
        }
    return res;
}

Mat HarmonicMean(Mat src)
{
    Mat res;
    src.copyTo(res);
    int Radx = 2;
    int Rady = 2;
    for (int x = 0; x < src.cols; x++)
        for (int y = 0; y < src.rows; y++)
        {
            double resR = 0.0;
            double resG = 0.0;
            double resB = 0.0;
            for (int i = -Rady; i < Rady; i++)
                for (int j = -Radx; j < Radx; j++)
                {
                    int X = std::min(x + j, src.cols - 1);
                    X = std::max(X, 0);
                    int Y = std::min(y + i, src.rows - 1);
                    Y = std::max(Y, 0);
                    Vec3b bgr = src.at<Vec3b>(Y, X);
                    resR += 1.0/bgr[2];
                    resG += 1.0/bgr[1];
                    resB += 1.0/bgr[0];
                }
            res.at<Vec3b>(y, x)[0] = std::min((int)(16.0 / resB), 255);
            res.at<Vec3b>(y, x)[1] = std::min((int)(16.0 / resG), 255);
            res.at<Vec3b>(y, x)[2] = std::min((int)(16.0 / resR), 255);
        }
    return res;
}

float GetIntensity(Mat src)
{
  float res = 0;
  for (int y = 0; y < src.rows; y++)
    for (int x = 0; x < src.cols; x++)
    {
      Vec3b bgr = src.at<Vec3b>(y, x);
      res += (bgr[0] + bgr[1] + bgr[2]);
    }
  res = res / (src.rows * src.cols);
  return res;
}

float GetContrast(Mat src)
{
  float res = 0;
  float M = GetIntensity(src);
  for (int y = 0; y < src.rows; y++)
    for (int x = 0; x < src.cols; x++)
    {
      Vec3b bgr = src.at<Vec3b>(y, x);
      res += pow((bgr[0] + bgr[1] + bgr[2]) - M, 2);
    }
  res = sqrt(res / (src.rows * src.cols));
  return res;
}

float GetCov(Mat src1, Mat src2)
{
  if (src1.size != src2.size)
    throw 1;
  float res = 0;
  float M1 = GetIntensity(src1);
  float M2 = GetIntensity(src2);
  for (int y = 0; y < src1.rows; y++)
    for (int x = 0; x < src1.cols; x++)
    {
      Vec3b bgr1 = src1.at<Vec3b>(y, x);
      Vec3b bgr2 = src2.at<Vec3b>(y, x);
      res += (bgr1[0] + bgr1[1] + bgr1[2] - M1)*(bgr2[0] + bgr2[1] + bgr2[2] - M2);
    }
  return res / (src1.rows * src1.cols);
}

float SSIMMetric(Mat src1, Mat src2)
{
  return (2.f * GetIntensity(src1) * GetIntensity(src2))*(2 * GetCov(src1, src2))/((GetIntensity(src1) * GetIntensity(src1) + GetIntensity(src2) * GetIntensity(src2)) * (GetContrast(src1) * GetContrast(src1) + GetContrast(src2) * GetContrast(src2)));
}


int main(int argc, char** argv)
{
    srand(time(NULL));
    Mat img = imread("C:\\Users\\dimen\\Pictures\\sarcasm.jpg");
    Mat noise_img = AddNoise(img);
    Mat geometric_img = GeometricMean(noise_img);
    Mat arifmetic_img = ArifmeticMean(noise_img);
    Mat harmonic_img = HarmonicMean(noise_img);
    imshow("before", noise_img);
    imshow("Geometric", geometric_img);
    imshow("Arifmetic", arifmetic_img);
    imshow("Harmonic", harmonic_img);
    std::cout << "SSIM for arifmeic mean " << SSIMMetric(arifmetic_img, img) << std::endl;
    std::cout << "SSIM for geometric mean " << SSIMMetric(geometric_img, img) << std::endl;
    std::cout << "SSIM for harmonic mean " << SSIMMetric(harmonic_img, img) << std::endl;
    waitKey();
    return 0;
}