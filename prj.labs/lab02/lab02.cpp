#include <opencv2/opencv.hpp>
#include <vector>
#include "../../utils/utils.cpp"

cv::Mat getMosaic(cv::Mat& img) {
  int rows = img.rows;
  int cols = img.cols;

  cv::Mat mosaic(rows * 2, cols * 2, CV_8UC3);
  cv::Mat channels[3];
  cv::split(img, channels);
  cv::Mat zeros = cv::Mat::zeros(rows, cols, CV_8UC1);
  cv::Rect2d rc({0, 0, (double) cols, (double) rows});


  cv::Mat leftUpper = mosaic(rc);
  img.copyTo(leftUpper);

  rc.x = 256;
  cv::Mat rightUpper = mosaic(rc);
  std::vector<cv::Mat> blackRed({zeros, zeros, channels[2]});
  cv::merge(blackRed, rightUpper);

  rc.x = 0;
  rc.y = 256;
  cv::Mat leftLower = mosaic(rc);
  std::vector<cv::Mat> blackGreen({zeros, channels[1], zeros});
  cv::merge(blackGreen, leftLower);

  rc.x = 256;
  cv::Mat rightLower = mosaic(rc);
  std::vector<cv::Mat> blackBlue({channels[0], zeros, zeros});
  cv::merge(blackBlue, rightLower);

  return mosaic;
}

int main() {
  // read png
  auto imgPath = cv::samples::findFile("../data/cross_0256x0256.png");
  cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);

  // write 25% jpeg
  std::vector<int> compression({cv::IMWRITE_JPEG_QUALITY, 25});
  cv::imwrite("cross_0256x0256_025.jpg", img, compression);

  // png-mosaic
  cv::Mat pngMosaic = getMosaic(img);
  cv::imwrite("cross_0256x0256_png_channels.png", pngMosaic);

  // jpg-mosaic
  auto jpgPath = cv::samples::findFile("cross_0256x0256_025.jpg");
  cv::Mat jpg = cv::imread(jpgPath, cv::IMREAD_COLOR);
  cv::Mat jpgMosaic = getMosaic(jpg);
  cv::imwrite("cross_0256x0256_jpg_channels.png", jpgMosaic);


  cv::Mat pngHist = getThreeChannelHist(img);
  cv::Mat jpgHist = getThreeChannelHist(jpg);

  cv::Mat resultHist(400, 1044, CV_8UC3);
  cv::Rect2d rc({ 0, 0, 512, 400 });
  pngHist.copyTo(resultHist(rc));
  rc.x = 532;
  jpgHist.copyTo(resultHist(rc));

  cv::imwrite("cross_0256x0256_hists.png", resultHist);
}
