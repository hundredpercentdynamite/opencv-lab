#include <opencv2/opencv.hpp>
#include <vector>

/// Функция  y = (sin(exp(x^2))^3 + x) / 2
double func(double x) {
  double exp = std::exp(x * x);
  double sin = std::sin(exp);

  return (sin * sin * sin + x) / 2;
}

cv::Mat drawFunc(std::vector<double>& xCords, std::vector<double>& yCords) {
  double size = 517;
  cv::Mat graph = cv::Mat::ones((int)size, (int)size, CV_8UC1) * 255;
  line(graph, cv::Point2d(0, 507), cv::Point2d(517, 507), {0}, 2, 8, 0);
  line(graph, cv::Point2d(5, 517), cv::Point2d(5,  0), {0}, 2, 8, 0);
  cv::putText(graph, "X", cv::Point2f(500, 500), 1, 1, { 0 });
  cv::putText(graph, "Y", cv::Point2f(10, 15), 1, 1, { 0 });
  cv::putText(graph, "0", cv::Point2f(7, 503), 1, 1, { 0 });
  for (int i = 1; i < xCords.size(); ++i) {
    double prevXCord = xCords[i - 1];
    double prevYCord = yCords[i - 1];
    double xCord = xCords[i];
    double yCord = yCords[i];
    cv::Point2d prevPoint(size * prevXCord + 5, size - (size * prevYCord) - 5);
    cv::Point2d point(size * xCord + 5, size - (size * yCord) - 5);
    line( graph, prevPoint, point, { 100 }, 2, 8, 0);
  }

  return graph;
}

int main() {
  auto imgPath = cv::samples::findFile("../data/cross_0256x0256.png");
  cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);

  std::vector<double> xCords(256);
  std::vector<double> yCords(256);
  std::vector<uint8_t> lutY(256);

  for (int i = 0; i < 256; ++i) {
    double xValue = (double)i / 255;
    xCords[i] = xValue;
    double yValue = func(xValue);
    yCords[i] = yValue;
    lutY[i] = cv::saturate_cast<uint8_t>(yValue * 255);
  }


  /// LUT
  cv::Mat grayscale(256, 256, CV_8UC1);
  cv::cvtColor(img, grayscale, cv::COLOR_BGR2GRAY);

  cv::Mat grayscaleLut(256, 256, CV_8UC1);
  cv::LUT(grayscale, lutY, grayscaleLut);

  cv::Mat transformed(256, 256, CV_8UC3);
  cv::LUT(img, lutY, transformed);

  cv::Mat graph = drawFunc(xCords, yCords);

  cv::imwrite("lab03_rgb.png", img);
  cv::imwrite("lab03_gre.png", grayscale);
  cv::imwrite("lab03_gre_res.png", grayscaleLut);
  cv::imwrite("lab03_rgb_res.png", transformed);
  cv::imwrite("lab03_viz_func.png", graph);

  /// Display
  namedWindow("calcHist Demo", cv::WINDOW_AUTOSIZE );
  imshow("calcHist Demo", graph );

  cv::waitKey(0);
  return 0;
}