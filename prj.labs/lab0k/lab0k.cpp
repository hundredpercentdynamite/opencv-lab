#include <opencv2/opencv.hpp>
#include <iostream>

void drawGradient (cv::Mat& mx, double x, double y) {
  cv::Rect2d rc = {x, y, 768, 60 };
  cv::Mat gradient = mx(rc);
  for (int i = 0; i <= rc.height; i++) {
    uint8_t color = 0;
    for (int j = 1; j <= rc.width; j += 3) {
      auto& point_l = gradient.at<uint8_t>(i, j - 1);
      auto& point = gradient.at<uint8_t>(i, j);
      auto& point_r = gradient.at<uint8_t>(i, j + 1);
      point_l = color;
      point = color;
      point_r = color;
      color += 1;
    }
  }
}

int main() {
  cv::Mat img(300, 450, CV_32FC1);
  std::vector<std::vector<float>> colors = {{0, 127, 255}, { 127, 255, 0 }};
  int width = 150;
  int radius = 75;
  cv::Rect2d rect(0, 0, width, width);

  for (int i = 0; i < 2; ++i) {
    rect.y = width * i;
    for (int j = 0; j < 3; ++j) {
      rect.x = width * j;
      cv::Mat zone = img(rect);
      zone = colors[i][j] / 255;
      cv::Point center = cv::Point((int)(rect.x + radius), (int)(rect.y + radius));
      cv::circle(img, center, 50, colors[1 - i][j] / 255, -1);
    }
  }

  cv::namedWindow("image");
  cv::imshow("image", img);
  cv::imwrite("image.png", img * 255);


  cv::Mat kernel1 = cv::Mat::zeros(2, 2, CV_32FC1);
  kernel1.at<float>(0, 1) = 1;  // 0 1
  kernel1.at<float>(1, 0) = -1; // 1 0
  cv::Mat filtered1;
  cv::filter2D(img, filtered1, -1, kernel1, cv::Point(0, 0));




  cv::Mat kernel2 = cv::Mat::zeros(2, 2, CV_32FC1);
  kernel2.at<float>(0, 0) = 1;
  kernel2.at<float>(1, 1) = -1;
  cv::Mat filtered2;
  cv::filter2D(img, filtered2, -1, kernel2, cv::Point(0, 0));

  cv::Mat middle = cv::Mat::zeros(300, 450, CV_32FC1);
  for (int i = 0; i < filtered1.rows; ++i) {
    for (int j = 0; j < filtered1.cols; ++j) {
      float f1 = filtered1.at<float>(i, j);
      float f2 = filtered2.at<float>(i, j);
      float calc = std::sqrt(f1 * f1 + f2 * f2);
      middle.at<float>(i, j) = calc;
    }
  }

  filtered1 = (filtered1 + 1) / 2;
  filtered2 = (filtered2 + 1) / 2;
  middle = middle * (1 / std::sqrt(2));
  cv::namedWindow("fiilter1");
  cv::imshow("fiilter1", filtered1);
  cv::imwrite("filter1.png", filtered1 * 255);

  cv::namedWindow("fiilter2");
  cv::imshow("fiilter2", filtered2);
  cv::imwrite("filter2.png", filtered2 * 255);

  cv::namedWindow("middle");
  cv::imshow("middle", middle);
  cv::imwrite("middle.png", middle * 255);

  cv::waitKey();
}
