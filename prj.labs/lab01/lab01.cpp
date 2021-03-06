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
  cv::Mat img(180, 768, CV_8UC1);
  // draw gradient
  img = 0;
  drawGradient(img, 0, 0);


  // draw gamma with pow
  drawGradient(img, 0, 60);
  auto startPow = std::chrono::high_resolution_clock::now();
  cv::Rect2d rc3 = {0, 60, 768, 60 };
  img.convertTo(img, CV_32FC1);
  cv::Mat thirdrect = img(rc3);
  thirdrect = thirdrect / 255;
  cv::Mat tmp;
  cv::pow(thirdrect, 2.2, tmp);
  tmp.copyTo(thirdrect);
  thirdrect = thirdrect * 255;
  img.convertTo(img, CV_8UC1);
  auto stopPow = std::chrono::high_resolution_clock::now();
  auto powDuration = std::chrono::duration_cast<std::chrono::milliseconds>(stopPow - startPow);
  std::cout << "Pow duration: " << powDuration.count() << std::endl;

  // draw gamma
  drawGradient(img, 0, 120);
  auto start = std::chrono::high_resolution_clock::now();
  cv::Rect2d rc2 = {0, 120, 768, 60 };
  cv::Mat secondrect = img(rc2);
  for (int i = 0; i <= rc2.height; i++) {
    for (int j = 0; j <= rc2.width; j++) {
      auto& point = secondrect.at<uint8_t>(i, j);
      float corrected = powf(float(point) / float(255), 2.2) * 255;
      point = cv::saturate_cast<u_char>(corrected);
    }
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Point access duration: " << duration.count() << std::endl;
  //   save result
  cv::imwrite("lab01.png", img);
}
