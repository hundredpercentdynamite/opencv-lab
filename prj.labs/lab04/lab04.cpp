#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include "../../utils/utils.cpp"

cv::Mat getMask(cv::Mat& image) {
  cv::Mat imgCopy;
  image.copyTo(imgCopy);

  cv::threshold(imgCopy, imgCopy, 170, 255, cv::THRESH_OTSU);

  int erosion_type = 0;
  int erosion_size = 1;
  cv::Mat element = getStructuringElement( erosion_type,
                                           cv::Size( 2 * erosion_size + 1, 2 * erosion_size + 1 ),
                                           cv::Point( -1, -1 ) );
  erode( imgCopy, imgCopy, element );
  erosion_size = 4;
  element = getStructuringElement( erosion_type,
                                   cv::Size( 2 * erosion_size + 1, 2 * erosion_size + 1 ),
                                   cv::Point( -1, -1 ) );
  dilate( imgCopy, imgCopy, element, cv::Point(-1, -1), 2);

  return imgCopy;
}

std::vector<int> getFramesIndexes(int total) {
  int proportion = total / 5;
  return std::vector<int>({ proportion * 2, proportion * 3, proportion * 4 });
}

std::vector<cv::Mat> getFrames(cv::VideoCapture& vid) {
  auto frameCount = vid.get(cv::CAP_PROP_FRAME_COUNT);
  auto indexes = getFramesIndexes((int)frameCount);
  std::vector<cv::Mat> frames;
  for (int i : indexes) {
    vid.set(cv::CAP_PROP_POS_FRAMES, i - 1);
    cv::Mat frame;
    vid >> frame;
    frames.push_back(frame);
  }

  return frames;
}

cv::Mat getLargestComponent(cv::Mat& img) {
  cv::Mat colored = getColoredComponents(img);
  cv::Mat components;
  cv::Mat stats;
  cv::Mat centroids;
  int nLabels = cv::connectedComponentsWithStats(img, components, stats, centroids);
  int maxSize = 0;
  int maxLabel = 0;
  for (int l = 1; l < nLabels; l++) {
    int size = stats.at<int>(l, cv::CC_STAT_AREA);
    if (size > maxSize) {
      maxLabel = l;
      maxSize = size;
    }
  }

  cv::Mat component(img.size(), CV_8UC1);
  for(int r = 0; r < component.rows; ++r){
    for(int c = 0; c < component.cols; ++c){
      int lbl = components.at<int>(r, c);
      if (lbl == maxLabel) {
        auto &pixel = component.at<uint8_t>(r, c);
        pixel = (uint8_t)255;
      }
    }
  }

  cv::Mat element = getStructuringElement( 0,
                                           cv::Size( 15, 15 ),
                                           cv::Point( -1, -1 ) );
  cv::morphologyEx(component, component, cv::MORPH_CLOSE, element);

  return component;
}

cv::Mat getDiff(cv::Mat& standart, cv::Mat& mask, cv::Mat& grayscale) {
  cv::Mat dst(grayscale.size(), CV_8UC3);
  cv::cvtColor(grayscale, dst, cv::COLOR_GRAY2BGR);

  for(int r = 0; r < mask.rows; ++r){
    for(int c = 0; c < mask.cols; ++c){
      auto &stdPixel = standart.at<uint8_t>(r, c);
      auto &maskPixel = mask.at<uint8_t>(r, c);
      auto &dstPixel = dst.at<cv::Vec3b>(r, c);
      if (stdPixel == 0 && maskPixel == 255) {
        dstPixel = cv::Vec3b(0, 0, 255);
      }
      if (stdPixel == 255 && maskPixel == 0) {
        dstPixel = cv::Vec3b(0, 255, 0);
      }
    }
  }
  cv::Mat maskCopy(standart.size(), CV_8UC3);
  cv::cvtColor(standart, maskCopy, cv::COLOR_GRAY2BGR);

  cv::addWeighted( dst, 0.4, maskCopy, 0.6, 0.0, dst);
  return dst;
}

double getMaskRatio(cv::Mat& standart, cv::Mat& mask) {
  int standartTotal = cv::countNonZero(standart);
  double commonPixelsTotal = 0;
  for (int r = 0; r < standart.rows; ++r) {
    for (int c = 0; c < standart.cols; ++c) {
      auto stdPixel = standart.at<uint8_t>(r, c);
      auto mskPixel = mask.at<uint8_t>(r, c);
      if (stdPixel == mskPixel && mskPixel == cv::saturate_cast<uint8_t>(255)) {
        commonPixelsTotal += 1;
      }
    }
  }

  double total = commonPixelsTotal / standartTotal;
  return total;
}

int main() {
  const std::vector<const std::string> videosList({
    "../data/video/video_1.MOV",
    "../data/video/video_2.MOV",
    "../data/video/video_3.MOV",
    "../data/video/video_4.MOV",
    "../data/video/video_5.MOV",
    "../data/video/video_6.MOV",
    "../data/video/video_7.MOV"
  });
  std::vector<double> ratios;
  for (int v = 0; v < videosList.size(); v++) {
    std::string videoPath = videosList[v];
    auto video = cv::VideoCapture(videoPath);
    std::vector<cv::Mat> frames = getFrames(video);

    for (int f = 0; f < frames.size(); f++) {
      int videoIndex = v + 1;
      int frameIndex = f + 1;
      cv::Mat frame = frames[f];
      cv::imwrite("frame" + std::to_string(videoIndex) + "_" + std::to_string(frameIndex) + ".jpg", frame);
      cv::Mat grayscale;
      cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY);

      cv::Mat mask = getMask(grayscale);
      cv::Mat finalMask = getLargestComponent(mask);

      std::string maskName = "mask" + std::to_string(videoIndex) + "_" + std::to_string(frameIndex);
      cv::imwrite(maskName + ".jpg", finalMask);
      cv::Mat standart = cv::imread("../data/standart.mask/" + maskName + "_ideal.jpg", CV_8UC1);

      cv::Mat diff = getDiff(standart, finalMask, grayscale);
      double ratio = getMaskRatio(standart, finalMask);
      ratios.push_back(ratio);

      cv::putText(diff, "K: " + toStringWithPrecision<double>(ratio, 3), cv::Point2f(30, 100), 0, 3, { 200, 200, 200 });
      cv::imwrite(maskName + "_diff" + ".jpg", diff);
    }
    video.release();
  }

  double sum = 0;
  for (double ratio : ratios) {
    sum += ratio;
  }
  double result = sum / (double)ratios.size();
  std::cout << "Average ratio: " << toStringWithPrecision(result, 3);

  return 0;
}