#include <opencv2/opencv.hpp>
#include <string>

cv::Mat getThreeChannelHist(cv::Mat& img) {
  float range[] = { 0, 256 } ;
  const float* histRange = { range };
  cv::Mat channels[3];
  cv::split(img, channels);
  int histSize = 256;
  bool uniform = true;
  bool accumulate = false;

  cv::Mat b_hist, g_hist, r_hist;

  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  calcHist( &channels[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &channels[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &channels[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

  cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 255,255,255) );

  normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
  normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
  normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

  for( int i = 1; i < histSize; i++ )
  {
    line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
          cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
          cv::Scalar( 255, 0, 0), 2, 8, 0  );
    line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
          cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
          cv::Scalar( 0, 255, 0), 2, 8, 0  );
    line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
          cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
          cv::Scalar( 0, 0, 255), 2, 8, 0  );
  }

  return histImage;
}

cv::Mat getOneChannelHist(cv::Mat& img) {
  float range[] = { 0, 256 } ;
  const float* histRange = { range };
  int histSize = 256;
  bool uniform = true;
  bool accumulate = false;

  cv::Mat g_hist;

  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  calcHist( &img, 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );

  cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 255,255,255) );

  normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

  for( int i = 1; i < histSize; i++ )
  {
    line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
          cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
          cv::Scalar( 0, 255, 0), 2, 8, 0  );
  }

  return histImage;
}

cv::Mat getColoredComponents(cv::Mat& img) {
  cv::Mat components;
  int nLabels = cv::connectedComponents(img, components);
  std::vector<cv::Vec3b> colors(nLabels);
  colors[0] = cv::Vec3b(0, 0, 0);//background
  for(int label = 1; label < nLabels; ++label){
    colors[label] = cv::Vec3b( (rand()&255), (rand()&255), (rand()&255) );
  }
  cv::Mat dst(img.size(), CV_8UC3);
  for(int r = 0; r < dst.rows; ++r){
    for(int c = 0; c < dst.cols; ++c){
      int label = components.at<int>(r, c);
      auto &pixel = dst.at<cv::Vec3b>(r, c);
      pixel = colors[label];
    }
  }

  return dst;
}

template<typename T>
std::string toStringWithPrecision(T value, int precision) {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(precision) << value;
  return ss.str();
}