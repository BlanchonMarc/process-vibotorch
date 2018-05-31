#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <iomanip>
#include <sstream>

using namespace cv;

void ExponentialMap(const cv::Mat1d& v, const double& d, cv::Mat1d & Trans);
void InterractionMatrix(const cv::Mat1d& K, const Mat& Ic, cv::Mat1d & L);
void SSD2D(const Mat& Is, const Mat& Ic, Mat1d& Error);
long SSD(const Mat& Is, const Mat& Ic);
void readme();
std::string ZeroPadNumber(int num);

/** @function main */
int main( int argc, char** argv )
{
  if( argc != 10 )
  { readme(); return -1; }

  //Images

  std::string pathKin, pathIDS, pathPola, pathNIR;
  std::string outKin, outIDS, outPola, outNIR;
  std::string mask;

  pathKin = argv[1];
  pathIDS = argv[2];
  pathPola = argv[3];
  pathNIR = argv[4];
  mask = argv[5];
  outKin = argv[6];
  outIDS = argv[7];
  outPola = argv[8];
  outNIR = argv[9];

  std::vector<float> corner1X, corner1Y, corner2X, corner2Y, corner3X, corner3Y, corner4X, corner4Y;
  std::vector<Point2f> corner1, corner2, corner3, corner4;

  std::vector<std::string> imagename;
  int n_image = 204;
  std::vector<Mat> vec; vec.resize(3);

  for(int i = 1 ; i < n_image + 1; i++){imagename.push_back("/image_"+ZeroPadNumber(i)+".tiff");}
  // Mat img_scene = imread( pathKin + imagename[203] , CV_LOAD_IMAGE_GRAYSCALE );
  // imshow("Hello",img_scene);
  // waitKey(0);
  // return 0;
  Mat img_scene;
  Mat img_mask = imread( mask );
  Mat img_object_diff;
  std::vector<cv::Mat> img_object_warped;

  int indx = 10;

  std::vector<Mat> H;

  img_scene = imread( pathKin + imagename[indx]);
  vec[0] = imread( pathIDS + imagename[indx]);
  vec[1] = imread( pathPola + imagename[indx]);
  vec[2] = imread( pathNIR + imagename[indx]);

  cvtColor(img_mask, img_mask, cv::COLOR_RGB2GRAY);
  cvtColor(img_scene, img_scene, cv::COLOR_RGB2GRAY);
  cvtColor(vec[0], vec[0], cv::COLOR_RGB2GRAY);
  cvtColor(vec[1], vec[1], cv::COLOR_RGB2GRAY);
  cvtColor(vec[2], vec[2], cv::COLOR_RGB2GRAY);


  if( !vec[0].data || !img_scene.data || !vec[1].data || !vec[2].data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }
  // int cnt = 0;
  for (std::vector<Mat>::iterator it = vec.begin() ; it != vec.end(); ++it){
      //-- Step 1: Detect the keypoints using SURF Detector
      int minHessian = 1000;

      //cv::xfeatures2d::SiftFeatureDetector detector;
     // cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create();
      cv::Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
      std::vector<KeyPoint> keypoints_object, keypoints_scene;

      detector->detect( *it, keypoints_object );
      detector->detect( img_scene, keypoints_scene, img_mask );

      //-- Step 2: Calculate descriptors (feature vectors)

      Mat descriptors_object, descriptors_scene;

      detector->compute( *it, keypoints_object, descriptors_object );
      detector->compute( img_scene, keypoints_scene, descriptors_scene );

      //-- Step 3: Matching descriptor vectors using FLANN matcher
      FlannBasedMatcher matcher;
      std::vector< DMatch > matches;
      matcher.match( descriptors_object, descriptors_scene, matches );

      double max_dist = 0; double min_dist = 150;

      //-- Quick calculation of max and min distances between keypoints
      for( int i = 0; i < descriptors_object.rows; i++ )
      { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
      }

      //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
      std::vector< DMatch > good_matches;

      for( int i = 0; i < descriptors_object.rows; i++ )
      { if( matches[i].distance < 3*min_dist )
         { good_matches.push_back( matches[i]); }
      }

      Mat img_matches;
      drawMatches( *it, keypoints_object, img_scene, keypoints_scene,
                   good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                   std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

      //-- Localize the object
      std::vector<Point2f> obj;
      std::vector<Point2f> scene;

      for( int i = 0; i < good_matches.size(); i++ )
      {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
      }

      H.push_back(findHomography( obj, scene, FM_RANSAC, 4 ));

      //-- Get the corners from the image_1 ( the object to be "detected" )
      std::vector<Point2f> obj_corners(4);
      obj_corners[0] = cv::Point(0,0);
      obj_corners[1] = cv::Point( it->cols, 0 );
      obj_corners[2] = cv::Point( it->cols, it->rows );
      obj_corners[3] = cv::Point( 0, it->rows );
      std::vector<Point2f> scene_corners(4);
      img_object_warped.push_back(*it);
      perspectiveTransform( obj_corners, scene_corners, H.back());
      warpPerspective(*it,img_object_warped.back(),H.back(),img_scene.size());


      absdiff(img_scene, img_object_warped.back(), img_object_diff);

      corner1.push_back(scene_corners[0]);
      corner2.push_back(scene_corners[1]);
      corner3.push_back(scene_corners[2]);
      corner4.push_back(scene_corners[3]);


      corner1X.push_back(corner1.back().x);
      corner1Y.push_back(corner1.back().y);

      corner2X.push_back(corner2.back().x);
      corner2Y.push_back(corner2.back().y);

      corner3X.push_back(corner3.back().x);
      corner3Y.push_back(corner3.back().y);

      corner4X.push_back(corner4.back().x);
      corner4Y.push_back(corner4.back().y);


  }

  float maxX14 = max(*max_element(corner1X.begin(), corner1X.end()), *max_element(corner4X.begin(), corner4X.end()));
  float maxY12 = max(*max_element(corner1Y.begin(), corner1Y.end()), *max_element(corner2Y.begin(), corner2Y.end()));
  float minX23 = min(*min_element(corner2X.begin(), corner2X.end()), *min_element(corner3X.begin(), corner3X.end()));
  float minY34 = min(*min_element(corner3Y.begin(), corner3Y.end()), *min_element(corner4Y.begin(), corner4Y.end()));

  //intrinsicsKin = [533.7370635979429, 534.4286075842739, 486.3393110591146, 251.75124952046906];
  //instrinsicsNIR = [1239.3251374485167, 1242.4590777209896, 327.0358191974078, 179.40393683791095]

  //Mat K_Kin = Mat::eye(3, 3, CV_64F);

  // imshow("First Warp" , img_object_warped[0]);



/////////////////////

  int verbose = 0;

  float K_00 = 533.7370635979429;
  float K_11 = 534.4286075842739;
  float K_02 = 486.3393110591146;
  float K_12 = 251.75124952046906;

  cv::Mat1d K_Kin = (cv::Mat1d(3, 3) <<
    533.7370635979429, 0, 486.3393110591146,
    0, 534.4286075842739, 251.75124952046906,
    0, 0,   1.0);

  std::vector<cv::Mat> R, t, Normals;

  if(verbose == 1){
  std::cout << "__________________________________________" << std::endl;
  std::cout << "Homography" << std::endl;
  std::cout << H[0] << std::endl;
  std::cout << "__________________________________________" << std::endl;
  std::cout << "K_Kinect" << std::endl;
  std::cout << K_Kin << std::endl;
  }
  decomposeHomographyMat(H[0], K_Kin, R, t, Normals);
  if(verbose == 1){
  std::cout << "__________________________________________" << std::endl;
  std::cout << "R:" << std::endl;
  for (std::vector<Mat>::iterator it = R.begin() ; it != R.end(); ++it){std::cout << *it << std::endl << std::endl;}
  std::cout << "__________________________________________" << std::endl;
  std::cout << "T:" << std::endl;
  for (std::vector<Mat>::iterator it = t.begin() ; it != t.end(); ++it){std::cout << *it << std::endl << std::endl;}
  std::cout << "__________________________________________" << std::endl;
  std::cout << "Normals:" << std::endl;
  for (std::vector<Mat>::iterator it = Normals.begin() ; it != Normals.end(); ++it){std::cout << *it << std::endl << std::endl;}

  std::cout << "__________________________________________" << std::endl;
  std::cout << "Estimated Homography 1:" << std::endl;
  }
  cv::Mat H2 = R[0] + (t[0] * Normals[0].t());
  H2 = K_Kin * H2 * K_Kin.inv();
  H2 = H2 / H2.at<double>(2,2);

  if(verbose == 1){
  std::cout << H2 << std::endl;

  std::cout << "__________________________________________" << std::endl;
  std::cout << "Estimated Homography 2:" << std::endl;
  }
  cv::Mat H3 = R[1] + (t[1] * Normals[1].t());
  H3 = K_Kin * H3 * K_Kin.inv();
  H3 = H3 / H3.at<double>(2,2);
  if(verbose == 1){
  std::cout << H3 << std::endl;

  std::cout << "__________________________________________" << std::endl;
  std::cout << "Estimated Homography 3:" << std::endl;
  }
  cv::Mat H4 = R[2] + (t[2] * Normals[2].t());
  H4 = K_Kin * H4 * K_Kin.inv();
  H4 = H4 / H4.at<double>(2,2);
  if(verbose == 1){
  std::cout << H4 << std::endl;

  std::cout << "__________________________________________" << std::endl;
  std::cout << "Estimated Homography 4:" << std::endl;
  }
  cv::Mat H5 = R[3] + (t[3] * Normals[3].t());
  H5 = K_Kin * H5 * K_Kin.inv();
  H5 = H5 / H5.at<double>(2,2);
  if(verbose == 1){
  std::cout << H5 << std::endl;

  cv::Mat1d P1 = (cv::Mat1d(3, 1) <<
    3,
    5,
    1);

  std::cout << "__________________________________________" << std::endl;
  std::cout << "Estimated Point" << std::endl;
  std::cout << H[0] * P1 << std::endl;
  std::cout << "__________________________________________" << std::endl;
  std::cout << "Estimated Point" << std::endl;
  std::cout << H2 * P1 << std::endl;
  std::cout << "__________________________________________" << std::endl;
  std::cout << "Estimated Point" << std::endl;
  std::cout << H3 * P1 << std::endl;
  std::cout << "__________________________________________" << std::endl;
  std::cout << "Estimated Point" << std::endl;
  std::cout << H4 * P1 << std::endl;
  std::cout << "__________________________________________" << std::endl;
  std::cout << "Estimated Point" << std::endl;
  std::cout << H5 * P1 << std::endl;
  }
  unsigned int it, MAX_IT;
  it = 0; MAX_IT = 1000;

  double Lambda = 0.8;
  Mat1d Re, Te, Ne;
  Mat He, Ic, Id, Ii;
  long err;

  Re = R[0];
  Te = t[0];
  Ne = Normals[0];
  int synthesis = 1;

  if(synthesis == 1){

      img_scene = imread( "/Users/marc/Downloads/Kinect_Synthese.tiff");
      cvtColor(img_scene, img_scene, cv::COLOR_RGB2GRAY);


      Ii = imread( "/Users/marc/Downloads/Kinect_Synthese.tiff");
      cvtColor(Ii, Ii, cv::COLOR_RGB2GRAY);


      Mat Warped2;
      Mat1d RotNew = (Mat1d(3,3) <<
      1.0 , 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0);

      Mat1d TransNew = (Mat1d(3,1) <<
      0.02 ,
      0.02,
      0.02);

      Mat1d NNew = (Mat1d(3,1) <<
      0 ,
      0,
      -1.0);

      cv::Mat HNew = RotNew + (TransNew * NNew.t());
      HNew = K_Kin * HNew * K_Kin.inv();
      HNew = HNew / HNew.at<double>(2,2);

      Re = RotNew;
      Te = TransNew;
      Ne = NNew;

      warpPerspective(Ii,Warped2,HNew,img_scene.size());

      Id = img_scene;


      Ic = Warped2;

    }else{
        int n = 40;
        img_scene = imread(  pathKin + imagename[n]);
        cvtColor(img_scene, img_scene, cv::COLOR_RGB2GRAY);


        Ii = imread(  pathIDS + imagename[n]);
        cvtColor(Ii, Ii, cv::COLOR_RGB2GRAY);

        Re = R[3];
        Te = t[3];
        Ne = Normals[3];
        Mat Warped2;
        warpPerspective(Ii,Warped2,H[0],img_scene.size());

        Id = img_scene;


        Ic = Warped2;

  }

  imshow("Id" , Id);
  imshow("Ii" , Ic);



  while(it < MAX_IT){
      Mat1d L, error, v, Linv, Trans;

      SSD2D(Id, Ic, error);

      InterractionMatrix(K_Kin, Ic, L);

      invert(L, Linv, DECOMP_SVD);
      v = - Lambda * Linv * error;
      double x, y, z, rx, ry, rz;

      x=v.at<double>(0,0);
      y=v.at<double>(0,1);
      z=v.at<double>(0,2);

      rx=v.at<double>(0,3);
      ry=v.at<double>(0,4);
      rz=v.at<double>(0,5);

      if(verbose == 0){
      std::cout << "__________________________________________" << std::endl;
      std::cout<<"Velocity"<<std::endl;
      std::cout<<v.t()<<std::endl;
      std::cout << "__________________________________________" << std::endl;
      std::cout<<"Error"<<std::endl;
      long err = SSD(Id, Ic);
      std::cout<<err<<std::endl;
      }

      Mat1d v2 = (Mat1d(6,1) << x, y, z, rx, ry, rz);
      ExponentialMap(v2, 1.0, Trans);

      if(verbose == 1){
      std::cout << "__________________________________________" << std::endl;
      std::cout<<"Transformation"<<std::endl;
      std::cout<<Trans<<std::endl;
      }

      if(it > 0){
          Mat1d Me = (Mat1d(4,4)<<
          Re.at<double>(0,0),                      Re.at<double>(0,1),        Re.at<double>(0,2),   Te.at<double>(0,0),
          Re.at<double>(1,0),                      Re.at<double>(1,1),        Re.at<double>(1,2),   Te.at<double>(1,0),
          Re.at<double>(2,0),                      Re.at<double>(2,1),        Re.at<double>(2,2),   Te.at<double>(2,0),
          0.0,                      0.0,                            0.0,                        1.0);

          Mat1d M = Trans * Me ;

          Re = (cv::Mat1d(3,3)<<
            M.at<double>(0,0),                      M.at<double>(0,1),        M.at<double>(0,2),
            M.at<double>(1,0),                      M.at<double>(1,1),        M.at<double>(1,2),
            M.at<double>(2,0),                      M.at<double>(2,1),        M.at<double>(2,2));

          Te = (cv::Mat1d(3,1)<<
            M.at<double>(0,3),
            M.at<double>(1,3),
            M.at<double>(2,3));


          He = Re + (Te * Ne.t()) ;
          He = K_Kin * He * K_Kin.inv();
          He = He / He.at<double>(2,2);

          warpPerspective(Ii,Ic,He,img_scene.size());

      }
      it++;

      absdiff(Id, Ic, img_object_diff);
      imshow("Ic" , Ic);
      imshow("New Diff" , img_object_diff);

      waitKey(20);

  }

return 0;

}

  /** @function readme */
void readme()
{ std::cout << " Incorrect amount of arguments" << std::endl; }


std::string ZeroPadNumber(int num){
    std::ostringstream ss;
    ss << std::setw( 5 ) << std::setfill( '0' ) << num;
    return ss.str();
}

long SSD(const Mat& Is, const Mat& Ic){
    long SumDifference = 0;
    for(int indy = 0; indy < Ic.rows; indy++){
        for(int indx = 0; indx < Ic.cols; indx++){
            if(Ic.at<unsigned char>(indy, indx) != 0){
                int t_tmp_orig = Is.at<unsigned char>(indy, indx);
                int t_tmp = Ic.at<unsigned char>(indy, indx);
                SumDifference += long((t_tmp - t_tmp_orig) * (t_tmp - t_tmp_orig));
            }
        }
    }
    return SumDifference;
}

void SSD2D(const Mat& Is, const Mat& Ic, Mat1d& Error){
    int t_cnt = 0;
    for(int indx = 0; indx < Ic.rows; indx++){
        for(int indy = 0; indy < Ic.cols; indy++){
            if(Ic.at<unsigned char>(indx, indy) != 0){
            t_cnt++;
            }
        }
    }
    Error = cv::Mat1d(t_cnt, 1);

    int cnt = 0;
    long SumDifference = 0;
    for(int indy = 0; indy < Ic.rows; indy++){
        for(int indx = 0; indx < Ic.cols; indx++){
            if(Ic.at<unsigned char>(indy, indx) != 0){
                int t_tmp_orig = Is.at<unsigned char>(indy, indx);
                int t_tmp = Ic.at<unsigned char>(indy, indx);
                Error.at<double>(cnt,0) = double(t_tmp - t_tmp_orig);
                cnt ++;
            }

        }
    }
}

void InterractionMatrix(const cv::Mat1d& K, const Mat& Ic, cv::Mat1d & L){
    Mat grad_x, grad_y;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;
    int counter = 0;

    Sobel( Ic, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    Sobel( Ic, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

    for(int n = 0 ; n < grad_x.rows ; n++){
        for(int m = 0 ; m < grad_x.cols ; m++){
            if(Ic.at<unsigned char>(n, m) != 0){
                counter++;
            }
        }
    }

    Mat1f Grad = cv::Mat1f(counter, 2);

    int t_cnt = 0;
    for(int n = 0 ; n < grad_x.rows ; n++){
        for(int m = 0 ; m < grad_x.cols ; m++){
            if(Ic.at<unsigned char>(n, m) != 0){
                Grad.at<float>(t_cnt,0) = grad_x.at<float>(n,m);
                Grad.at<float>(t_cnt,1) = grad_y.at<float>(n,m);
                t_cnt ++;
            }
        }
    }

    L = Mat1d(counter, 6);

    double x, y, Z, Zinv, Ix, Iy;
    int u, v;
    Z = 1.0;
    Zinv = 1.0/Z;

    int index = 0;
    for(int ind = 0 ; ind < Ic.rows * Ic.cols ; ind++ ){
        u = ind % Ic.cols ;
        v = ind / Ic.cols ;

        if (int(Ic.at<unsigned char>(v, u)) != 0){
            Ix = grad_x.at<float>(v,u) * K.at<double>(0,0);
            Iy = grad_y.at<float>(v,u) * K.at<double>(1,1);

            x = (double(u) - K.at<double>(0,2)) / K.at<double>(0,0);
            y = (double(v) - K.at<double>(1,2)) / K.at<double>(1,1);


            L.at<double>(index,0) = Ix * Zinv;
            L.at<double>(index,1) = Iy * Zinv;
            L.at<double>(index,2) = -(x*Ix+y*Iy)*Zinv;
            L.at<double>(index,3) = -Ix*x*y-(1+y*y)*Iy;
            L.at<double>(index,4) = (1+x*x)*Ix + Iy*x*y;
            L.at<double>(index,5) = Iy*x - Ix*y;
            index ++;
        }
    }
}

void ExponentialMap(const cv::Mat1d& v, const double& d, cv::Mat1d & Trans){

    cv::Mat1d M = (cv::Mat1d(4,4)<<
      1.0,                      0.0 - v.at<double>(0,5),        0.0 + v.at<double>(0,4),    0.0 + v.at<double>(0,0),
      0.0 + v.at<double>(0,5),  1.0,                            0.0 - v.at<double>(0,3),    0.0 + v.at<double>(0,1),
      0.0 - v.at<double>(0,4),  0.0 + v.at<double>(0,3),        1.0,                        0.0 + v.at<double>(0,2),
      0.0,                      0.0,                            0.0,                        1.0);


    Mat1d t1 = (cv::Mat1d(3,1)<<
                M.at<double>(0,1),
                M.at<double>(1,1),
                M.at<double>(2,1));

    Mat1d t2 = (cv::Mat1d(3,1)<<
                M.at<double>(0,2),
                M.at<double>(1,2),
                M.at<double>(2,2));

    Mat1d n = cv::Mat1d(3,1);
    n = t1.cross(t2);

      Mat1d RotTrans = (cv::Mat1d(3,3)<<
      n.at<double>(0,0),                      M.at<double>(0,1),        M.at<double>(0,2),
      n.at<double>(1,0),                      M.at<double>(1,1),        M.at<double>(1,2),
      n.at<double>(2,0),                      M.at<double>(2,1),        M.at<double>(2,2));
      RotTrans = RotTrans.inv();

      Trans = (cv::Mat1d(4,4)<<
        RotTrans.at<double>(0,0),                      RotTrans.at<double>(0,1),        RotTrans.at<double>(0,2),    M.at<double>(0,3),
        RotTrans.at<double>(1,0),                      RotTrans.at<double>(1,1),        RotTrans.at<double>(1,2),    M.at<double>(1,3),
        RotTrans.at<double>(2,0),                      RotTrans.at<double>(2,1),        RotTrans.at<double>(2,2),    M.at<double>(2,3),
        0.0, 0.0, 0.0, 1.0);
}
