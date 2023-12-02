#include <iostream>
#include "imagefusion.h"
const int imageWidth = 384;                               //【可能需要修改的程序参数1】：单目图像的宽度
const int imageHeight = 288;                              //【可能需要修改的程序参数2】：单目图像的高度
Size imageSize = Size(imageWidth, imageHeight);
Rect validROIL, validROIR;
Mat Rl, Rr, Pl, Pr, Q;



int main() {

    cv::Mat homography(3, 3, CV_64F);

    // 设置单应性矩阵的值（请根据实际矩阵设置值）
    homography.at<double>(0, 0) = 0.8879300854939006;
    homography.at<double>(0, 1) = 0.2762556360860489;
    homography.at<double>(0, 2) = -77.10765576954297;
    homography.at<double>(1, 0) = -0.1510636181764731;
    homography.at<double>(1, 1) = 0.2562562744501071;
    homography.at<double>(1, 2) = 133.8627898417418;
    homography.at<double>(2, 0) = -0.0003091258322138193;
    homography.at<double>(2, 1) = -0.0002902757393753061;
    homography.at<double>(2, 2) = 1.0;

    Mat rgbImageL, rgbImageR, grayImageL, grayImageR,thermalImage;
    Mat rectifyImageL, rectifyImageR;				//立体校正后的图像
    Mat mapLx, mapLy, mapRx, mapRy;                           //left&right映射表
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
                  -1, imageSize, &validROIL, &validROIR);
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

    //--//【可能需要修改的程序参数5】：读取图片--------------------------------------------------------------
    rgbImageL = imread("D:\\code\\CLionProjects\\imagefusion-trans\\left\\30.jpg");//Left
//    resize(rgbImageL, rgbImageL, Size(rgbImageL.cols * 0.785, rgbImageL.rows * 0.785), 0, 0, INTER_NEAREST);//缩放图像
//    rgbImageL = rgbImageL(cv::Rect(330, 90, 383, 287)); // 裁剪后的图

    rgbImageR = imread("D:\\code\\CLionProjects\\imagefusion-trans\\right\\30.jpg");//Right
//    resize(rgbImageR, rgbImageR, Size(rgbImageR.cols * 0.785, rgbImageR.rows * 0.785), 0, 0, INTER_NEAREST);//缩放图像
//    rgbImageR = rgbImageR(cv::Rect(330, 90, 383, 287)); // 裁剪后的图

    thermalImage = imread("D:\\code\\CLionProjects\\imagefusion-trans\\ir\\30.jpg"); //读取热红外图像IR (31).jpg


    namedWindow("ImageTIR Before Rectify", 1);  imshow("ImageTIR Before Rectify", thermalImage);

    //--经过remap之后，左右相机的图像已经共面并且行对齐-------------------------------------------------------
    remap(rgbImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
    imwrite("rectifyImageL.jpg", rectifyImageL);
    remap(rgbImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);
    imwrite("rectifyImageR.jpg", rectifyImageR);
    //--把左右相机校正结果显示出来--------------------------------------------------------------------------
    namedWindow("LR_ImageL After Rectify", 1); imshow("LR_ImageL After Rectify", rectifyImageL);
    namedWindow("LR_ImageR After Rectify", 1); imshow("LR_ImageR After Rectify", rectifyImageR);
    ImageFusion fusion;
    vector<KeyPoint> KeyPointL,KeyPointR;
    vector<DMatch> good_matches;
    vector<Point2f> good_keypoints_L;
    vector<Point2f> good_keypoints_R;
    vector< Point3f >worldcoordinates;
    vector<Point2f> imagePoints;
    //Surf特征点匹配,并计算相应特征点对的世界坐标
    fusion.OpenSurf(rectifyImageL,rectifyImageR,KeyPointL,KeyPointR,good_matches,good_keypoints_L,good_keypoints_R);

    //--把左右相机校正结果显示出来--------------------------------------------------------------------------
    namedWindow("LR_ImageL After surf", 1); imshow("LR_ImageL After surf", rectifyImageL);
    namedWindow("LR_ImageR After surf", 1); imshow("LR_ImageR After surf", rectifyImageR);
//    //三角测量求世界坐标
//    fusion.triangulation(KeyPointL, KeyPointR, good_matches, R, T, worldcoordinates);
//    //计算世界坐标系投影到热红外坐标系上
//    vector< Point3f > imagePoints3f_2f = fusion.World_Pixel_Project(worldcoordinates);
//    convertPointsFromHomogeneous(imagePoints3f_2f, imagePoints);
    fusion.FindWarpTrans(good_keypoints_L,imagePoints);

    for (int i = 0; i < imagePoints.size(); i++)
    {
        cout << "World Points Coordinate in TIR" << i << imagePoints[i] << endl;

        circle(thermalImage, cvPoint(imagePoints[i].x, imagePoints[i].y), 3, cvScalar(0, 0, 255), -1);
    }
    namedWindow("Draw ProjectPoints In TIR", 1);  imshow("Draw ProjectPoints In TIR", thermalImage);

    //findHomography使用RANSAC从热图像和一幅彩色图像中选择四对点进行最佳单应变换
    fusion.FindHomographyRANSAC(rectifyImageL,thermalImage,good_keypoints_L,imagePoints,good_matches);
    cv::waitKey(0);
    std::cout << "Hello, World!" << std::endl;
    std::cout <<CV_VERSION<<std::endl;
    return 0;
}

//int main() {
//
//        cv::Mat image2 = cv::imread("/home/lishuai/CLionProjects/imagefusion/left.jpg");
//        cv::Mat image1 = cv::imread("/home/lishuai/CLionProjects/imagefusion/ir.jpg");
//
//        // 计算平移偏移量
//        int tx = 15;  // 水平平移偏移量
//        int ty = -20;  // 垂直平移偏移量
//
//        // 创建平移矩阵
//        cv::Mat translation_matrix = (cv::Mat_<double>(2, 3) << 1, 0, tx, 0, 1, ty);
//
//        // 应用平移变换
//        cv::Mat translated_image;
//        cv::warpAffine(image1, translated_image, translation_matrix, image1.size());
//        Mat Registration2_2 = translated_image * 0.7 + image2 * 0.3;
//        // 保存平移后的图像
//        cv::imshow("fusion_image.jpg", Registration2_2 );
//        cv::waitKey(0);
//        return 0;
//
//}
