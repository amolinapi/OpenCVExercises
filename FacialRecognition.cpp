// FacialRecognition.cpp : Este archivo contiene la función "main". La ejecución del programa comienza y termina ahí.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <thread>

using namespace cv;
using namespace std;
using namespace dnn;

void exercise2(Mat image) {
    Mat grayImage;

    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    imshow("Imagen de prueba", image);

    waitKey(0);

    imshow("Imagen de prueba", grayImage);

    imwrite("../data/images/test_gray.jpg", grayImage);
}

void exercise3(Mat image) {
    Mat blurImage, gaussianImage, medianImage, bilateralImage;

    blur(image, blurImage, Size(5,5));
    GaussianBlur(image, gaussianImage, Size(5,5), 0);
    medianBlur(image, medianImage, 5);
    bilateralFilter(image, bilateralImage, 9, 75, 75);

    imshow("Imagen Original", image);
    imshow("Imagen Blur", blurImage);
    imshow("Imagen Gaussian", gaussianImage);
    imshow("Imagen Median", medianImage);
    imshow("Imagen Bilateral", bilateralImage);
}

void exercise4() {
    Mat image = Mat::zeros(500, 500, CV_8UC3);

    line(image, Point(50,50), Point(450,50), Scalar(255,0,0), 2);
    rectangle(image, Point(100,100), Point(400,300), Scalar(0,255,0), 3);
    circle(image, Point(250,250), 50, Scalar(0,0,255), -1);

    putText(image, "OpenCV dibujos", Point(120,450), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,0), 2);

    imshow("Dibujos OpenCV", image);
    imwrite("../data/images/test_dibujos.jpg", image);
}

void exercise5() {
    VideoCapture cap(0);

    if (!cap.isOpened()) {
        cout << "ERROR: No se pudo abrir la camara" << endl;
        return;
    }

    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    VideoWriter video("../data/videos/captura.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(frame_width, frame_height));

    if (!video.isOpened()) {
        cout << "ERROR: El video no se pudo abrir" << endl;
        return;
    }

    Mat frame;
    while (true) {
        cap >> frame;

        if (frame.empty()) {
            cout << "ERROR: No se pudo capturar el fotograma" << endl;
            break;
        }

        imshow("Webcam", frame);

        video.write(frame);

        if (waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    video.release();
    cv::destroyAllWindows();
}

void exercise6() {
    Mat image = imread("../data/images/test_gray.jpg");

    Mat blurredImage;
    GaussianBlur(image, blurredImage, Size(5, 5), 1.5);

    Mat edges;
    Canny(blurredImage, edges, 50, 150);

    imshow("Imagen Original", image);
    imshow("Bordes", edges);

    imwrite("../data/images/test_bordes.jpg", edges);
}

void exercise7() {
    Mat image = imread("../data/images/test_gray.jpg");

    Mat blurredImage;
    GaussianBlur(image, blurredImage, Size(5, 5), 1.5);

    Mat edges;
    Canny(blurredImage, edges, 50, 150);
    vector<vector<Point>> contours;
    vector<Vec4i> hierachy;

    // El material con la imagen tiene que venir de canny, ya que tiene que ser de tipo CV_8UC1
    findContours(edges, contours, hierachy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat contourImage = Mat::zeros(image.size(), CV_8UC3);

    drawContours(contourImage, contours, -1, Scalar(0,255,0), 2);
    imshow("Imagen Original", image);
    imshow("Contornos detectados", contourImage);

    imwrite("../data/images/test_contornos.jpg", contourImage);
}

void exercise8(Mat image) {
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    Scalar lowerBound(20, 100, 100);
    Scalar upperBound(30, 255, 255);

    Mat mask;
    inRange(hsvImage, lowerBound, upperBound, mask);

    Mat result;
    bitwise_and(image, image, result, mask);

    imshow("Imagen Original", image);
    imshow("Mascara", mask);
    imshow("Segmentacion de color", result);

    imwrite("../data/images/test_segmentacion.jpg", result);
}

void exercise9(Mat image) {
    int width = image.cols;
    int height = image.rows;

    Point2f center(width / 2.0f, height / 2.0f);

    Mat rotated90, rotated180, rotated270;

    Mat m90 = getRotationMatrix2D(center, -90, 1.0);
    warpAffine(image, rotated90, m90, Size(height, width));

    Mat m180 = getRotationMatrix2D(center, -90, 1.0);
    warpAffine(image, rotated180, m180, Size(width, height));

    Mat m270 = getRotationMatrix2D(center, -90, 1.0);
    warpAffine(image, rotated270, m270, Size(height, width));

    Mat scaledUp, scaledDown;

    resize(image, scaledUp, Size(), 2.0,2.0,INTER_LINEAR);
    resize(image, scaledDown, Size(), 0.5, 0.5, INTER_LINEAR);

    imshow("Image Original", image);
    imshow("Image rotada 90", rotated90);
    imshow("Image rotada 180", rotated180);
    imshow("Image rotada 270", rotated270);
    imshow("Image escalada x2", scaledUp);
    imshow("Image escalada x0.5", scaledDown);

    imwrite("../data/images/test_rotada90.jpg", rotated90);
    imwrite("../data/images/test_rotada180.jpg", rotated180);
    imwrite("../data/images/test_rotada270.jpg", rotated270);
    imwrite("../data/images/test_escaladaup.jpg", scaledUp);
    imwrite("../data/images/test_escaladadown.jpg", scaledDown);
}

void exercise10() {
    Mat image = imread("../data/images/test2.jpg");  // Cargar una imagen

    if (image.empty()) {
        cout << "No se pudo cargar la imagen." << endl;
        return;
    }

    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat eroded, dilated, opened, closed;
    erode(image,eroded, kernel);
    dilate(image, dilated, kernel);
    morphologyEx(image,opened, MORPH_OPEN, kernel);
    morphologyEx(image,closed, MORPH_CLOSE, kernel);

    imshow("Imagen Original", image);
    imshow("Erosion", eroded);
    imshow("Dilatacion", dilated);
    imshow("Apertura (MORPH_OPEN)", opened);
    imshow("Cierre (MORPH_OPEN)", closed);

    imwrite("../data/images/test2_erode.jpg", eroded);
    imwrite("../data/images/test2_dilatada.jpg", dilated);
    imwrite("../data/images/test2_opened.jpg", opened);
    imwrite("../data/images/test2_closed.jpg", closed);
}

void exercise11() {
    CascadeClassifier face_cascade;
    if (!face_cascade.load("../data/files/haarcascade_frontalface_default.xml")) {
        cout << "Error al cargar el archivo" << endl;
        return;
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error al cargar la camara" << endl;
        return;
    }

    Mat frame, gray;
    vector<Rect> faces;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        face_cascade.detectMultiScale(gray, faces, 1.3, 5, 0, Size(30, 30));

        if (!faces.empty()) {
            for (size_t i = 0; i < faces.size(); i++) {
                rectangle(frame, faces[i], Scalar(0, 255, 0), 2);
            }
        }

        imshow("Deteccion de caras", frame);

        if (waitKey(30) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
}

void exercise12() {
    VideoCapture cap(0);

    if (!cap.isOpened()) {
        cout << "ERROR: La camara no se pudo abrir";
        return;
    }

    Mat frame, hsv, mask;

    Scalar lower_bounds(35,100,100);
    Scalar upper_bounds(85,255,255);

    while (true) {
        cap >> frame;

        if (frame.empty()) break;

        cvtColor(frame, hsv, COLOR_BGR2HSV);

        inRange(hsv, lower_bounds, upper_bounds, mask);

        Moments m = moments(mask, true);

        if (m.m00 > 0) {
            int cx = int(m.m10 / m.m00);
            int cy = int(m.m01 / m.m00);
            circle(frame, Point(cx, cy), 10, Scalar(0,0,255), -1);
        }

        imshow("Video original", frame);
        imshow("Masrcara", mask);

        if (waitKey(30) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
}

void exercise13() {
    Mat img = imread("../data/images/texto.jpg");
    if (img.empty()) {
        cout << "No se pudo cargar la imagen." << endl;
        return;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat thresh;
    threshold(gray, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);

    imwrite("../data/images/temp.png", thresh);

    tesseract::TessBaseAPI ocr;
    if (ocr.Init(NULL, "eng")) {
        cout << "ERROR AL INICIAR TESSERACT" << endl;
        return;
    }

    ocr.SetImage(thresh.data, thresh.cols, thresh.rows, 1, thresh.step);
    string text = string(ocr.GetUTF8Text());

    cout << "Texto detectado: " << text << endl;

    imshow("Imagen original", img);
    imshow("Imagen procesada", thresh);
}

void exercise14() {
    VideoCapture cap(0);

    if (!cap.isOpened()) {
        cout << "ERROR AL ABRIR LA CAMARA" << endl;
        return;
    }

    Mat frame1, frame2, diff, gray, thresh;

    cap >> frame1;
    if (frame1.empty()) {
        cout << "ERROR: No se captura el primer frame" << endl;
        return;
    }

    cvtColor(frame1, frame1, COLOR_BGR2GRAY);

    while (true) {
        cap >> frame2;

        if (frame2.empty()) break;

        cvtColor(frame2, frame2, COLOR_BGR2GRAY);

        absdiff(frame1, frame2, diff);

        threshold(diff, thresh, 25, 255, THRESH_BINARY);

        imshow("Diferencia de frames", thresh);

        frame1 = frame2.clone();

        if (waitKey(30) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
}

vector<string> getOutputsNames(const Net& net) {
    vector<int> outLayers = net.getUnconnectedOutLayers();
    vector<string> layersNames = net.getLayerNames();
    vector<string> names;
    for (int i : outLayers) {
        names.push_back(layersNames[i - 1]);
    }
    return names;
}

void exercise15() {
    string modelWeights = "../data/files/yolov4-tiny.weights";
    string modelConfig = "../data/files/yolov4-tiny.cfg";
    string classFile = "../data/files/coco.names";

    Net net = readNet(modelWeights, modelConfig, "Darknet");

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CUDA);

    vector<string> classes;
    ifstream ifs(classFile);
    string line;
    while (getline(ifs, line)) {
        classes.push_back(line);
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "ERROR AL ABRIR LA CAMARA" << endl;
        return;
    }

    Mat frame, blob;
    vector<Mat> outs;
    bool ready = false;

    thread yoloThread([&]() {
        while (true) {
            if (!ready) continue;

            net.setInput(blob);
            net.forward(outs, getOutputsNames(net));
            ready = false;
        }
    });

    while (cap.isOpened()) {
        cap >> frame;
        if (frame.empty()) break;

        blobFromImage(frame, blob, 1 / 255.0, Size(320, 320), Scalar(), true, false);
        ready = true;

        if (!outs.empty()) {
            vector<int> classIds;
            vector<float> confidences;
            vector<Rect> boxes;

            for (auto& out : outs) {
                float* data = (float*)out.data;
                for (int i = 0; i < out.rows; i++, data += out.cols) {
                    Mat scores = out.row(i).colRange(5, out.cols);
                    Point classIdPoint;
                    double confidence;
                    cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                    if (confidence > 0.6) {
                        int centerX = (int)(data[0] * frame.cols);
                        int centerY = (int)(data[1] * frame.rows);
                        int width = (int)(data[2] * frame.cols);
                        int height = (int)(data[3] * frame.rows);
                        int left = centerX - width / 2;
                        int top = centerY - height / 2;

                        classIds.push_back(classIdPoint.x);
                        confidences.push_back((float)confidence);
                        boxes.push_back(Rect(left, top, width, height));
                    }
                }
            }

            vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

            for (int i : indices) {
                rectangle(frame, boxes[i], Scalar(0, 255, 0), 3);
                string label = format("%.2f", confidences[i]) + " " + classes[classIds[i]];
                putText(frame, label, boxes[i].tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
            }
        }        

        imshow("Deteccion de objetos", frame);

        if (waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    yoloThread.join();
}

int main()
{
    Mat image = imread("../data/images/test.jpg");  // Cargar una imagen

    if (image.empty()) {
        cout << "No se pudo cargar la imagen." << endl;
        return -1;
    }

    // EXERCISE 2
    //exercise2(image);

    // EXERCISE 3
    //exercise3(image);

    // EXERCISE 4
    //exercise4();

    //EXERCISE 5
    //exercise5();

    //EXERCISE 6
    //exercise6();

    //EXERCISE 7
    //exercise7();

    //EXERCISE 8
    //exercise8(image);

    //EXERCISE 9
    //exercise9(image);

    //EXERCISE 10
    //exercise10();

    //EXERCISE 11
    //exercise11();

    //EXERCISE 12
    //exercise12();

    //EXERCISE 13
    //exercise13();

    //EXERCISE 14
    //exercise14();

    //EXERCISE 15
    exercise15();

    waitKey(0);
    return 0;
}

