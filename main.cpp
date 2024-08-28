///////////////FUNCIONA

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>
#include <fstream>
#include <windows.h>  // Para la función Beep

using namespace cv;
using namespace std;

// Función para cargar las etiquetas
std::vector<std::string> loadLabels(const std::string& filename) {
    std::vector<std::string> labels;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
    return labels;
}

// Función para detectar objetos
void objectDetect(cv::dnn::Net& net, const cv::Mat& img, cv::Mat& output) {
    int dim = 300;
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0, cv::Size(dim, dim), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    output = net.forward();
}

// Función para mostrar texto
void drawText(cv::Mat& img, const std::string& text, int x, int y) {
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseline);
    cv::rectangle(img, cv::Point(x, y - textSize.height - baseline),
        cv::Point(x + textSize.width, y + baseline), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(img, text, cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1);
}

// Función para reproducir el pitido
void playBeep() {
    Beep(1000, 500); // Frecuencia 1000 Hz, duración 500 ms
}

// Función para dibujar objetos y activar alarma si la persona toca la zona prohibida
bool drawObjects(cv::Mat& img, const cv::Mat& detections, const std::vector<std::string>& labels, float threshold, const std::vector<cv::Point>& safeZonePolygon) {
    bool personDetected = false;
    for (int i = 0; i < detections.rows; i++) {
        float confidence = detections.at<float>(i, 2);
        if (confidence > threshold) {
            int classId = static_cast<int>(detections.at<float>(i, 1));
            int left = static_cast<int>(detections.at<float>(i, 3) * img.cols);
            int top = static_cast<int>(detections.at<float>(i, 4) * img.rows);
            int right = static_cast<int>(detections.at<float>(i, 5) * img.cols);
            int bottom = static_cast<int>(detections.at<float>(i, 6) * img.rows);

            if (classId < labels.size() && labels[classId] == "person") {
                personDetected = true;
                cv::Rect personRect(left, top, right - left, bottom - top);
                cv::rectangle(img, personRect, cv::Scalar(0, 255, 0), 2);
                std::string label = cv::format("Person %.2f%%", confidence * 100);
                drawText(img, label, left, top);

                // Verificar si la persona toca la zona prohibida
                if (!safeZonePolygon.empty()) {
                    bool inSafeZone = false;
                    for (const auto& point : safeZonePolygon) {
                        if (personRect.contains(point)) {
                            inSafeZone = true;
                            break;
                        }
                    }
                    if (inSafeZone) {
                        std::cout << "ALERTA: Persona tocando la zona prohibida con precisión de " << confidence * 100 << "%" << std::endl;
                        playBeep(); // Reproducir el pitido
                    }
                }
            }
        }
    }
    return personDetected;
}

int main() {
    // Cargar parámetros de calibración
    cv::FileStorage fs("C:/Users/ASUS TUF A15/Documents/Sexto_Semestre/Aplicaciones_basadas/camara_calibrada.xml", cv::FileStorage::READ);
    cv::Mat cameraMatrix, distCoeffs;
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    // Cargar modelo de detección de objetos
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow("C:/Users/ASUS TUF A15/Documents/Sexto_Semestre/Aplicaciones_basadas/Modelo_personas.pb",
        "C:/Users/ASUS TUF A15/Documents/Sexto_Semestre/Aplicaciones_basadas/datos_modelo.pbtxt");

    // Cargar etiquetas
    std::vector<std::string> labels = loadLabels("C:/Users/ASUS TUF A15/Documents/Sexto_Semestre/Aplicaciones_basadas/etiquetas.txt");

    // Inicializar el detector ArUco
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);

    cv::VideoCapture cap(1);
    if (!cap.isOpened()) {
        std::cout << "Error al abrir la cámara" << std::endl;
        return -1;
    }

    std::vector<cv::Point> lastValidZone;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Undistort frame
        cv::Mat undistortedFrame;
        cv::undistort(frame, undistortedFrame, cameraMatrix, distCoeffs);

        // Detect ArUco markers
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        detector.detectMarkers(undistortedFrame, corners, ids);

        if (!ids.empty()) {
            cv::aruco::drawDetectedMarkers(undistortedFrame, corners, ids);
        }

        std::vector<Point2f> safeZonePolygon;
        if (ids.size() == 4) {
            // Extraer las esquinas de los arucos
            for (const auto& corners : corners) {
                safeZonePolygon.push_back(corners[0]); // Esquina superior izquierda de cada aruco
            }
        }

        if (!ids.empty() && corners.size() >= 4) {
            // Calcular el centro manualmente
            cv::Point2f center(0, 0);
            for (const auto& corner : corners) {
                for (const auto& point : corner) {
                    center += point;
                }
            }
            center /= static_cast<float>(corners.size() * 4);

            if (ids.size() == 4) {
                safeZonePolygon.clear();
                for (const auto& corners : corners) {
                    safeZonePolygon.push_back(corners[0]); // Esquina superior izquierda de cada aruco
                }

                // Calcular el centro de los puntos
                cv::Point2f center(0, 0);
                for (const auto& pt : safeZonePolygon) {
                    center += pt;
                }
                center.x /= 4;
                center.y /= 4;

                // Convertir cv::Point2f a cv::Point
                std::vector<cv::Point> safeZonePolygonInt;
                for (const auto& pt : safeZonePolygon) {
                    safeZonePolygonInt.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
                }

                // Ordenar los puntos alrededor del centro en sentido antihorario
                std::sort(safeZonePolygonInt.begin(), safeZonePolygonInt.end(), [center](const Point& a, const Point& b) {
                    return atan2(a.y - center.y, a.x - center.x) < atan2(b.y - center.y, b.x - center.x);
                    });

                // Actualizar lastValidZone
                lastValidZone = safeZonePolygonInt;
            }
        }

        // Detección de objetos
        cv::Mat detections;
        objectDetect(net, undistortedFrame, detections);
        cv::Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());

        // Dibujar objetos detectados y comprobar si están en la zona prohibida
        bool personDetected = drawObjects(undistortedFrame, detectionMat, labels, 0.5, lastValidZone);

        // Cambiar el color del polígono según si se ha detectado una persona
        cv::Scalar polygonColor = personDetected ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0); // Verde si se detecta persona, rojo si no

        // Dibujar el polígono en la zona prohibida
        if (lastValidZone.size() == 4) {
            cv::polylines(undistortedFrame, lastValidZone, true, polygonColor, 2);
        }

        cv::imshow("Detección", undistortedFrame);
        if (cv::waitKey(1) == 27) break; // Presiona ESC para salir
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
