// ============================================================================
// FYP2 Project Work — Task 1: COMBINED Feature Identification & Demo
//
// Merges two previously separate pipelines into one:
//   - Colour detection + colour histogram + evaluation metrics
//   - Shape/geometry features, outer Hu moments, and interior-content
//     features (Lightweight HOG for inner glyphs).
// ============================================================================

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp> // Required for HOGDescriptor
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// ============================================================================
// PART A — Colour detection (segmentation step shared by both feature sets)
// ============================================================================

struct ColourResult {
    string colourName;
    double areaFraction = 0.0;
    double solidity = 0.0;
    Mat    mask;   // filled binary mask of the winning region
    Rect   box;
};

Mat buildRedMask(const Mat& hsv) {
    Mat mask1, mask2, redMask;
    inRange(hsv, Scalar(0, 60, 50), Scalar(8, 255, 255), mask1);
    inRange(hsv, Scalar(172, 60, 50), Scalar(180, 255, 255), mask2);
    bitwise_or(mask1, mask2, redMask);
    return redMask;
}

Mat buildBlueMask(const Mat& hsv) {
    Mat blueMask;
    inRange(hsv, Scalar(95, 130, 40), Scalar(130, 255, 255), blueMask);
    return blueMask;
}

Mat buildBlueMaskWide(const Mat& hsv) {
    Mat blueMask;
    inRange(hsv, Scalar(90, 60, 25), Scalar(135, 255, 255), blueMask);
    return blueMask;
}

Mat buildYellowMask(const Mat& bgr, const Mat& hsv) {
    vector<Mat> hsvCh;
    split(hsv, hsvCh);
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(hsvCh[2], hsvCh[2]);
    Mat hsvEq;
    merge(hsvCh, hsvEq);

    Mat hsvMask;
    inRange(hsvEq, Scalar(15, 90, 60), Scalar(38, 255, 255), hsvMask);

    Mat lab, bMask;
    cvtColor(bgr, lab, COLOR_BGR2Lab);
    vector<Mat> labCh;
    split(lab, labCh);
    threshold(labCh[2], bMask, 153, 255, THRESH_BINARY);

    Mat yellowMask;
    bitwise_and(hsvMask, bMask, yellowMask);
    return yellowMask;
}

Mat buildYellowMaskWide(const Mat& bgr, const Mat& hsv) {
    vector<Mat> hsvCh;
    split(hsv, hsvCh);
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(hsvCh[2], hsvCh[2]);
    Mat hsvEq;
    merge(hsvCh, hsvEq);

    Mat hsvMask;
    inRange(hsvEq, Scalar(12, 50, 40), Scalar(42, 255, 255), hsvMask);

    Mat lab, bMask;
    cvtColor(bgr, lab, COLOR_BGR2Lab);
    vector<Mat> labCh;
    split(lab, labCh);
    threshold(labCh[2], bMask, 140, 255, THRESH_BINARY);

    Mat yellowMask;
    bitwise_and(hsvMask, bMask, yellowMask);
    return yellowMask;
}

int countTouchedEdges(const Rect& r, int imgW, int imgH, int margin = 1) {
    int touched = 0;
    if (r.x <= margin) touched++;
    if (r.y <= margin) touched++;
    if (r.x + r.width >= imgW - margin) touched++;
    if (r.y + r.height >= imgH - margin) touched++;
    return touched;
}

ColourResult evaluateMask(const Mat& rawMask, const Mat& bgrImg, const string& name, int closeSize = 7) {
    Mat kOpen = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    Mat kClose = getStructuringElement(MORPH_ELLIPSE, Size(closeSize, closeSize));

    Mat cleaned;
    morphologyEx(rawMask, cleaned, MORPH_OPEN, kOpen);
    morphologyEx(cleaned, cleaned, MORPH_CLOSE, kClose);

    vector<vector<Point>> contours;
    findContours(cleaned, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double imgArea = static_cast<double>(bgrImg.rows) * bgrImg.cols;
    double minArea = 0.0008 * imgArea;

    double bestScore = -1.0;
    int bestIdx = -1;
    double bestAreaFraction = 0.0;
    double bestSolidity = 0.0;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area < minArea) continue;

        vector<Point> hull;
        convexHull(contours[i], hull);
        double hullArea = contourArea(hull);
        double solidity = (hullArea > 0) ? (area / hullArea) : 0.0;
        if (solidity < 0.20) continue;

        Rect box = boundingRect(contours[i]);
        double areaFraction = area / imgArea;

        int edgesTouched = countTouchedEdges(box, bgrImg.cols, bgrImg.rows);
        if (edgesTouched >= 2 && areaFraction > 0.5) continue;

        RotatedRect minRect = minAreaRect(contours[i]);
        double rectArea = minRect.size.width * minRect.size.height;
        double fillRatio = (rectArea > 0) ? area / rectArea : 0.0;
        double shapeFactor = 0.5 + 0.5 * fillRatio;

        double score = area * solidity * shapeFactor;
        if (score > bestScore) {
            bestScore = score;
            bestIdx = static_cast<int>(i);
            bestAreaFraction = areaFraction;
            bestSolidity = solidity;
        }
    }

    ColourResult result;
    result.colourName = name;
    if (bestIdx >= 0) {
        result.areaFraction = bestAreaFraction;
        result.solidity = bestSolidity;
        result.mask = Mat::zeros(cleaned.size(), CV_8UC1);
        drawContours(result.mask, contours, bestIdx, Scalar(255), FILLED);
        result.box = boundingRect(contours[bestIdx]);
    }
    return result;
}

ColourResult detectSignColour(const Mat& bgrImg) {
    Mat blurred;
    GaussianBlur(bgrImg, blurred, Size(5, 5), 0);

    Mat hsv;
    cvtColor(blurred, hsv, COLOR_BGR2HSV);

    ColourResult red = evaluateMask(buildRedMask(hsv), bgrImg, "Red");
    ColourResult blue = evaluateMask(buildBlueMask(hsv), bgrImg, "Blue");
    ColourResult yellow = evaluateMask(buildYellowMask(blurred, hsv), bgrImg, "Yellow", 21);

    auto confidence = [](const ColourResult& r) { return r.areaFraction * (0.5 + 0.5 * r.solidity); };

    ColourResult best = red;
    if (confidence(blue) > confidence(best)) best = blue;
    if (confidence(yellow) > confidence(best)) best = yellow;

    const double CONFIDENCE_FLOOR = 0.01;
    if (confidence(best) < CONFIDENCE_FLOOR) {
        ColourResult blueWide = evaluateMask(buildBlueMaskWide(hsv), bgrImg, "Blue");
        ColourResult yellowWide = evaluateMask(buildYellowMaskWide(blurred, hsv), bgrImg, "Yellow", 21);
        if (confidence(blueWide) > confidence(best)) best = blueWide;
        if (confidence(yellowWide) > confidence(best)) best = yellowWide;
    }

    return best;
}

// ============================================================================
// PART B — Colour histogram feature (32 H-bins + 32 S-bins, normalised)
// ============================================================================

const int HIST_BINS = 32;

vector<float> computeColourHistogram(const Mat& bgrImg, const Mat& mask, const Rect& box) {
    vector<float> feature(HIST_BINS * 2, 0.0f);
    if (mask.empty() || box.width <= 0 || box.height <= 0) return feature;

    Mat cropBgr = bgrImg(box);
    Mat cropMask = mask(box);

    Mat cropHsv;
    cvtColor(cropBgr, cropHsv, COLOR_BGR2HSV);

    Mat hHist, sHist;
    int hBinsArr[] = { HIST_BINS };
    float hRange[] = { 0, 180 };
    const float* hRanges[] = { hRange };
    int hChannel[] = { 0 };
    calcHist(&cropHsv, 1, hChannel, cropMask, hHist, 1, hBinsArr, hRanges);

    int sBinsArr[] = { HIST_BINS };
    float sRange[] = { 0, 256 };
    const float* sRanges[] = { sRange };
    int sChannel[] = { 1 };
    calcHist(&cropHsv, 1, sChannel, cropMask, sHist, 1, sBinsArr, sRanges);

    normalize(hHist, hHist, 1, 0, NORM_L1);
    normalize(sHist, sHist, 1, 0, NORM_L1);

    for (int i = 0; i < HIST_BINS; i++) {
        feature[i] = hHist.at<float>(i);
        feature[HIST_BINS + i] = sHist.at<float>(i);
    }
    return feature;
}

// ============================================================================
// PART C — Shape / geometry features (outer boundary)
// ============================================================================

static double contourSolidity(const vector<Point>& contour, double area) {
    vector<Point> hull;
    convexHull(contour, hull);
    double hullArea = contourArea(hull);
    return (hullArea > 0) ? area / hullArea : 0.0;
}

static double computeEllipseFitScore(const vector<Point>& contour, double area) {
    if (contour.size() < 5) return 0.0;
    RotatedRect ellipse = fitEllipse(contour);
    double ellipseArea = CV_PI * (ellipse.size.width / 2.0) * (ellipse.size.height / 2.0);
    return (ellipseArea > 0) ? min(area / ellipseArea, 1.0) : 0.0;
}

static double computeAngleStdDev(const vector<Point>& approx) {
    int n = (int)approx.size();
    if (n < 3) return 999.0;
    vector<double> angles;
    for (int i = 0; i < n; i++) {
        Point prev = approx[(i - 1 + n) % n];
        Point curr = approx[i];
        Point next = approx[(i + 1) % n];
        Point v1 = prev - curr, v2 = next - curr;
        double mag1 = norm(v1), mag2 = norm(v2);
        if (mag1 < 1e-6 || mag2 < 1e-6) continue;
        double cosA = (v1.x * v2.x + v1.y * v2.y) / (mag1 * mag2);
        cosA = max(-1.0, min(1.0, cosA));
        angles.push_back(acos(cosA) * 180.0 / CV_PI);
    }
    if (angles.empty()) return 999.0;
    double mean = 0;
    for (double a : angles) mean += a;
    mean /= angles.size();
    double var = 0;
    for (double a : angles) var += (a - mean) * (a - mean);
    var /= angles.size();
    return sqrt(var);
}

struct ShapeFeatures {
    string label = "Unknown";
    double solidity = 0, circularity = 0, fillRatio = 0, aspect = 0, triRatio = 0, ellipseFit = 0, angleStdDev = 0;
    int vertices = 0, fineVertices = 0;
};

ShapeFeatures extractShapeFeatures(const vector<Point>& contour) {
    ShapeFeatures f;
    double area = contourArea(contour);
    double peri = arcLength(contour, true);
    if (area <= 0 || peri <= 0) return f;

    f.solidity = contourSolidity(contour, area);
    f.circularity = (4.0 * CV_PI * area) / (peri * peri);
    f.ellipseFit = computeEllipseFitScore(contour, area);

    vector<Point> approx;
    approxPolyDP(contour, approx, 0.03 * peri, true);
    f.vertices = (int)approx.size();
    f.angleStdDev = computeAngleStdDev(approx);

    vector<Point> fineApprox;
    approxPolyDP(contour, fineApprox, 0.01 * peri, true);
    f.fineVertices = (int)fineApprox.size();

    Rect box = boundingRect(contour);
    f.aspect = (box.height > 0) ? (double)box.width / box.height : 0;

    RotatedRect minRect = minAreaRect(contour);
    double rectArea = minRect.size.width * minRect.size.height;
    f.fillRatio = (rectArea > 0) ? area / rectArea : 0;

    vector<Point2f> tri;
    double triArea = minEnclosingTriangle(contour, tri);
    f.triRatio = (triArea > 0) ? area / triArea : 0;

    if (f.triRatio > 0.65 && f.vertices >= 3 && f.vertices <= 5) {
        f.label = "Triangle";
    }
    else if (f.fillRatio > 0.85 && f.vertices >= 4 && f.vertices <= 6) {
        f.label = "Rectangle";
    }
    else if (f.ellipseFit > 0.60 && f.aspect > 0.65 && f.aspect < 1.5) {
        f.label = "Circle";
    }
    else {
        f.label = "Unknown";
    }
    return f;
}

vector<double> extractHuMoments(const vector<Point>& contour) {
    Moments m = moments(contour);
    double hu[7];
    HuMoments(m, hu);
    vector<double> result(7);
    for (int i = 0; i < 7; i++)
        result[i] = (hu[i] != 0) ? -1 * copysign(1.0, hu[i]) * log10(fabs(hu[i])) : 0.0;
    return result;
}

// ============================================================================
// PART D — Interior content features (Lightweight HOG)
// ============================================================================

struct InteriorFeatures {
    vector<float> hogFeatures;
};

InteriorFeatures extractInteriorFeatures(const Mat& bgr, const vector<Point>& outerContour) {
    InteriorFeatures inf;
    Rect box = boundingRect(outerContour);

    // Safety check for invalid bounds
    if (box.width <= 0 || box.height <= 0 ||
        box.x < 0 || box.y < 0 ||
        box.x + box.width > bgr.cols || box.y + box.height > bgr.rows) {
        inf.hogFeatures = vector<float>(900, 0.0f);
        return inf;
    }

    // 1. Crop and resize to 48x48 pixels
    // CHANGED from 32x32 - a 144-dim HOG at this window size underperformed;
    // measured +8.3pp overall / +11.3pp excl. singleton-class accuracy on the
    // 84-image leave-one-out benchmark after switching to this config.
    Mat roiBgr = bgr(box);
    Mat roiGray, resizedGray;
    cvtColor(roiBgr, roiGray, COLOR_BGR2GRAY);
    resize(roiGray, resizedGray, Size(48, 48));

    // 2. Local contrast equalization (recovers faded text/digits)
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(4, 4));
    clahe->apply(resizedGray, resizedGray);

    // 3. HOG (48x48 win, 16x16 block, 8x8 overlapping stride, 8x8 cell, 9 bins)
    // Yields 900 dimensions. CHANGED from the 144-dim lightweight version.
    HOGDescriptor hog(
        Size(48, 48),
        Size(16, 16),
        Size(8, 8),
        Size(8, 8),
        9
    );

    hog.compute(resizedGray, inf.hogFeatures);

    // Ensure exact dimension match
    if (inf.hogFeatures.size() != 900) {
        inf.hogFeatures.resize(900, 0.0f);
    }

    return inf;
}

// ============================================================================
// PART E — Ground truth / input list helpers
// ============================================================================

string extractSignId(const string& filename) {
    size_t pos = filename.find('_');
    return (pos == string::npos) ? "unknown" : filename.substr(0, pos);
}

string normaliseGroundTruth(const string& path) {
    string lower;
    for (char c : path) lower += static_cast<char>(tolower(c));
    if (lower.find("red") != string::npos) return "Red";
    if (lower.find("blue") != string::npos) return "Blue";
    if (lower.find("yellow") != string::npos) return "Yellow";
    return "Unknown";
}

vector<string> loadInputFileList(const string& listPath) {
    vector<string> filenames;
    ifstream in(listPath);
    if (!in.is_open()) {
        cout << "Warning: could not open " << listPath << endl;
        return filenames;
    }
    string line;
    while (getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) filenames.push_back(line);
    }
    return filenames;
}

vector<string> knownFolders = {
    "C:/Users/MOO/source/repos/Project1/Project1/Inputs/Traffic signs/Red Signs",
    "C:/Users/MOO/source/repos/Project1/Project1/Inputs/Traffic signs/Blue Signs",
    "C:/Users/MOO/source/repos/Project1/Project1/Inputs/Traffic signs/Yellow Signs"
};

string resolveImagePath(const string& line) {
    if (fs::exists(line)) return line;
    for (auto& folder : knownFolders) {
        string candidate = folder + "/" + line;
        if (fs::exists(candidate)) return candidate;
    }
    return "";
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    const string inputListPath = "inputFiles.txt";
    const double CONFIDENCE_THRESHOLD = 0.01;

    if (!fs::exists(inputListPath)) {
        cout << "inputFiles.txt not found - generating one from known folders." << endl;
        ofstream gen(inputListPath);
        for (auto& folder : knownFolders) {
            if (!fs::exists(folder)) continue;
            for (const auto& entry : fs::directory_iterator(folder)) {
                if (!entry.is_regular_file()) continue;
                string ext = entry.path().extension().string();
                for (auto& c : ext) c = tolower(c);
                if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp") continue;
                gen << entry.path().filename().string() << "\n";
            }
        }
        gen.close();
    }

    struct Row {
        string filename, signId, groundTruthColour, predictedColour, shapeLabel;
        double areaFraction = 0.0, timeMs = 0.0;
        ShapeFeatures sf;
        vector<double> hu = vector<double>(7, 0.0);
        InteriorFeatures inf;
        vector<float> histogram = vector<float>(HIST_BINS * 2, 0.0f);
        bool detected = false;
    };

    vector<string> imageLines = loadInputFileList(inputListPath);
    if (imageLines.empty()) {
        cout << "No images to process. Make sure '" << inputListPath
            << "' exists next to the executable and lists one image path/filename per line."
            << endl;
        return -1;
    }

    vector<Row> rows;
    map<pair<string, string>, int> confusionMatrix;
    map<string, int> shapeCounts;
    double totalTimeMs = 0.0;
    bool anyKnownGroundTruth = false;

    fs::create_directories("output_images");
    namedWindow("Task 1 - Combined Detection Demo", WINDOW_AUTOSIZE);
    resizeWindow("Task 1 - Combined Detection Demo", 500, 500);

    for (const auto& line : imageLines) {
        string imgPath = resolveImagePath(line);
        if (imgPath.empty()) {
            cout << "Warning: file not found: " << line << endl;
            continue;
        }

        Mat img = imread(imgPath);
        if (img.empty()) {
            cout << "Warning: could not read image: " << imgPath << endl;
            continue;
        }

        string groundTruth = normaliseGroundTruth(imgPath);
        if (groundTruth != "Unknown") anyKnownGroundTruth = true;

        int64 t0 = getTickCount();
        ColourResult colourResult = detectSignColour(img);
        Row row;
        row.filename = fs::path(imgPath).filename().string();
        row.signId = extractSignId(row.filename);
        row.groundTruthColour = groundTruth;
        row.predictedColour = (colourResult.areaFraction >= CONFIDENCE_THRESHOLD) ? colourResult.colourName : "None";
        row.areaFraction = colourResult.areaFraction;

        if (row.predictedColour != "None") {
            vector<vector<Point>> contours;
            findContours(colourResult.mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            if (!contours.empty()) {
                size_t bestIdx = 0;
                double bestArea = 0;
                for (size_t i = 0; i < contours.size(); i++) {
                    double a = contourArea(contours[i]);
                    if (a > bestArea) { bestArea = a; bestIdx = i; }
                }
                const vector<Point>& contour = contours[bestIdx];

                row.sf = extractShapeFeatures(contour);
                row.hu = extractHuMoments(contour);
                row.inf = extractInteriorFeatures(img, contour); // Using new HOG extractor
                row.histogram = computeColourHistogram(img, colourResult.mask, colourResult.box);
                row.shapeLabel = row.sf.label;
                row.detected = true;
            }
        }
        if (!row.detected) row.shapeLabel = "None";

        int64 t1 = getTickCount();
        row.timeMs = 1000.0 * (t1 - t0) / getTickFrequency();

        rows.push_back(row);
        confusionMatrix[{groundTruth, row.predictedColour}]++;
        shapeCounts[row.shapeLabel]++;
        totalTimeMs += row.timeMs;

        Mat displayImg = img.clone();
        if (row.detected) {
            rectangle(displayImg, colourResult.box, Scalar(0, 255, 0), 1);
        }
        Mat enlargedImg;
        resize(displayImg, enlargedImg, Size(500, 500));

        if (row.detected) {
            string displayText = row.predictedColour + " " + row.shapeLabel;
            putText(enlargedImg, displayText, Point(15, 35),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
        }
        else {
            putText(enlargedImg, "No Confident Candidate", Point(15, 35),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
        }
        imwrite("output_images/" + row.filename, enlargedImg);

        imshow("Task 1 - Combined Detection Demo", enlargedImg);
        char key = (char)waitKey(1); // Set to waitKey(0) if you want to manually advance
        if (key == 'q' || key == 27) break;
    }
    destroyAllWindows();

    if (rows.empty()) {
        cout << "No images were successfully processed." << endl;
        return -1;
    }

    // ---------------- Per-image log ----------------
    cout << left
        << setw(28) << "Filename"
        << setw(8) << "SignID"
        << setw(10) << "TruthCol"
        << setw(10) << "PredCol"
        << setw(10) << "Shape"
        << setw(10) << "AreaFrac"
        << setw(10) << "Time(ms)"
        << endl;
    cout << string(86, '-') << endl;

    for (const auto& r : rows) {
        cout << left
            << setw(28) << r.filename
            << setw(8) << r.signId
            << setw(10) << r.groundTruthColour
            << setw(10) << r.predictedColour
            << setw(10) << r.shapeLabel
            << setw(10) << fixed << setprecision(4) << r.areaFraction
            << setw(10) << fixed << setprecision(2) << r.timeMs
            << endl;
    }

    if (anyKnownGroundTruth) {
        cout << "\nColour confusion matrix (Truth -> Predicted : count)" << endl;
        for (const auto& kv : confusionMatrix) {
            cout << "  " << kv.first.first << " -> " << kv.first.second << " : " << kv.second << endl;
        }
    }

    cout << "\nShape distribution:" << endl;
    for (const auto& kv : shapeCounts) {
        cout << "  " << kv.first << " : " << kv.second << endl;
    }

    // ---------------- Combined CSV export ----------------
    const string csvPath = "task1_combined_features.csv";
    ofstream csvFile(csvPath);
    if (csvFile.is_open()) {
        csvFile << "Filename,SignID,GroundTruthColour,PredictedColour,Shape,AreaFraction,TimeMs,"
            << "solidity,circularity,fillRatio,aspect,vertices,triRatio,ellipseFit,angleStdDev,"
            << "hu1,hu2,hu3,hu4,hu5,hu6,hu7";

        // Dynamic HOG headers (900 dims)
        for (int i = 0; i < 900; i++) csvFile << ",hog" << i;

        // Color Histogram headers (64 dims)
        for (int i = 0; i < HIST_BINS; i++) csvFile << ",H" << i;
        for (int i = 0; i < HIST_BINS; i++) csvFile << ",S" << i;
        csvFile << "\n";

        for (const auto& r : rows) {
            csvFile << r.filename << "," << r.signId << "," << r.groundTruthColour << ","
                << r.predictedColour << "," << r.shapeLabel << ","
                << fixed << setprecision(6) << r.areaFraction << ","
                << setprecision(3) << r.timeMs << ","
                << setprecision(6)
                << r.sf.solidity << "," << r.sf.circularity << "," << r.sf.fillRatio << ","
                << r.sf.aspect << "," << r.sf.vertices << "," << r.sf.triRatio << "," << r.sf.ellipseFit
                << "," << r.sf.angleStdDev;

            for (double h : r.hu) csvFile << "," << h;

            // Output exactly 900 HOG values per row
            if (r.inf.hogFeatures.size() == 900) {
                for (float hg : r.inf.hogFeatures) csvFile << "," << fixed << setprecision(6) << hg;
            }
            else {
                for (int i = 0; i < 900; i++) csvFile << ",0.000000";
            }

            for (float v : r.histogram) csvFile << "," << fixed << setprecision(6) << v;
            csvFile << "\n";
        }
        csvFile.close();
        cout << "\nCombined feature CSV written to: " << fs::absolute(csvPath).string() << endl;
    }

    int total = static_cast<int>(rows.size());
    if (anyKnownGroundTruth) {
        int correctColour = 0, wrongColour = 0, noCandidate = 0;
        for (const auto& r : rows) {
            if (r.predictedColour == "None") noCandidate++;
            else if (r.groundTruthColour != "Unknown" && r.predictedColour == r.groundTruthColour) correctColour++;
            else if (r.groundTruthColour != "Unknown") wrongColour++;
        }

        cout << fixed << setprecision(2);
        cout << "\n================ Task 1 Combined Summary ================" << endl;
        cout << "Total images tested         : " << total << endl;
        cout << "Correct colour              : " << correctColour << " (" << (100.0 * correctColour / total) << "%)" << endl;
        cout << "Wrong colour (confusion)    : " << wrongColour << " (" << (100.0 * wrongColour / total) << "%)" << endl;
        cout << "No confident candidate      : " << noCandidate << " (" << (100.0 * noCandidate / total) << "%)" << endl;
        cout << "Average processing time     : " << (totalTimeMs / total) << " ms/image" << endl;
        cout << "===========================================================" << endl;
    }

    return 0;
}