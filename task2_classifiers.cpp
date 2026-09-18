// ============================================================================
// UCCC2513 Mini Project — Task 2: Traffic Sign Recognition
//
// Use the best feature from Task 1 with 3 machine learning classifiers:
//   1. KNN
//   2. ANN
//   3. Naive Bayes
//
// Evaluation:
//   - Accuracy
//   - Precision
//   - Recall
//   - F1-score
//   - Confusion matrix
//   - Visualization of results
//
// Selected Task 1 feature: HOG (144 dimensions)
// ============================================================================

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::ml;
namespace fs = std::filesystem;

// Shared with main()'s single-CSV evaluation and the external-training
// evaluation added below, so both always agree on the feature width.
static const int HOG_FEATURES = 144;

// NormalBayesClassifier fits a full covariance matrix per class, which is
// singular whenever a class has fewer training examples than feature
// dimensions - with up to 57 classes and some having as few as 2 external
// training images, that's true for most classes at 144-dim. A singular
// covariance makes that class's discriminant score blow up and win every
// single prediction regardless of input (observed: Naive Bayes predicted
// the same one class for all 84 test images). Reducing to this many PCA
// dimensions (fit on training data only) keeps the estimate stable for any
// class with >10 real examples; classes below that remain fundamentally
// hard for any classifier - same small-data limit as elsewhere in this
// project, not something dimensionality reduction can fix.
static const int NAIVE_BAYES_PCA_DIMS = 10;

// ============================================================================
// Helper 1: split one CSV line
// The Task 1 CSV does not contain commas inside the fields, so a simple parser
// is enough here.
// ============================================================================

vector<string> splitCSV(const string& line) {
    vector<string> cells;
    string cell;
    stringstream ss(line);

    while (getline(ss, cell, ','))
        cells.push_back(cell);

    // Keep the final empty field, if any
    if (!line.empty() && line.back() == ',')
        cells.push_back("");

    return cells;
}

// ============================================================================
// Helper 2: structure for one row from Task 1
// We keep Filename, the original SignID and the selected feature values.
// The real traffic-sign type used for classification is loaded separately
// (see loadSignTypeLabels() below) from sign_type_labels.csv, since a SignID
// is not automatically the correct class - different SignIDs can be the
// same real sign type, or need consolidating after manual review.
// ============================================================================

struct Sample {
    string filename;
    string signID;      // kept only as a reference to the Task 1 CSV row
    string signType;     // the real class used for Task 2 classification
    vector<float> hog;
};

// ============================================================================
// Helper 3: read Task 1 CSV
// ============================================================================

// Original 144-column selection (hog0..hog143), kept as the default so the
// existing single-CSV evaluation in main() is untouched.
vector<string> defaultHogColumns() {
    vector<string> cols;
    for (int h = 0; h < HOG_FEATURES; h++) cols.push_back("hog" + to_string(h));
    return cols;
}

// featureColumns lets a caller load an arbitrary named set of columns
// instead of the hardcoded hog0..hog143 - used by evaluateWithExternalTraining()
// below to load the exact same ANOVA-selected feature set the Python side
// uses, for a fair classifier-vs-classifier comparison (same information,
// different algorithm) rather than comparing different feature engineering.
vector<Sample> loadTask1CSV(const string& csvPath, const vector<string>& featureColumns = defaultHogColumns()) {
    vector<Sample> samples;
    ifstream in(csvPath);

    if (!in.is_open()) {
        cout << "Error: could not open " << csvPath << endl;
        return samples;
    }

    string headerLine;
    getline(in, headerLine);
    vector<string> header = splitCSV(headerLine);

    int filenameCol = -1;
    int signIDCol = -1;
    int numFeatures = (int)featureColumns.size();
    vector<int> featureCols(numFeatures, -1);

    for (int c = 0; c < (int)header.size(); c++) {
        if (header[c] == "Filename") filenameCol = c;
        if (header[c] == "SignID") signIDCol = c;

        for (int h = 0; h < numFeatures; h++) {
            if (header[c] == featureColumns[h])
                featureCols[h] = c;
        }
    }

    if (filenameCol < 0 || signIDCol < 0) {
        cout << "Error: Filename or SignID column is missing." << endl;
        return samples;
    }

    for (int h = 0; h < numFeatures; h++) {
        if (featureCols[h] < 0) {
            cout << "Error: " << featureColumns[h] << " column is missing." << endl;
            return samples;
        }
    }

    string line;
    while (getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        vector<string> cells = splitCSV(line);
        if ((int)cells.size() < (int)header.size()) continue;

        Sample s;
        s.filename = cells[filenameCol];
        s.signID = cells[signIDCol];
        s.hog.resize(numFeatures, 0.0f);

        bool valid = true;
        for (int h = 0; h < numFeatures; h++) {
            try {
                s.hog[h] = stof(cells[featureCols[h]]);
            }
            catch (...) {
                valid = false;
                break;
            }
        }

        if (valid)
            samples.push_back(s);
    }

    return samples;
}

// ============================================================================
// Helper 3B: load the REAL traffic-sign type for every image
//
// IMPORTANT:
// SignID is not automatically treated as the sign class. Different SignIDs may
// contain the same type of traffic sign. Therefore Task 2 should train and test
// using the actual sign type.
//
// The program expects:
//     sign_type_labels.csv
//
// Format:
//     Filename,SignID,SignType
//
// Example:
//     020_0003.png,020,Keep right
//
// On the Python side (task2_combined.py), this file is generated and kept
// complete automatically, pre-filled from its ID_LABEL dict for every image
// in both the project's own CSV and TSRD_external - run that first if this
// file doesn't exist yet or is missing rows for images this program needs.
// If it's still missing entries after that, edit the CSV by hand and rerun.
// ============================================================================

// quiet: when the caller is going to filter out unlabelled samples and
// continue anyway (e.g. evaluateWithExternalTraining(), which expects some
// TSRD_external images to fall outside our mapped classes), skip the
// per-image "Missing SignType" spam and the "before training" message,
// which are misleading there since training proceeds regardless.
bool loadSignTypeLabels(const string& labelPath,
    vector<Sample>& samples, bool quiet = false) {

    ifstream in(labelPath);

    // No file at all: create a template that can be filled in by hand (or
    // run task2_combined.py first, which fills it in automatically).
    if (!in.is_open()) {
        ofstream out(labelPath);

        if (!out.is_open()) {
            cout << "Error: could not create " << labelPath << endl;
            return false;
        }

        out << "Filename,SignID,SignType\n";

        for (const Sample& s : samples) {
            out << fs::path(s.filename).filename().string()
                << "," << s.signID << ",\n";
        }

        out.close();

        cout << "\nCreated: " << labelPath << endl;
        cout << "Run task2_combined.py first to fill this in automatically, "
             << "or open this CSV and fill in the SignType column by hand." << endl;
        cout << "Example:" << endl;
        cout << "020_0003.png,020,Keep right" << endl;
        cout << "\nAfter filling all sign types, save the CSV and run again." << endl;

        return false;
    }

    string headerLine;
    getline(in, headerLine);

    vector<string> header = splitCSV(headerLine);

    int filenameCol = -1;
    int signTypeCol = -1;

    for (int c = 0; c < (int)header.size(); c++) {
        if (header[c] == "Filename")
            filenameCol = c;

        if (header[c] == "SignType")
            signTypeCol = c;
    }

    if (filenameCol < 0 || signTypeCol < 0) {
        cout << "Error: " << labelPath
            << " must contain Filename and SignType columns." << endl;
        return false;
    }

    // Store both the exact filename and the basename so the mapping works
    // whether Task 1 stored a complete path or only a filename.
    map<string, string> filenameToType;

    string line;
    while (getline(in, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        if (line.empty())
            continue;

        vector<string> cells = splitCSV(line);

        if (filenameCol >= (int)cells.size() ||
            signTypeCol >= (int)cells.size())
            continue;

        string filename = cells[filenameCol];
        string signType = cells[signTypeCol];

        if (filename.empty() || signType.empty())
            continue;

        filenameToType[filename] = signType;
        filenameToType[fs::path(filename).filename().string()] = signType;
    }

    int missingLabels = 0;

    for (Sample& s : samples) {
        string exactName = s.filename;
        string baseName = fs::path(s.filename).filename().string();

        auto exactIt = filenameToType.find(exactName);
        auto baseIt = filenameToType.find(baseName);

        if (exactIt != filenameToType.end()) {
            s.signType = exactIt->second;
        }
        else if (baseIt != filenameToType.end()) {
            s.signType = baseIt->second;
        }
        else {
            s.signType.clear();
            missingLabels++;

            if (!quiet)
                cout << "Missing SignType for: "
                    << baseName << endl;
        }
    }

    if (missingLabels > 0) {
        if (!quiet) {
            cout << "\n" << missingLabels
                << " image(s) still have no SignType in "
                << labelPath << endl;
            cout << "Fill every missing SignType before training." << endl;
        }
        return false;
    }

    return true;
}

// ============================================================================
// Helper 4: normalization
// Same idea as the lecturer's examples:
// Xnew = (X - mean) / sigma
//
// IMPORTANT: mean and sigma are calculated from TRAINING DATA only.
// The same values are then used to normalize verification data.
// ============================================================================

void normalizeTrainingData(Mat& trainingData,
    Mat& verificationData,
    vector<double>& mean,
    vector<double>& sigma) {

    mean.clear();
    sigma.clear();

    for (int i = 0; i < trainingData.cols; i++) {
        Scalar meanOut, sigmaOut;
        meanStdDev(trainingData.col(i), meanOut, sigmaOut);

        double m = meanOut[0];
        double s = sigmaOut[0];

        // Avoid division by zero if one HOG dimension is constant
        if (s < 1e-9) s = 1.0;

        mean.push_back(m);
        sigma.push_back(s);
    }

    for (int i = 0; i < trainingData.cols; i++) {
        trainingData.col(i) = (trainingData.col(i) - mean[i]) / sigma[i];
        verificationData.col(i) = (verificationData.col(i) - mean[i]) / sigma[i];
    }
}

// ============================================================================
// Evaluation result
// ============================================================================

struct EvaluationResult {
    string modelName;
    double accuracy = 0.0;
    double precision = 0.0;  // macro precision
    double recall = 0.0;     // macro recall
    double f1 = 0.0;         // macro F1-score
    Mat confusion;
    vector<double> classPrecision;
    vector<double> classRecall;
    vector<double> classF1;
};

// ============================================================================
// Helper 5: calculate accuracy, precision, recall, F1 and confusion matrix
// ============================================================================

EvaluationResult evaluateModel(const string& modelName,
    const vector<int>& truth,
    const vector<int>& predicted,
    int numberOfClasses) {

    EvaluationResult result;
    result.modelName = modelName;
    result.confusion = Mat::zeros(numberOfClasses, numberOfClasses, CV_32S);

    int correct = 0;

    for (int i = 0; i < (int)truth.size(); i++) {
        int actual = truth[i];
        int pred = predicted[i];

        if (actual >= 0 && actual < numberOfClasses &&
            pred >= 0 && pred < numberOfClasses) {
            result.confusion.at<int>(actual, pred)++;
        }

        if (actual == pred) correct++;
    }

    result.accuracy = truth.empty() ? 0.0 : (double)correct / truth.size();

    result.classPrecision.resize(numberOfClasses, 0.0);
    result.classRecall.resize(numberOfClasses, 0.0);
    result.classF1.resize(numberOfClasses, 0.0);

    double precisionSum = 0.0;
    double recallSum = 0.0;
    double f1Sum = 0.0;

    for (int c = 0; c < numberOfClasses; c++) {
        int TP = result.confusion.at<int>(c, c);
        int FP = 0;
        int FN = 0;

        for (int r = 0; r < numberOfClasses; r++) {
            if (r != c) FP += result.confusion.at<int>(r, c);
        }

        for (int k = 0; k < numberOfClasses; k++) {
            if (k != c) FN += result.confusion.at<int>(c, k);
        }

        double p = (TP + FP > 0) ? (double)TP / (TP + FP) : 0.0;
        double r = (TP + FN > 0) ? (double)TP / (TP + FN) : 0.0;
        double f = (p + r > 0.0) ? 2.0 * p * r / (p + r) : 0.0;

        result.classPrecision[c] = p;
        result.classRecall[c] = r;
        result.classF1[c] = f;

        precisionSum += p;
        recallSum += r;
        f1Sum += f;
    }

    result.precision = precisionSum / numberOfClasses;
    result.recall = recallSum / numberOfClasses;
    result.f1 = f1Sum / numberOfClasses;

    return result;
}

// ============================================================================
// Helper 6: show one confusion matrix as an OpenCV image
// Rows = actual class, columns = predicted class
// ============================================================================

Mat drawConfusionMatrix(const EvaluationResult& result,
    const vector<string>& classNames) {

    int n = (int)classNames.size();

    int cell = 45;
    int leftMargin = 85;
    int topMargin = 95;

    // Class names are now full sign-type text (e.g. "Keep right"), much
    // longer than a three-digit SignID. Use short labels C0, C1, ... inside
    // the matrix and show the real traffic-sign type in a legend below.
    int legendColumns = 2;
    int legendRows = (n + legendColumns - 1) / legendColumns;
    int legendHeight = 35 + legendRows * 28;

    int width = max(900, leftMargin + n * cell + 40);
    int height = topMargin + n * cell + legendHeight;

    Mat image(height, width, CV_8UC3, Scalar(255, 255, 255));

    putText(image, result.modelName + " Confusion Matrix",
        Point(15, 28), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2);

    putText(image, "Predicted class",
        Point(leftMargin, 55), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);

    putText(image, "Actual",
        Point(10, topMargin - 15), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1);

    int maxValue = 1;
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
            maxValue = max(maxValue, result.confusion.at<int>(r, c));

    for (int c = 0; c < n; c++) {
        putText(image, "C" + to_string(c),
            Point(leftMargin + c * cell + 8, topMargin - 12),
            FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0, 0, 0), 1);
    }

    for (int r = 0; r < n; r++) {
        putText(image, "C" + to_string(r),
            Point(40, topMargin + r * cell + 28),
            FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0, 0, 0), 1);

        for (int c = 0; c < n; c++) {
            int value = result.confusion.at<int>(r, c);
            int intensity = (int)(220.0 * value / maxValue);

            // Stronger count = darker cell
            Scalar fillColour(255 - intensity, 255 - intensity, 255);
            Rect box(leftMargin + c * cell, topMargin + r * cell, cell, cell);
            rectangle(image, box, fillColour, FILLED);
            rectangle(image, box, Scalar(180, 180, 180), 1);

            Scalar textColour = (intensity > 130) ? Scalar(255, 255, 255) : Scalar(0, 0, 0);
            putText(image, to_string(value),
                Point(box.x + 14, box.y + 27),
                FONT_HERSHEY_SIMPLEX, 0.4, textColour, 1);
        }
    }

    int legendStartY = topMargin + n * cell + 32;

    putText(image, "Class legend:",
        Point(15, legendStartY),
        FONT_HERSHEY_SIMPLEX, 0.48, Scalar(0, 0, 0), 1);

    for (int i = 0; i < n; i++) {
        int column = i % legendColumns;
        int row = i / legendColumns;

        int x = 20 + column * (width / 2);
        int y = legendStartY + 28 + row * 28;

        string label = "C" + to_string(i) + " = " + classNames[i];

        putText(image, label,
            Point(x, y),
            FONT_HERSHEY_SIMPLEX, 0.38, Scalar(0, 0, 0), 1);
    }

    return image;
}

// ============================================================================
// Helper 7: visualization comparing the 3 classifiers
// ============================================================================

Mat drawModelComparison(const vector<EvaluationResult>& results,
    const string& subtitle = "") {
    const int width = 950;
    const int height = 600;
    const int left = 90;
    const int right = 35;
    const int top = 70;
    const int bottom = 100;

    Mat image(height, width, CV_8UC3, Scalar(255, 255, 255));

    putText(image, "Task 2 - Classifier Performance",
        Point(220, 35), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);

    if (!subtitle.empty()) {
        putText(image, subtitle,
            Point(210, 55), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(90, 90, 90), 1);
    }

    int plotW = width - left - right;
    int plotH = height - top - bottom;

    // Axis
    line(image, Point(left, top), Point(left, top + plotH), Scalar(0, 0, 0), 2);
    line(image, Point(left, top + plotH), Point(left + plotW, top + plotH), Scalar(0, 0, 0), 2);

    // 0.0 to 1.0 scale
    for (int i = 0; i <= 5; i++) {
        double value = i * 0.2;
        int y = top + plotH - (int)(value * plotH);

        line(image, Point(left - 5, y), Point(left + plotW, y), Scalar(220, 220, 220), 1);

        stringstream ss;
        ss << fixed << setprecision(1) << value;
        putText(image, ss.str(), Point(45, y + 5),
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 1);
    }

    vector<string> metricNames = { "Accuracy", "Precision", "Recall", "F1-score" };
    vector<Scalar> modelColours = {
        Scalar(220, 120, 40),
        Scalar(80, 170, 80),
        Scalar(60, 80, 220)
    };

    int groups = 4;
    int groupWidth = plotW / groups;
    int barWidth = 34;
    int gap = 8;

    for (int m = 0; m < groups; m++) {
        int groupX = left + m * groupWidth;

        for (int model = 0; model < (int)results.size(); model++) {
            double value = 0.0;
            if (m == 0) value = results[model].accuracy;
            if (m == 1) value = results[model].precision;
            if (m == 2) value = results[model].recall;
            if (m == 3) value = results[model].f1;

            int x = groupX + 35 + model * (barWidth + gap);
            int barHeight = (int)(value * plotH);
            int y = top + plotH - barHeight;

            rectangle(image, Rect(x, y, barWidth, barHeight),
                modelColours[model], FILLED);

            stringstream ss;
            ss << fixed << setprecision(2) << value;
            putText(image, ss.str(), Point(x - 2, y - 6),
                FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0, 0, 0), 1);
        }

        putText(image, metricNames[m],
            Point(groupX + 35, top + plotH + 35),
            FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1);
    }

    // Legend
    int legendY = height - 35;
    for (int i = 0; i < (int)results.size(); i++) {
        int x = 260 + i * 180;
        rectangle(image, Rect(x, legendY - 15, 20, 20), modelColours[i], FILLED);
        putText(image, results[i].modelName, Point(x + 28, legendY),
            FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1);
    }

    return image;
}

// ============================================================================
// Helper 7B: locate an image used by Task 1
//
// The CSV may store either a full path or only the filename. If only the
// filename is stored, the program searches recursively inside Input (this
// project's actual folder - contains "Red Signs", "Blue Signs", "Yellow
// Signs" subfolders) so the images may remain wherever Task 1 organised them.
// ============================================================================

string findImagePath(const string& filename,
    const string& searchRoot = "Input") {

    fs::path originalPath(filename);

    // First try the path exactly as written in the CSV.
    if (fs::exists(originalPath) && fs::is_regular_file(originalPath))
        return originalPath.string();

    // If that does not work, search using only the filename.
    string wantedName = originalPath.filename().string();

    fs::path root(searchRoot);
    if (!fs::exists(root))
        return "";

    try {
        for (const auto& entry :
            fs::recursive_directory_iterator(
                root,
                fs::directory_options::skip_permission_denied)) {

            if (!entry.is_regular_file())
                continue;

            if (entry.path().filename().string() == wantedName)
                return entry.path().string();
        }
    }
    catch (...) {
        return "";
    }

    return "";
}

// ============================================================================
// Helper 7C: create one image showing the classification result
//
// Green text = the classifier predicted the correct traffic-sign type.
// Red text   = the classifier predicted the wrong traffic-sign type.
// The correct sign type is also shown so the prediction can be compared directly.
// ============================================================================

Mat drawIndividualPrediction(const Sample& sample,
    const string& actualClass,
    const string& knnClass,
    const string& annClass,
    const string& bayesClass) {

    string imagePath = findImagePath(sample.filename);
    Mat sourceImage;

    if (!imagePath.empty())
        sourceImage = imread(imagePath);

    // If the original image cannot be found, still create a useful result
    // window showing the filename and classifier predictions.
    if (sourceImage.empty()) {
        sourceImage = Mat(360, 640, CV_8UC3, Scalar(55, 55, 55));

        putText(sourceImage,
            "Image could not be found",
            Point(120, 170),
            FONT_HERSHEY_SIMPLEX,
            0.8,
            Scalar(255, 255, 255),
            2);

        putText(sourceImage,
            sample.filename,
            Point(40, 220),
            FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar(220, 220, 220),
            1);
    }

    // Resize very large images so the result window fits on the screen.
    const int MAX_IMAGE_WIDTH = 850;
    const int MAX_IMAGE_HEIGHT = 520;

    double scaleX =
        (double)MAX_IMAGE_WIDTH / sourceImage.cols;
    double scaleY =
        (double)MAX_IMAGE_HEIGHT / sourceImage.rows;

    double scale = min(1.0, min(scaleX, scaleY));

    Mat displayImage;
    resize(sourceImage, displayImage, Size(),
        scale, scale, INTER_AREA);

    const int TEXT_AREA_HEIGHT = 230;
    const int MIN_CANVAS_WIDTH = 1100;

    int canvasWidth = max(MIN_CANVAS_WIDTH, displayImage.cols);
    int canvasHeight = displayImage.rows + TEXT_AREA_HEIGHT;

    Mat resultImage(
        canvasHeight,
        canvasWidth,
        CV_8UC3,
        Scalar(35, 35, 35));

    // Centre the traffic-sign image horizontally.
    int imageX = (canvasWidth - displayImage.cols) / 2;

    displayImage.copyTo(
        resultImage(
            Rect(imageX, 0,
                displayImage.cols,
                displayImage.rows)));

    int textY = displayImage.rows + 32;

    Scalar white(255, 255, 255);
    Scalar green(0, 220, 0);
    Scalar red(0, 0, 255);
    Scalar yellow(0, 255, 255);

    string shownFilename =
        fs::path(sample.filename).filename().string();

    putText(resultImage,
        "Image: " + shownFilename,
        Point(20, textY),
        FONT_HERSHEY_SIMPLEX,
        0.60,
        white,
        1);

    textY += 38;

    putText(resultImage,
        "Correct sign: " + actualClass,
        Point(20, textY),
        FONT_HERSHEY_SIMPLEX,
        0.65,
        yellow,
        2);

    textY += 42;

    bool knnCorrect = (knnClass == actualClass);
    bool annCorrect = (annClass == actualClass);
    bool bayesCorrect = (bayesClass == actualClass);

    putText(resultImage,
        "KNN predicted: " + knnClass +
        (knnCorrect ? "  - CORRECT" : "  - WRONG"),
        Point(20, textY),
        FONT_HERSHEY_SIMPLEX,
        0.62,
        knnCorrect ? green : red,
        2);

    textY += 38;

    putText(resultImage,
        "ANN predicted: " + annClass +
        (annCorrect ? "  - CORRECT" : "  - WRONG"),
        Point(20, textY),
        FONT_HERSHEY_SIMPLEX,
        0.62,
        annCorrect ? green : red,
        2);

    textY += 38;

    putText(resultImage,
        "Naive Bayes predicted: " + bayesClass +
        (bayesCorrect ? "  - CORRECT" : "  - WRONG"),
        Point(20, textY),
        FONT_HERSHEY_SIMPLEX,
        0.62,
        bayesCorrect ? green : red,
        2);

    return resultImage;
}

// Reads one column name per line - used to load the exact feature set the
// Python side's SelectKBest chose (shared_selected_features.txt), so this
// evaluation compares classifiers on identical information rather than
// different feature engineering.
vector<string> readColumnList(const string& path) {
    vector<string> cols;
    ifstream in(path);
    if (!in.is_open()) return cols;
    string line;
    while (getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) cols.push_back(line);
    }
    return cols;
}

// ============================================================================
// Helper 8: external-training evaluation
//
// Train KNN/ANN/Naive Bayes on TSRD_external only, then predict on the
// original 84 project images - the same protocol as the Python side's
// Stage 0, so the two halves of Task 2 can be compared head-to-head on the
// exact same held-out test set. Reuses normalizeTrainingData() and
// evaluateModel() rather than reimplementing them, so this can't silently
// drift from the single-CSV evaluation's own logic. Also loads the same
// feature COLUMNS the Python side selected (shared_selected_features.txt),
// instead of this file's own 144-column HOG truncation, so this is a fair
// classifier-vs-classifier comparison, not a features-vs-features one.
// ============================================================================

void evaluateWithExternalTraining(const string& trainCsvPath, const string& testCsvPath,
    Mat& outKnnCM, Mat& outAnnCM, Mat& outBayesCM, Mat& outComparison) {
    if (!fs::exists(trainCsvPath)) {
        cout << "\nExternal-training evaluation skipped: " << trainCsvPath << " not found.\n";
        return;
    }

    vector<string> featureColumns = readColumnList("shared_selected_features.txt");
    if (featureColumns.empty()) {
        cout << "\nshared_selected_features.txt not found or empty - falling back to "
             << "this file's own " << HOG_FEATURES << "-column HOG selection "
             << "(run task2_combined.py's Stage 0 first to generate a fair shared "
             << "feature list).\n";
        featureColumns = defaultHogColumns();
    }
    int numFeatures = (int)featureColumns.size();
    cout << "\nUsing " << numFeatures << " shared feature column(s) for this evaluation.\n";

    vector<Sample> trainSamples = loadTask1CSV(trainCsvPath, featureColumns);
    vector<Sample> testSamples = loadTask1CSV(testCsvPath, featureColumns);
    if (trainSamples.empty() || testSamples.empty()) {
        cout << "\nExternal-training evaluation skipped: could not load data.\n";
        return;
    }

    // Real sign-type labels, same file/philosophy as main()'s single-CSV
    // evaluation and the Python side: run task2_combined.py first so
    // sign_type_labels.csv already has every TSRD_external filename covered
    // (it pre-fills those automatically). Some TSRD_external images have a
    // SignID outside our mapped classes and are deliberately left blank
    // there (same ones the Python side drops) - rather than hard-failing
    // this optional comparison over that, drop those specific samples and
    // continue with whatever is labelled. quiet=true because this is
    // expected, not an error - main()'s single-CSV evaluation below still
    // hard-fails with the full explanation if *our own 84* are unlabelled.
    const string signTypeLabelPath = "sign_type_labels.csv";
    size_t trainBefore = trainSamples.size(), testBefore = testSamples.size();
    loadSignTypeLabels(signTypeLabelPath, trainSamples, /*quiet=*/true);
    loadSignTypeLabels(signTypeLabelPath, testSamples, /*quiet=*/true);
    trainSamples.erase(remove_if(trainSamples.begin(), trainSamples.end(),
        [](const Sample& s) { return s.signType.empty(); }), trainSamples.end());
    testSamples.erase(remove_if(testSamples.begin(), testSamples.end(),
        [](const Sample& s) { return s.signType.empty(); }), testSamples.end());
    if (trainBefore - trainSamples.size() > 0 || testBefore - testSamples.size() > 0)
        cout << "\nDropped " << (trainBefore - trainSamples.size()) << " training + "
             << (testBefore - testSamples.size()) << " test image(s) with no SignType "
             << "in " << signTypeLabelPath << " (classes outside our mapped set - "
             << "expected, matches the Python side's own exclusion).\n";
    if (trainSamples.empty() || testSamples.empty()) {
        cout << "\nExternal-training evaluation skipped: no samples with a "
             << "SignType in " << signTypeLabelPath << ".\n";
        return;
    }

    cout << "\n================================================================\n";
    cout << "EXTERNAL-TRAINING EVALUATION (train on TSRD_external, test on our 84)\n";
    cout << "================================================================\n";
    cout << "Training images (external): " << trainSamples.size() << "\n";
    cout << "Test images (ours)        : " << testSamples.size() << "\n";

    // Class universe = every sign type seen in either set, so a class that
    // only appears in the test set (zero external examples) still gets a
    // slot - it will simply always be misclassified, same as the Python side.
    set<string> classSet, trainClasses;
    for (auto& s : trainSamples) { classSet.insert(s.signType); trainClasses.insert(s.signType); }
    for (auto& s : testSamples) classSet.insert(s.signType);
    vector<string> classNames(classSet.begin(), classSet.end());
    int numberOfClasses = (int)classNames.size();
    map<string, int> classToNumber;
    for (int i = 0; i < numberOfClasses; i++) classToNumber[classNames[i]] = i;

    int zeroExampleClasses = 0;
    for (auto& c : classNames) if (!trainClasses.count(c)) zeroExampleClasses++;
    if (zeroExampleClasses)
        cout << zeroExampleClasses << " class(es) have zero external training examples "
             << "(always misclassified - same caveat as the Python side).\n";

    Mat trainingData((int)trainSamples.size(), numFeatures, CV_32F);
    Mat trainingLabels((int)trainSamples.size(), 1, CV_32S);
    for (int r = 0; r < (int)trainSamples.size(); r++) {
        trainingLabels.at<int>(r, 0) = classToNumber[trainSamples[r].signType];
        for (int c = 0; c < numFeatures; c++)
            trainingData.at<float>(r, c) = trainSamples[r].hog[c];
    }

    Mat verificationData((int)testSamples.size(), numFeatures, CV_32F);
    Mat verificationLabels((int)testSamples.size(), 1, CV_32S);
    for (int r = 0; r < (int)testSamples.size(); r++) {
        verificationLabels.at<int>(r, 0) = classToNumber[testSamples[r].signType];
        for (int c = 0; c < numFeatures; c++)
            verificationData.at<float>(r, c) = testSamples[r].hog[c];
    }

    vector<double> mean, sigma;
    normalizeTrainingData(trainingData, verificationData, mean, sigma);

    // ----- Train (identical setup to the single-CSV evaluation above) -----
    // Progress messages below: with thousands of external training images,
    // ANN backprop in particular can take several minutes in an unoptimized
    // Debug build with zero visual feedback otherwise - easy to mistake for
    // a hang. A Release build (cmake --build ... --config Release) trains
    // much faster if this matters for repeated runs.
    cout << "Training KNN...\n";
    Ptr<KNearest> knn = KNearest::create();
    knn->setIsClassifier(true);
    knn->setDefaultK(1);
    knn->train(trainingData, ROW_SAMPLE, trainingLabels);
    cout << "Training ANN (this is the slow one - can take several minutes "
         << "on " << trainingData.rows << " images in a Debug build)...\n";

    Mat annTrainingLabels = Mat::zeros(trainingData.rows, numberOfClasses, CV_32F);
    for (int r = 0; r < trainingLabels.rows; r++)
        annTrainingLabels.at<float>(r, trainingLabels.at<int>(r, 0)) = 1.0f;

    Ptr<ANN_MLP> ann = ANN_MLP::create();
    Mat layerSizes = (Mat_<int>(1, 3) << numFeatures, 64, numberOfClasses);
    ann->setLayerSizes(layerSizes);
    ann->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
    ann->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.1);
    ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 2000, 1e-6));
    theRNG().state = 12345;
    ann->train(trainingData, ROW_SAMPLE, annTrainingLabels);
    cout << "ANN trained. Training Naive Bayes...\n";

    // See NAIVE_BAYES_PCA_DIMS above - fit PCA on training data only (no
    // leakage), then project both train and test through it just for the
    // Naive Bayes path. KNN/ANN don't estimate a covariance matrix, so they
    // aren't affected by this and keep using the full numFeatures-dim features.
    PCA bayesPCA(trainingData, Mat(), PCA::DATA_AS_ROW, NAIVE_BAYES_PCA_DIMS);
    Mat trainingDataBayes = bayesPCA.project(trainingData);
    Mat verificationDataBayes = bayesPCA.project(verificationData);

    Ptr<NormalBayesClassifier> bayes = NormalBayesClassifier::create();
    bayes->train(trainingDataBayes, ROW_SAMPLE, trainingLabels);
    cout << "All 3 classifiers trained. Predicting on the 84 test images...\n";

    // ----- Predict on the 84 (genuinely held out - never trained on) -----
    vector<int> truth, predKNN, predANN, predBayes;
    for (int i = 0; i < verificationData.rows; i++) {
        truth.push_back(verificationLabels.at<int>(i, 0));

        Mat knnResult;
        float knnResponse = knn->findNearest(verificationData.row(i), 1, knnResult);
        predKNN.push_back(cvRound(knnResponse));

        Mat annOutput;
        ann->predict(verificationData.row(i), annOutput);
        Point maxLoc;
        minMaxLoc(annOutput, nullptr, nullptr, nullptr, &maxLoc);
        predANN.push_back(maxLoc.x);

        float bayesResponse = bayes->predict(verificationDataBayes.row(i));
        predBayes.push_back(cvRound(bayesResponse));
    }

    // Same per-image popup viewer as main()'s single-CSV mode (Section 9B),
    // but over every one of the test images here - unlike that mode, this
    // evaluation has no MIN_SAMPLES_PER_CLASS cutoff, so it covers all of
    // them, not just the classes with >=3 examples.
    fs::create_directories("task2_external_prediction_visualizations");
    bool showExternalIndividualWindows = true;
    for (int i = 0; i < (int)truth.size(); i++) {
        const Sample& sample = testSamples[i];

        Mat predictionView = drawIndividualPrediction(
            sample,
            classNames[truth[i]],
            classNames[predKNN[i]],
            classNames[predANN[i]],
            classNames[predBayes[i]]
        );

        string outputName = fs::path(sample.filename).filename().string();
        imwrite("task2_external_prediction_visualizations/Result_" + outputName, predictionView);

        if (showExternalIndividualWindows) {
            string windowName = "External Verification " + to_string(i + 1) +
                " of " + to_string(truth.size());

            imshow(windowName, predictionView);

            cout << "\nShowing external verification image "
                << i + 1 << " of " << truth.size() << endl;
            cout << "Press any key for the next image." << endl;
            cout << "Press ESC to stop individual-image display." << endl;

            int key = waitKey(0);
            destroyWindow(windowName);

            if (key == 27)
                showExternalIndividualWindows = false;
        }
    }

    EvaluationResult knnResult = evaluateModel("KNN", truth, predKNN, numberOfClasses);
    EvaluationResult annResult = evaluateModel("ANN", truth, predANN, numberOfClasses);
    EvaluationResult bayesResult = evaluateModel("Naive Bayes", truth, predBayes, numberOfClasses);
    vector<EvaluationResult> allResults = { knnResult, annResult, bayesResult };

    cout << fixed << setprecision(4);
    cout << "\nClassifier     Accuracy    Precision   Recall      F1-score    \n";
    cout << string(65, '-') << "\n";
    for (const auto& r : allResults) {
        cout << left << setw(15) << r.modelName << setw(12) << r.accuracy
             << setw(12) << r.precision << setw(12) << r.recall << setw(12) << r.f1 << "\n";
    }

    // Distinct filenames so this never collides with the single-CSV mode's
    // own task2_*.csv/png output, or with the Python side's task2_external_*.
    ofstream metricsFile("task2_cpp_external_metrics.csv");
    if (metricsFile.is_open()) {
        metricsFile << "Classifier,Accuracy,Precision,Recall,F1Score\n";
        for (const auto& r : allResults)
            metricsFile << r.modelName << "," << r.accuracy << "," << r.precision << ","
                        << r.recall << "," << r.f1 << "\n";
    }

    ofstream predictionFile("task2_cpp_external_predictions.csv");
    if (predictionFile.is_open()) {
        predictionFile << "Filename,Truth,KNN,ANN,NaiveBayes\n";
        for (int i = 0; i < (int)truth.size(); i++) {
            predictionFile << testSamples[i].filename << "," << classNames[truth[i]] << ","
                           << classNames[predKNN[i]] << "," << classNames[predANN[i]] << ","
                           << classNames[predBayes[i]] << "\n";
        }
    }

    outKnnCM = drawConfusionMatrix(knnResult, classNames);
    outAnnCM = drawConfusionMatrix(annResult, classNames);
    outBayesCM = drawConfusionMatrix(bayesResult, classNames);
    outComparison = drawModelComparison(allResults,
        "Official result - trained on TSRD external dataset, tested on all 84 images");

    imwrite("task2_cpp_external_knn_confusion_matrix.png", outKnnCM);
    imwrite("task2_cpp_external_ann_confusion_matrix.png", outAnnCM);
    imwrite("task2_cpp_external_bayes_confusion_matrix.png", outBayesCM);
    imwrite("task2_cpp_external_classifier_comparison.png", outComparison);

    cout << "\nFiles written:\n"
         << "  task2_cpp_external_metrics.csv\n"
         << "  task2_cpp_external_predictions.csv\n"
         << "  task2_cpp_external_knn_confusion_matrix.png\n"
         << "  task2_cpp_external_ann_confusion_matrix.png\n"
         << "  task2_cpp_external_bayes_confusion_matrix.png\n"
         << "  task2_cpp_external_classifier_comparison.png\n"
         << "  task2_external_prediction_visualizations/Result_<filename>\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    const string csvPath = "task1_combined_features.csv";
    const string signTypeLabelPath = "sign_type_labels.csv";

    // We hold out one image per class for verification.
    // A class therefore needs at least 3 images so that Naive Bayes still has
    // at least 2 training examples for estimating its statistics.
    const int MIN_SAMPLES_PER_CLASS = 3;

    // ========================================================================
    // Section 1: Read the Task 1 feature data
    // ========================================================================

    vector<Sample> samples = loadTask1CSV(csvPath);

    if (samples.empty()) {
        cout << "No data loaded from " << csvPath << endl;
        return -1;
    }

    cout << "Task 1 CSV loaded successfully." << endl;
    cout << "Total images in CSV: " << samples.size() << endl;

    // Load the REAL sign type for each image (see loadSignTypeLabels() /
    // Helper 3B above for why SignID alone isn't the class).
    if (!loadSignTypeLabels(signTypeLabelPath, samples))
        return -1;

    cout << "Traffic-sign type labels loaded successfully." << endl;

    // ========================================================================
    // Section 2: Select the best Task 1 feature
    //
    // HOG is used because traffic-sign recognition depends strongly on the
    // inner symbol / digit / arrow of the traffic sign. Colour and outer
    // shape are useful for segmentation, but different sign types can share
    // them.
    // ========================================================================

    cout << "\nSelected feature for Task 2: HOG" << endl;
    cout << "Number of HOG features       : " << HOG_FEATURES << endl;

    // Group the images according to their REAL traffic-sign type.
    map<string, vector<int>> groupedRows;
    for (int i = 0; i < (int)samples.size(); i++)
        groupedRows[samples[i].signType].push_back(i);

    cout << "Total sign type classes      : " << groupedRows.size() << endl;

    // ========================================================================
    // Section 3: Keep classes that can be evaluated fairly
    // ========================================================================

    vector<string> classNames;
    vector<int> excludedCounts;

    cout << "\nClasses used for quantitative evaluation:" << endl;
    for (const auto& item : groupedRows) {
        if ((int)item.second.size() >= MIN_SAMPLES_PER_CLASS) {
            classNames.push_back(item.first);
            cout << "  " << item.first << " : " << item.second.size() << " images" << endl;
        }
    }

    cout << "\nClasses not used in the train/verification experiment" << endl;
    cout << "(fewer than " << MIN_SAMPLES_PER_CLASS << " images):" << endl;
    for (const auto& item : groupedRows) {
        if ((int)item.second.size() < MIN_SAMPLES_PER_CLASS) {
            cout << "  " << item.first << " : " << item.second.size() << " image(s)" << endl;
        }
    }

    if (classNames.size() < 2) {
        cout << "Not enough classes for classification." << endl;
        return -1;
    }

    int numberOfClasses = (int)classNames.size();

    // Map each real sign type to 0,1,2,... for OpenCV ML
    map<string, int> classToNumber;
    for (int i = 0; i < numberOfClasses; i++)
        classToNumber[classNames[i]] = i;

    // ========================================================================
    // Section 4: Set up training data and verification data
    //
    // One image from every usable sign type is reserved for verification.
    // The remaining images of that sign type are used for training.
    // A fixed random seed is used so that every classifier sees exactly the
    // same train/verification split every time the program is run.
    // ========================================================================

    vector<int> trainingRows;
    vector<int> verificationRows;

    mt19937 rng(42);

    for (const string& signType : classNames) {
        vector<int> rows = groupedRows[signType];
        shuffle(rows.begin(), rows.end(), rng);

        verificationRows.push_back(rows[0]);
        for (int i = 1; i < (int)rows.size(); i++)
            trainingRows.push_back(rows[i]);
    }

    Mat trainingData((int)trainingRows.size(), HOG_FEATURES, CV_32F);
    Mat trainingLabels((int)trainingRows.size(), 1, CV_32S);

    Mat verificationData((int)verificationRows.size(), HOG_FEATURES, CV_32F);
    Mat verificationLabels((int)verificationRows.size(), 1, CV_32S);

    for (int r = 0; r < (int)trainingRows.size(); r++) {
        const Sample& s = samples[trainingRows[r]];
        trainingLabels.at<int>(r, 0) = classToNumber[s.signType];

        for (int c = 0; c < HOG_FEATURES; c++)
            trainingData.at<float>(r, c) = s.hog[c];
    }

    for (int r = 0; r < (int)verificationRows.size(); r++) {
        const Sample& s = samples[verificationRows[r]];
        verificationLabels.at<int>(r, 0) = classToNumber[s.signType];

        for (int c = 0; c < HOG_FEATURES; c++)
            verificationData.at<float>(r, c) = s.hog[c];
    }

    cout << "\nTraining images              : " << trainingData.rows << endl;
    cout << "Verification images          : " << verificationData.rows << endl;
    cout << "Classes evaluated            : " << numberOfClasses << endl;

    // ========================================================================
    // Section 5: Do normalization of the data
    // Xnew = (X - mean) / sigma
    // ========================================================================

    vector<double> mean, sigma;
    normalizeTrainingData(trainingData, verificationData, mean, sigma);

    // ========================================================================
    // Section 6: Train KNN
    //
    // K = 1 is used because this dataset has only a small number of training
    // images for each SignID after one image is held out for verification.
    // ========================================================================

    Ptr<KNearest> knn = KNearest::create();
    knn->setIsClassifier(true);
    knn->setDefaultK(1);
    knn->train(trainingData, ROW_SAMPLE, trainingLabels);

    // ========================================================================
    // Section 7: Train ANN
    // ========================================================================

    Mat annTrainingLabels = Mat::zeros(trainingData.rows, numberOfClasses, CV_32F);
    for (int r = 0; r < trainingLabels.rows; r++) {
        int classNo = trainingLabels.at<int>(r, 0);
        annTrainingLabels.at<float>(r, classNo) = 1.0f;
    }

    Ptr<ANN_MLP> ann = ANN_MLP::create();

    Mat layerSizes = (Mat_<int>(1, 3) << HOG_FEATURES, 64, numberOfClasses);
    ann->setLayerSizes(layerSizes);
    ann->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
    ann->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.1);
    ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS,
        2000, 1e-6));

    // Fix OpenCV random state to make the ANN experiment repeatable
    theRNG().state = 12345;
    ann->train(trainingData, ROW_SAMPLE, annTrainingLabels);

    // ========================================================================
    // Section 8: Train Naive Bayes
    // ========================================================================

    Ptr<NormalBayesClassifier> bayes = NormalBayesClassifier::create();
    bayes->train(trainingData, ROW_SAMPLE, trainingLabels);

    // ========================================================================
    // Section 9: Test all 3 classifiers using the SAME verification images
    // ========================================================================

    vector<int> truth;
    vector<int> predKNN;
    vector<int> predANN;
    vector<int> predBayes;

    cout << "\nVerification results" << endl;
    cout << left
        << setw(26) << "Filename"
        << setw(28) << "Correct sign"
        << setw(28) << "KNN"
        << setw(28) << "ANN"
        << setw(28) << "Bayes"
        << endl;
    cout << string(138, '-') << endl;

    for (int i = 0; i < verificationData.rows; i++) {
        int actual = verificationLabels.at<int>(i, 0);
        truth.push_back(actual);

        // ----- KNN prediction -----
        Mat knnResult;
        float knnResponse = knn->findNearest(verificationData.row(i), 1, knnResult);
        int knnClass = cvRound(knnResponse);
        predKNN.push_back(knnClass);

        // ----- ANN prediction -----
        Mat annOutput;
        ann->predict(verificationData.row(i), annOutput);

        Point maxLoc;
        minMaxLoc(annOutput, nullptr, nullptr, nullptr, &maxLoc);
        int annClass = maxLoc.x;
        predANN.push_back(annClass);

        // ----- Naive Bayes prediction -----
        float bayesResponse = bayes->predict(verificationData.row(i));
        int bayesClass = cvRound(bayesResponse);
        predBayes.push_back(bayesClass);

        const Sample& sample = samples[verificationRows[i]];

        cout << left
            << setw(26) << sample.filename
            << setw(28) << classNames[actual]
            << setw(28) << classNames[knnClass]
            << setw(28) << classNames[annClass]
            << setw(28) << classNames[bayesClass]
            << endl;
    }

    // ========================================================================
    // Section 9B: Show every verification image with classifier predictions
    //
    // Correct prediction = green text
    // Wrong prediction   = red text
    //
    // The correct sign type is shown in yellow for comparison.
    // Press any key to move to the next verification image.
    // Press ESC to stop displaying individual images. The program will still
    // continue calculating and showing the overall evaluation results.
    // ========================================================================

    fs::create_directories("task2_prediction_visualizations");

    // Off by default: this is the internal 11-class subset, not the
    // reported result (see drawModelComparison's subtitle below), so the
    // program skips straight to the 84-image official evaluation's popups
    // instead of pausing here first. Every image is still saved to
    // task2_prediction_visualizations/ for the report either way.
    bool showIndividualWindows = false;

    for (int i = 0; i < (int)truth.size(); i++) {
        const Sample& sample = samples[verificationRows[i]];

        Mat predictionView = drawIndividualPrediction(
            sample,
            classNames[truth[i]],
            classNames[predKNN[i]],
            classNames[predANN[i]],
            classNames[predBayes[i]]
        );

        // Save every annotated verification image for the report.
        string outputName =
            fs::path(sample.filename).filename().string();

        imwrite(
            "task2_prediction_visualizations/Result_" + outputName,
            predictionView
        );

        if (showIndividualWindows) {
            string windowName =
                "Verification " + to_string(i + 1) +
                " of " + to_string(truth.size());

            imshow(windowName, predictionView);

            cout << "\nShowing verification image "
                << i + 1 << " of " << truth.size() << endl;
            cout << "Press any key for the next image." << endl;
            cout << "Press ESC to stop individual-image display." << endl;

            int key = waitKey(0);
            destroyWindow(windowName);

            if (key == 27)
                showIndividualWindows = false;
        }
    }

    // ========================================================================
    // Section 10: Accuracy, precision, recall, F1-score and confusion matrix
    // ========================================================================

    EvaluationResult knnResult = evaluateModel("KNN", truth, predKNN, numberOfClasses);
    EvaluationResult annResult = evaluateModel("ANN", truth, predANN, numberOfClasses);
    EvaluationResult bayesResult = evaluateModel("Naive Bayes", truth, predBayes, numberOfClasses);

    vector<EvaluationResult> allResults = { knnResult, annResult, bayesResult };

    cout << fixed << setprecision(4);
    cout << "\n================ Task 2 Performance =================" << endl;
    cout << left
        << setw(15) << "Classifier"
        << setw(12) << "Accuracy"
        << setw(12) << "Precision"
        << setw(12) << "Recall"
        << setw(12) << "F1-score"
        << endl;
    cout << string(63, '-') << endl;

    for (const auto& r : allResults) {
        cout << left
            << setw(15) << r.modelName
            << setw(12) << r.accuracy
            << setw(12) << r.precision
            << setw(12) << r.recall
            << setw(12) << r.f1
            << endl;
    }
    cout << "======================================================" << endl;

    // Show per-class metrics too
    for (const auto& r : allResults) {
        cout << "\n" << r.modelName << " - per-class results" << endl;
        cout << left
            << setw(30) << "Sign type"
            << setw(12) << "Precision"
            << setw(12) << "Recall"
            << setw(12) << "F1-score"
            << endl;

        for (int c = 0; c < numberOfClasses; c++) {
            cout << left
                << setw(30) << classNames[c]
                << setw(12) << r.classPrecision[c]
                << setw(12) << r.classRecall[c]
                << setw(12) << r.classF1[c]
                << endl;
        }
    }

    // ========================================================================
    // Section 11: Save numerical results to CSV
    // ========================================================================

    ofstream metricsFile("task2_classifier_metrics.csv");
    if (metricsFile.is_open()) {
        metricsFile << "Classifier,Accuracy,Precision,Recall,F1Score\n";
        for (const auto& r : allResults) {
            metricsFile << r.modelName << ","
                << r.accuracy << ","
                << r.precision << ","
                << r.recall << ","
                << r.f1 << "\n";
        }
        metricsFile.close();
    }

    ofstream predictionFile("task2_predictions.csv");
    if (predictionFile.is_open()) {
        predictionFile << "Filename,Truth,KNN,ANN,NaiveBayes\n";
        for (int i = 0; i < (int)truth.size(); i++) {
            const Sample& sample = samples[verificationRows[i]];
            predictionFile << sample.filename << ","
                << classNames[truth[i]] << ","
                << classNames[predKNN[i]] << ","
                << classNames[predANN[i]] << ","
                << classNames[predBayes[i]] << "\n";
        }
        predictionFile.close();
    }

    // ========================================================================
    // Section 12: Visualization of results
    // ========================================================================

    Mat knnCM = drawConfusionMatrix(knnResult, classNames);
    Mat annCM = drawConfusionMatrix(annResult, classNames);
    Mat bayesCM = drawConfusionMatrix(bayesResult, classNames);
    Mat comparison = drawModelComparison(allResults,
        "Internal subset only (11 classes, self-trained) - NOT the reported result");

    imwrite("task2_knn_confusion_matrix.png", knnCM);
    imwrite("task2_ann_confusion_matrix.png", annCM);
    imwrite("task2_bayes_confusion_matrix.png", bayesCM);
    imwrite("task2_classifier_comparison.png", comparison);

    cout << "\nFiles written:" << endl;
    cout << "  task2_classifier_metrics.csv" << endl;
    cout << "  task2_predictions.csv" << endl;
    cout << "  task2_knn_confusion_matrix.png" << endl;
    cout << "  task2_ann_confusion_matrix.png" << endl;
    cout << "  task2_bayes_confusion_matrix.png" << endl;
    cout << "  task2_classifier_comparison.png" << endl;
    cout << "  task2_prediction_visualizations/Result_<filename>" << endl;

    // Second evaluation: same 3 classifiers, trained on TSRD_external instead
    // of the internal split above, tested on the same 84 project images the
    // Python side's Stage 0 uses - a genuine head-to-head between this
    // file's classifiers and task2_combined.py's. Runs BEFORE the windows
    // below are shown (not after) - it can take a while, and a GUI window
    // left open without a waitKey() to pump its messages shows as "Not
    // Responding" in Windows for however long that takes, even though
    // nothing is actually wrong. Showing the windows only once we're about
    // to block on the real waitKey() avoids that.
    Mat extKnnCM, extAnnCM, extBayesCM, extComparison;
    evaluateWithExternalTraining("TSRD_external/task1_combined_features.csv", csvPath,
        extKnnCM, extAnnCM, extBayesCM, extComparison);

    // Show the official (external-training) result as the final windows -
    // that is the number reported for the project, not the internal
    // 11-class subset above. Fall back to the internal charts only if the
    // external evaluation could not run (e.g. TSRD_external missing).
    if (!extComparison.empty()) {
        imshow("KNN Confusion Matrix (Official)", extKnnCM);
        imshow("ANN Confusion Matrix (Official)", extAnnCM);
        imshow("Naive Bayes Confusion Matrix (Official)", extBayesCM);
        imshow("Classifier Comparison (Official)", extComparison);
    } else {
        imshow("KNN Confusion Matrix", knnCM);
        imshow("ANN Confusion Matrix", annCM);
        imshow("Naive Bayes Confusion Matrix", bayesCM);
        imshow("Classifier Comparison", comparison);
    }

    cout << "\nPress any key on an OpenCV window to finish." << endl;
    waitKey(0);
    destroyAllWindows();

    return 0;
}
